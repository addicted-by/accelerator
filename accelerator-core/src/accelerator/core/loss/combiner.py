import importlib
from typing import Any, Optional, Union

import torch

from accelerator.core.transform import LossTransformManager
from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.typings import _DEVICE, ModelOutputType

from .base import LossWrapper, LossWrapperBase
from .errors import LossAPIException, LossCalculationError, LossConfigurationError
from .statistics import LossStatistics
from .validation import InputValidator, ValidationConfig, ValidationError

LOSS_TARGET_FIELD = "_loss_target_"


class LossReportFormatter:
    """Handles formatting of loss reports and tables."""

    def __init__(self):
        self._max_loss_name_width = 20
        self._max_value_name_width = 20
        self._max_target_name_width = 20
        self._max_prediction_name_width = 20
        self._max_coefficient_name_width = 20
        self._header = [
            "Loss Name",
            "prediction_key",
            "target_key",
            "Value",
            "Coefficient",
        ]

    def update_widths(self, loss_name: str, prediction_key: Optional[str] = None) -> None:
        """Update column widths based on content."""
        self._max_loss_name_width = max(self._max_loss_name_width, len(loss_name))
        if prediction_key:
            self._max_prediction_name_width = max(self._max_prediction_name_width, len(prediction_key))

    def format_row(
        self,
        loss_name: str,
        prediction_key: str,
        target_key: str,
        value: str,
        coeff: str,
    ) -> str:
        """Format a single row of the loss report table."""
        return (
            f"{loss_name.ljust(self._max_loss_name_width)} | "
            f"{prediction_key.center(self._max_prediction_name_width)} | "
            f"{target_key.center(self._max_target_name_width)} | "
            f"{value.center(self._max_value_name_width)} | "
            f"{coeff.center(self._max_coefficient_name_width)}|\n"
        )

    @property
    def total_width(self) -> int:
        """Calculate total width of the report table."""
        return (
            self._max_loss_name_width
            + self._max_prediction_name_width
            + self._max_target_name_width
            + self._max_value_name_width
            + self._max_coefficient_name_width
            + 12
        )

    @property
    def separator(self) -> str:
        """Generate separator line for the report table."""
        return "=" * self.total_width + "|\n"

    def generate_report(
        self,
        losses: list[LossWrapper],
        combiner_stats: LossStatistics,
        combiner_name: str,
        combiner_coefficient: float,
        tb_logger=None,
        step=None,
    ) -> str:
        """Generate complete loss report."""
        try:
            report_lines = [
                f"\n{'Loss Report'.center(self.total_width)}\n\n",
                self.format_row(*self._header),
                self.separator,
            ]

            for loss in losses:
                (loss_name, loss_value, coeff, prediction_key, target_key) = loss.logger_step(
                    tb_logger=tb_logger, step=step
                )

                value_str = f"{loss_value:2.6E}" if loss_value is not None else "None"
                report_line = self.format_row(loss_name, prediction_key, target_key, value_str, f"{coeff:.2f}")
                report_lines.append(report_line)

            report_lines.append(self.separator)

            accumulated = combiner_stats.accumulated_loss
            loss_value_to_show = "inf" if accumulated is None else f"{accumulated:2.6E}"
            total_loss_line = self.format_row(
                combiner_name,
                "-",
                "-",
                loss_value_to_show,
                f"{combiner_coefficient:.2f}",
            )
            report_lines.append(total_loss_line)

            return "".join(report_lines)

        except Exception as e:
            raise LossAPIException(f"Report generation failed: {str(e)}") from e


class LossFactory:
    """Factory for creating loss instances from configuration."""

    @staticmethod
    def validate_config(loss_name: str, loss_config: dict[str, Any]) -> None:
        """Validate individual loss configuration."""
        if LOSS_TARGET_FIELD not in loss_config:
            raise LossConfigurationError(
                f"Loss '{loss_name}' configuration missing required '{LOSS_TARGET_FIELD}' field"
            )

        if "prediction_key" not in loss_config or "target_key" not in loss_config:
            raise LossConfigurationError(
                f"Loss '{loss_name}' configuration missing required 'prediction_key' " f"or 'target_key' fields"
            )

    @staticmethod
    def import_loss_class(target_path: str, loss_name: str) -> type:
        """Import loss class from target path."""
        try:
            package, loss_class_name = target_path.rsplit(".", 1)
            loss_module = importlib.import_module(package)
            return getattr(loss_module, loss_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import loss class '{loss_class_name}' from '{package}' " f"for loss '{loss_name}': {str(e)}"
            ) from e

    @staticmethod
    def create_loss_instance(
        loss_class: type, loss_name: str, loss_config: dict[str, Any], device: _DEVICE
    ) -> LossWrapper:
        """Create a single loss instance from configuration."""
        try:
            loss_config.update(device=device, name=loss_name)
            return loss_class(**loss_config)
        except Exception as e:
            raise TypeError(f"Failed to instantiate loss '{loss_name}' with config {loss_config}: {str(e)}") from e


class LossCombiner(LossWrapperBase):
    """
    Combines multiple loss functions with centralized input validation and shared transform caching.

    This class orchestrates multiple loss functions, validates inputs once for all losses,
    provides shared transform caching, and aggregates loss calculation.

    Attributes:
        _device: Device for tensor operations
        _active_losses: List of instantiated loss wrappers
        _validation_config: Configuration for input validation
        _statistics: Loss statistics tracker
        _formatter: Report formatter
        transform_manager: Shared transform manager for caching across all losses
    """

    def __init__(
        self,
        losses: list[LossWrapper],
        device: Optional[_DEVICE] = None,
        validation_config: Optional[ValidationConfig] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize LossCombiner with a list of loss instances.

        Args:
            losses: List of instantiated LossWrapper instances
            device: Device for tensor operations
            validation_config: Configuration for input validation
        """
        super().__init__(**kwargs)

        if not losses:
            raise LossConfigurationError("No losses provided")

        if not all(isinstance(loss, LossWrapper) for loss in losses):
            raise LossConfigurationError("All losses must be LossWrapper instances")

        self._device = device or distributed_state.device
        self._validation_config = validation_config or ValidationConfig()

        self._statistics = LossStatistics()
        self._formatter = LossReportFormatter()

        self._active_losses = losses
        self.transform_manager = LossTransformManager()

        self._setup_shared_transform_manager()
        self._update_formatter_widths()

    def _setup_shared_transform_manager(self) -> None:
        """Ensure all losses use the shared transform manager."""
        for loss in self._active_losses:
            if hasattr(loss, "_transform_manager"):
                loss._transform_manager = self.transform_manager

    def _update_formatter_widths(self) -> None:
        """Update formatter column widths based on loss names and keys."""
        for loss in self._active_losses:
            pred_key = getattr(loss, "_prediction_key", None)
            self._formatter.update_widths(loss.name, pred_key)

    @staticmethod
    def from_config(
        config: dict[str, Any],
        device: Optional[_DEVICE] = None,
        validation_config: Optional[ValidationConfig] = None,
        *args,
        **kwargs,
    ) -> "LossCombiner":
        """
        Create LossCombiner from configuration.

        Args:
            config: Loss configuration dictionary with 'active_losses' and 'loss_configs'
            device: Device for tensor operations
            validation_config: Configuration for input validation

        Returns:
            LossCombiner instance

        Raises:
            LossConfigurationError: If configuration is invalid
        """
        factory = LossFactory()
        transform_manager = LossTransformManager()

        LossCombiner._validate_config_structure(config)

        active_losses = []

        for loss_name in config["active_losses"]:
            if loss_name not in config["loss_configs"]:
                raise LossConfigurationError(f"Loss '{loss_name}' not found in loss_configs")

            loss_config = config["loss_configs"][loss_name].copy()

            factory.validate_config(loss_name, loss_config)

            loss_config["transform_manager"] = transform_manager

            target_path = loss_config[LOSS_TARGET_FIELD]
            loss_class = factory.import_loss_class(target_path, loss_name)

            loss_instance = factory.create_loss_instance(
                loss_class, loss_name, loss_config, device or distributed_state.device
            )

            active_losses.append(loss_instance)

        combiner = LossCombiner(
            losses=active_losses,
            device=device,
            validation_config=validation_config,
            *args,
            **kwargs,
        )

        return combiner

    @staticmethod
    def _validate_config_structure(config: dict[str, Any]) -> None:
        """Validate the overall configuration structure."""
        if "active_losses" not in config:
            raise LossConfigurationError("Config missing 'active_losses' key")

        if not config["active_losses"]:
            raise LossConfigurationError("No active losses specified")

    def validate_inputs(self, predictions: ModelOutputType, labels: ModelOutputType) -> None:
        """Validate inputs for all losses."""
        try:
            InputValidator.validate(predictions, labels, self._validation_config)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error: {str(e)}") from e

    def add_get_loss(self, predictions: ModelOutputType, labels: ModelOutputType, *args, **kwargs) -> torch.Tensor:
        """
        Calculate combined loss from all active losses with shared transform caching.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Combined loss tensor

        Raises:
            ValidationError: If input validation fails
            LossCalculationError: If loss calculation fails
        """
        try:
            self.validate_inputs(predictions, labels)

            result = None

            for loss in self._active_losses:
                try:
                    loss_value = loss.add_get_loss(
                        predictions,
                        labels,
                        transform_manager=self.transform_manager,
                        *args,
                        **kwargs,
                    )

                    if not isinstance(loss_value, torch.Tensor):
                        raise LossCalculationError(
                            f"Loss '{loss.name}' returned {type(loss_value)}, expected torch.Tensor"
                        )

                    if result is None:
                        result = loss_value
                    else:
                        result += loss_value

                except Exception as e:
                    raise LossCalculationError(f"Failed to calculate loss '{loss.name}': {str(e)}") from e

            if result is None:
                raise LossCalculationError("No active losses to calculate")

            self._statistics.update(result, 1)

            return result

        except (ValidationError, LossCalculationError):
            raise
        except Exception as e:
            raise LossCalculationError(f"Unexpected error in loss combination: {str(e)}") from e
        finally:
            self.transform_manager.clear_cache()

    def add_loss(self, loss: LossWrapper) -> None:
        """
        Add a new loss to the combiner.

        Args:
            loss: LossWrapper instance to add
        """
        if not isinstance(loss, LossWrapper):
            raise LossConfigurationError("Loss must be a LossWrapper instance")

        if hasattr(loss, "_transform_manager"):
            loss._transform_manager = self.transform_manager

        self._active_losses.append(loss)
        pred_key = getattr(loss, "_prediction_key", None)
        self._formatter.update_widths(loss.name, pred_key)

    def remove_loss(self, loss_name: str) -> bool:
        """
        Remove a loss by name.

        Args:
            loss_name: Name of the loss to remove

        Returns:
            True if loss was found and removed, False otherwise
        """
        for i, loss in enumerate(self._active_losses):
            if loss.name == loss_name:
                self._active_losses.pop(i)
                return True
        return False

    def get_loss(self, loss_name: str) -> Optional[LossWrapper]:
        """
        Get a loss by name.

        Args:
            loss_name: Name of the loss to retrieve

        Returns:
            LossWrapper instance or None if not found
        """
        for loss in self._active_losses:
            if loss.name == loss_name:
                return loss
        return None

    def clear(self) -> None:
        """Reset all internal state and clear individual losses."""
        self._statistics.clear()
        for loss in self._active_losses:
            loss.clear()
        self.transform_manager.clear_cache()

    @property
    def active_losses(self) -> list[LossWrapper]:
        """Get list of active losses."""
        return self._active_losses.copy()

    @property
    def loss_count(self) -> int:
        """Get number of active losses."""
        return len(self._active_losses)

    @property
    def accumulated_loss(self) -> Union[torch.Tensor, None]:
        """Get average accumulated loss."""
        return self._statistics.accumulated_loss

    @property
    def last_loss(self) -> Union[torch.Tensor, None]:
        """Get the most recent loss value."""
        return self._statistics.last_loss

    @property
    def transform_manager_stats(self) -> dict[str, int]:
        """Get transform cache statistics."""
        return {
            "pred_cache_size": len(self.transform_manager.pred_cache),
            "gt_cache_size": len(self.transform_manager.gt_cache),
            "joint_cache_size": len(self.transform_manager.joint_results_cache),
            "total_cache_size": (
                len(self.transform_manager.pred_cache)
                + len(self.transform_manager.gt_cache)
                + len(self.transform_manager.joint_results_cache)
            ),
        }

    def logger_step(self, tb_logger=None, step=None) -> str:
        """
        Generate loss report and log to tensorboard if provided.

        Args:
            tb_logger: Optional tensorboard logger
            step: Optional step number for logging

        Returns:
            Formatted loss report string

        Raises:
            LossAPIException: If logging fails
        """
        report = self._formatter.generate_report(
            self._active_losses,
            self._statistics,
            self.__class__.__name__,
            self.loss_coefficient,
            tb_logger,
            step,
        )

        if tb_logger is not None and step is not None:
            cache_stats = self.transform_manager_stats
            tb_logger.add_scalar("loss_cache/total_size", cache_stats["total_cache_size"], step)

        return report

    def __repr__(self) -> str:
        """String representation with configuration details."""
        lines = [
            "LossCombiner:",
            f"  Device: {self._device}",
            f"  Active losses: {len(self._active_losses)}",
            f"  Validation level: {self._validation_config.level.value}",
            f"  Samples processed: {self._statistics.num_samples}",
        ]

        if self._active_losses:
            lines.append("\nActive Losses:")
            for i, loss in enumerate(self._active_losses, 1):
                transform_info = f" [{loss.transform_count} transforms]" if loss.has_transforms else ""
                lines.append(
                    f"  {i}. {loss.name} (pred: {loss._prediction_key}, target: {loss._target_key}){transform_info}"
                )

        return "\n".join(lines)
