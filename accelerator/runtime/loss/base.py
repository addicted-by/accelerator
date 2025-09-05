import abc
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch

from accelerator.runtime.transform import LossTransformManager
from accelerator.typings.base import BatchTensorType, ModelOutputType
from accelerator.utilities.logging import get_logger

if TYPE_CHECKING:
    from accelerator import Context

from .debug import DebugConfig, DebugManager
from .errors import (LossAPIException, LossCalculationError,
                     LossConfigurationError)
from .statistics import GradientLogger, LossStatistics

logger = get_logger(__name__)


class LossWrapperBase(abc.ABC):
    """ """

    def __init__(self, debug_config: Optional[DebugConfig] = None):
        self._loss_coefficient: float = 1.0
        self._debug_manager = DebugManager(debug_config or DebugConfig())

    @abc.abstractmethod
    def add_get_loss(
        self,
        predictions: ModelOutputType,
        labels: ModelOutputType,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """ """
        ...

    def set_loss_coefficient(self, value: float) -> None:
        """
        Sets the loss coefficient value.

        Args:
            value: New coefficient value

        Raises:
            LossConfigurationError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise LossConfigurationError(
                f"Loss coefficient must be numeric, got {type(value)}"
            )

        if value < 0:
            raise LossConfigurationError(
                f"Loss coefficient must be non-negative, got {value}"
            )

        self._loss_coefficient = float(value)

    @property
    def loss_coefficient(self) -> float:
        """Gets the current loss coefficient value."""
        return self._loss_coefficient

    @abc.abstractmethod
    def logger_step(
        self, tb_logger: Optional[Any] = None, step: Optional[int] = None
    ) -> Any:
        """
        Performs logging after loss calculation.

        Args:
            tb_logger: Optional tensorboard logger instance
            step: Optional step number for logging

        Returns:
            Logging output in implementation-specific format

        Raises:
            LossAPIException: If logging fails
        """
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clears the current state of the wrapper.

        This method should be called at the start of each new epoch to
        reset any accumulated state.
        """
        ...


class LossWrapper(LossWrapperBase, abc.ABC):
    """
    Abstract base class for implementing specific loss functions with transform support.

    This class provides a framework for implementing loss functions with support for
    batch-wise loss calculation, transform pipelines, and state management.

    Attributes:
        _cfg: Optional configuration dictionary
        _prediction_key: Key for accessing predictions in dict-based inputs
        _target_key: Key for accessing targets in dict-based inputs
        _loss_coefficient: Scaling factor for the loss value
        _name: Name identifier for the loss function
        _statistics: Loss statistics tracker
        _gradient_logger: Gradient statistics logger
        _transform_pipeline: Transform pipeline for this loss
    """

    def __init__(
        self,
        prediction_key: Optional[str] = None,
        target_key: Optional[str] = None,
        loss_coefficient: float = 1.0,
        name: Optional[str] = None,
        debug_config: Optional[DebugConfig] = None,
        transforms=None,
        **kwargs,
    ):
        super().__init__(debug_config)

        if loss_coefficient <= 0:
            raise LossConfigurationError(
                f"loss_coefficient must be positive, got {loss_coefficient}"
            )

        if (prediction_key is None) != (target_key is None):
            raise LossConfigurationError(
                "Both prediction_key and target_key must be provided together or both None"
            )

        self._cfg = kwargs
        self._prediction_key = prediction_key
        self._target_key = target_key
        self._loss_coefficient = loss_coefficient
        self._name: str = name or self.__class__.__name__

        self._statistics = LossStatistics()
        self._gradient_logger = GradientLogger(self._name)

        self._transform_pipeline = []

        if transforms:
            try:
                self._transform_pipeline = LossTransformManager._instantiate_transforms(
                    transforms
                )
            except Exception as e:
                raise LossConfigurationError(
                    f"Failed to create transform pipeline for loss '{self._name}': {str(e)}"
                ) from e

    @abc.abstractmethod
    def calculate_batch_loss(
        self,
        net_result: torch.Tensor,
        ground_truth: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculates the loss for a single batch.

        Args:
            net_result: Model predictions for the batch
            ground_truth: Ground truth labels for the batch
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Calculated loss value for the batch

        Raises:
            LossCalculationError: If loss calculation fails
        """
        ...

    def add_get_loss(
        self,
        predictions: ModelOutputType,
        labels: ModelOutputType,
        transform_manager: Optional[LossTransformManager] = None,
        inputs: Optional[BatchTensorType] = None,
        context: Optional["Context"] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Processes inputs, applies transforms, and calculates the loss.

        Args:
            predictions: Model predictions
            labels: Ground truth labels
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Calculated loss value

        Raises:
            LossCalculationError: If loss calculation fails
            LossConfigurationError: If configuration is invalid
        """
        try:
            if transform_manager and self._transform_pipeline:
                net_result, ground_truth, transforms_info = (
                    transform_manager.apply_transforms(
                        self._prediction_key,
                        self._target_key,
                        predictions,
                        labels,
                        self._transform_pipeline,
                        inputs=inputs,
                        context=context,
                        **kwargs,
                    )
                )
            else:
                net_result, ground_truth = self._extract_tensors(predictions, labels)
                transforms_info = []

            loss = self.calculate_batch_loss(
                net_result,
                ground_truth,
                transforms_info=transforms_info,
                context=context,
                inputs=inputs,
                transform_manager=transform_manager,
                *args,
                **kwargs,
            )

            if not isinstance(loss, torch.Tensor):
                raise LossCalculationError(
                    f"calculate_batch_loss must return torch.Tensor, got {type(loss)}"
                )

            if isinstance(net_result, (list, tuple)):
                batch_size = net_result[0].shape[0] if net_result[0].ndim > 0 else 1
            else:
                batch_size = net_result.shape[0] if net_result.ndim > 0 else 1

            self._statistics.update(loss, batch_size)
            self._debug_manager.save_tensor(
                loss, f"loss_{self._name}", self._statistics.num_samples
            )

            return self.loss_coefficient * loss

        except (LossCalculationError, LossConfigurationError):
            raise
        except Exception as e:
            raise LossCalculationError(
                f"Unexpected error in loss calculation for {self._name}: {str(e)}"
            ) from e

    def _extract_tensors(
        self, predictions: ModelOutputType, labels: ModelOutputType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract tensors from predictions and labels based on configuration."""
        if isinstance(predictions, dict):
            if self._prediction_key is None or self._target_key is None:
                raise LossConfigurationError(
                    f"prediction_key and target_key must be set for dict inputs in loss {self._name}"
                )

            if self._prediction_key not in predictions:
                raise LossCalculationError(
                    f"prediction_key '{self._prediction_key}' not found in predictions"
                )

            if self._target_key not in labels:
                raise LossCalculationError(
                    f"target_key '{self._target_key}' not found in labels"
                )

            return predictions[self._prediction_key], labels[self._target_key]
        else:
            return predictions, labels

    def log_gradients(self, value: torch.Tensor) -> None:
        """
        Logs gradient statistics for the given tensor.

        Args:
            value: Tensor containing gradients to log
        """
        batch_size = value.shape[0] if value.ndim > 0 else 1
        self._gradient_logger.log_gradients(value, batch_size)

    @property
    def name(self) -> str:
        return self._name

    @property
    def gradients_info(self) -> Dict[str, float]:
        """Get normalized gradient statistics."""
        return self._gradient_logger.get_statistics()

    @property
    def accumulated_loss(self) -> Union[torch.Tensor, None]:
        """Get average accumulated loss."""
        return self._statistics.accumulated_loss

    @property
    def last_loss(self) -> Union[torch.Tensor, None]:
        """Get the most recent loss value."""
        return self._statistics.last_loss

    @property
    def has_transforms(self) -> bool:
        """Check if this loss has any transforms configured."""
        return bool(self._transform_pipeline)

    @property
    def transform_count(self) -> int:
        """Get the number of transforms in the pipeline."""
        return len(self._transform_pipeline)

    def logger_step(self, tb_logger=None, step=None) -> tuple:
        """
        Perform logging step with error handling.

        Returns:
            Tuple of (name, accumulated_loss, coefficient, prediction_key, target_key)

        Raises:
            LossAPIException: If logging fails
        """
        try:
            if tb_logger is not None:
                if self.accumulated_loss is not None:
                    tb_logger.add_scalar(
                        f"avg_{self._name}", self.accumulated_loss, step
                    )

                for name, value in self.gradients_info.items():
                    tb_logger.add_scalar(name, value, step)

            return (
                self._name,
                self.accumulated_loss,
                self._loss_coefficient,
                str(self._prediction_key),
                str(self._target_key),
            )

        except Exception as e:
            raise LossAPIException(
                f"Logging failed for loss {self._name}: {str(e)}"
            ) from e

    def clear(self) -> None:
        """Resets all internal state variables."""
        self._statistics.clear()
        self._gradient_logger.clear()

    def __repr__(self) -> str:
        """String representation with clean formatting."""
        description = dedent(f"""
            Loss: {self._name}
            Prediction key: {self._prediction_key} || Target key: {self._target_key}
            Loss coefficient: {self._loss_coefficient}
            Samples processed: {self._statistics.num_samples}
            Transforms: {self.transform_count}
        """).strip()

        if self._cfg:
            description += "\nConfiguration:\n"
            for key, value in self._cfg.items():
                description += f"  {key}: {value}\n"

        if self._transform_pipeline:
            description += "\nTransform Pipeline:\n"
            for i, transform in enumerate(self._transform_pipeline, 1):
                description += f"  {i}. {transform.name}\n"

        return description


class LossAdapter(LossWrapper):
    """
    Adapter to make standard PyTorch loss modules compatible with LossWrapper interface.

    This implementation properly integrates with the parent class and maintains
    all functionality while providing a clean interface to PyTorch losses.
    """

    def __init__(
        self,
        loss_module: torch.nn.Module,
        prediction_key: Optional[str] = None,
        target_key: Optional[str] = None,
        loss_coefficient: float = 1.0,
        name: Optional[str] = None,
        debug_config: Optional[DebugConfig] = None,
        transform_manager=None,
        **kwargs,
    ):
        if name is None:
            name = f"{loss_module.__class__.__name__}_Adapter"

        super().__init__(
            prediction_key=prediction_key,
            target_key=target_key,
            loss_coefficient=loss_coefficient,
            name=name,
            debug_config=debug_config,
            transform_manager=transform_manager,
            **kwargs,
        )

        if not isinstance(loss_module, torch.nn.Module):
            raise LossConfigurationError(
                f"loss_module must be a torch.nn.Module, got {type(loss_module)}"
            )

        self.loss_module = loss_module

    def calculate_batch_loss(
        self,
        net_result: torch.Tensor,
        ground_truth: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate loss using the wrapped PyTorch loss module.

        Args:
            net_result: Model predictions for the batch
            ground_truth: Ground truth labels for the batch
            *args: Additional positional arguments passed to loss module
            **kwargs: Additional keyword arguments passed to loss module

        Returns:
            Calculated loss value for the batch

        Raises:
            LossCalculationError: If the underlying loss module fails
        """
        try:
            return self.loss_module(net_result, ground_truth, *args, **kwargs)
        except Exception as e:
            raise LossCalculationError(
                f"PyTorch loss module {self.loss_module.__class__.__name__} failed: {str(e)}"
            ) from e
