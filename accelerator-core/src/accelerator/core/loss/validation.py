from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch

from accelerator.utilities.logging import get_logger

from .errors import ValidationError

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Levels of input validation."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


@dataclass
class ValidationConfig:
    """Configuration for input validation."""

    level: ValidationLevel = ValidationLevel.NONE
    check_device: bool = True
    check_dtype: bool = True
    allow_shape_mismatch: bool = False
    custom_validators: Optional[list[callable]] = None


class InputValidator:
    """Optimized input validation with configurable levels."""

    @staticmethod
    def validate(
        predictions: Union[torch.Tensor, dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, dict[str, torch.Tensor]],
        config: ValidationConfig = None,
    ) -> None:
        """Validate input predictions and labels.

        Args:
            predictions: Model predictions to validate
            labels: Ground truth labels to validate
            config: Validation configuration

        Raises:
            ValidationError: If inputs are invalid

        """
        config = config or ValidationConfig()

        if config.level == ValidationLevel.NONE:
            return

        try:
            InputValidator._validate_types(predictions, labels, config)

            if config.level == ValidationLevel.STRICT:
                InputValidator._validate_strict(predictions, labels, config)

            if config.custom_validators:
                for validator in config.custom_validators:
                    validator(predictions, labels)

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Validation failed: {str(e)}", context={"validation_level": config.level.value}
            ) from e

    @staticmethod
    def _validate_types(predictions, labels, config: ValidationConfig) -> None:
        """Basic type validation."""
        if isinstance(predictions, dict) != isinstance(labels, dict):
            raise ValidationError(
                "Predictions and labels must be of the same type",
                context={"predictions_type": type(predictions).__name__, "labels_type": type(labels).__name__},
            )

        if isinstance(predictions, dict):
            InputValidator._validate_dict_inputs(predictions, labels, config)
        else:
            InputValidator._validate_tensor_inputs(predictions, labels, config)

    @staticmethod
    def _validate_dict_inputs(predictions: dict, labels: dict, config: ValidationConfig) -> None:
        """Validate dictionary inputs."""
        for key, value in predictions.items():
            if not isinstance(value, torch.Tensor):
                raise ValidationError(f"All prediction values must be torch.Tensor, got {type(value)} for key '{key}'")

        for key, value in labels.items():
            if not isinstance(value, torch.Tensor):
                raise ValidationError(f"All label values must be torch.Tensor, got {type(value)} for key '{key}'")

        pred_keys = set(predictions.keys())
        label_keys = set(labels.keys())
        if pred_keys != label_keys:
            mismatched = pred_keys.symmetric_difference(label_keys)
            raise ValidationError(
                "Predictions and labels must have matching keys", context={"mismatched_keys": list(mismatched)}
            )

    @staticmethod
    def _validate_tensor_inputs(predictions: torch.Tensor, labels: torch.Tensor, config: ValidationConfig) -> None:
        """Validate tensor inputs."""
        if not isinstance(predictions, torch.Tensor):
            raise ValidationError(f"Predictions must be torch.Tensor, got {type(predictions)}")

        if not isinstance(labels, torch.Tensor):
            raise ValidationError(f"Labels must be torch.Tensor, got {type(labels)}")

    @staticmethod
    def _validate_strict(predictions, labels, config: ValidationConfig) -> None:
        """Strict validation including shapes and devices."""
        if isinstance(predictions, dict):
            for key in predictions.keys():
                pred_tensor = predictions[key]
                label_tensor = labels[key]
                InputValidator._validate_tensor_properties(pred_tensor, label_tensor, config, key)
        else:
            InputValidator._validate_tensor_properties(predictions, labels, config)

    @staticmethod
    def _validate_tensor_properties(
        pred: torch.Tensor, label: torch.Tensor, config: ValidationConfig, key: str = None
    ) -> None:
        """Validate tensor properties like shape, device, dtype."""
        context = {"key": key} if key else {}

        if not config.allow_shape_mismatch and pred.shape != label.shape:
            raise ValidationError(
                "Predictions and labels must have the same shape",
                context={**context, "pred_shape": tuple(pred.shape), "label_shape": tuple(label.shape)},
            )

        if config.check_device and pred.device != label.device:
            raise ValidationError(
                "Predictions and labels must be on the same device",
                context={**context, "pred_device": str(pred.device), "label_device": str(label.device)},
            )

        if config.check_dtype and pred.dtype != label.dtype:
            logger.warning(f"Predictions and labels have different dtypes: {pred.dtype} vs {label.dtype}")
