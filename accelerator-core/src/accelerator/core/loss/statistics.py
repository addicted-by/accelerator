from collections import defaultdict
from typing import Union

import torch

from .errors import LossCalculationError


class GradientLogger:
    """Handles gradient statistics logging for loss functions."""

    def __init__(self, name: str):
        self._name = name
        self._gradients_info = defaultdict(float)
        self._num_samples = 0

    def log_gradients(self, value: torch.Tensor, batch_size: int) -> None:
        """
        Logs gradient statistics for the given tensor.

        Args:
            value: Tensor containing gradients to log
            batch_size: Size of the current batch

        Raises:
            LossCalculationError: If gradient logging fails
        """
        try:
            if not isinstance(value, torch.Tensor):
                raise LossCalculationError(f"Expected torch.Tensor for gradient logging, got {type(value)}")

            grad_mean = value.detach().abs().mean().item()
            grad_max = value.detach().abs().max().item()

            self._gradients_info[f"grad_mean_{self._name}"] += grad_mean * batch_size
            self._gradients_info[f"grad_max_{self._name}"] += grad_max * batch_size
            self._num_samples += batch_size

        except Exception as e:
            raise LossCalculationError(f"Gradient logging failed: {str(e)}") from e

    def get_statistics(self) -> dict[str, float]:
        """Get normalized gradient statistics."""
        if self._num_samples == 0:
            return {}
        return {key: value / self._num_samples for key, value in self._gradients_info.items()}

    def clear(self) -> None:
        """Reset gradient statistics."""
        self._gradients_info.clear()
        self._num_samples = 0


class LossStatistics:
    """Tracks loss accumulation and statistics."""

    def __init__(self):
        self._loss_sum = None
        self._num_samples = 0
        self._last_loss = None

    def update(self, loss: torch.Tensor, batch_size: int) -> None:
        """
        Update statistics with new loss value.

        Args:
            loss: Loss tensor for current batch
            batch_size: Size of the current batch
        """
        self._num_samples += batch_size
        loss_detached = loss.detach().cpu() * batch_size

        if self._loss_sum is None:
            self._loss_sum = loss_detached
        else:
            self._loss_sum += loss_detached

        self._last_loss = loss.detach().cpu()

    @property
    def accumulated_loss(self) -> Union[torch.Tensor, None]:
        """Get average accumulated loss."""
        if self._loss_sum is None or self._num_samples == 0:
            return None
        return self._loss_sum / self._num_samples

    @property
    def last_loss(self) -> Union[torch.Tensor, None]:
        """Get the most recent loss value."""
        return self._last_loss

    @property
    def num_samples(self) -> int:
        """Get the number of processed samples."""
        return self._num_samples

    def clear(self) -> None:
        """Reset all statistics."""
        self._num_samples = 0
        self._loss_sum = None
        self._last_loss = None
