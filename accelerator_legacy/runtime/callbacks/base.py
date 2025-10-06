import abc
from typing import Any, Dict, Optional, TYPE_CHECKING

from accelerator.typings.base import ConfigType
from accelerator.utilities.hashable import _HashableConfigMixin


if TYPE_CHECKING:
    from accelerator.runtime.context.context import Context



class BaseCallback(abc.ABC, _HashableConfigMixin):
    def __init__(
        self, 
        config: Optional[ConfigType] = None,
        *args,
        **kwargs
    ):
        self.config = config or {}
        self.config.update(kwargs)
        
    @property
    def priority(self) -> int:
        """Execution order (lower numbers first)"""
        return 100
    
    @property
    def critical(self) -> bool:
        """Whether to stop training if this callback fails"""
        return True

    def on_acceleration_begin(self, context: 'Context'):
        """Called at the beginning of acceleration: some components could be set here"""
        pass

    def on_acceleration_end(self, context: 'Context'):
        """Called at the end of acceleration: some components could be reset here"""
        pass
    
    # Training lifecycle hooks
    def on_train_begin(self, context: 'Context'):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, context: 'Context'):
        """Called at the end of training."""
        pass

    # Validation lifecycle hooks
    def on_val_begin(self, context: 'Context'):
        """Called at the beginning of training."""
        pass

    def on_val_end(self, context: 'Context'):
        """Called at the end of validation."""
        pass

    # Test lifecycle hooks
    def on_test_begin(self, context: 'Context'):
        """Called at the beginnning of test."""
        pass

    def on_test_end(self, context: 'Context'):
        """Called at the end of test."""
        pass

    # Epoch-level hooks
    def on_train_epoch_begin(self, context: 'Context'):
        """Called at the beginning of a training epoch."""
        pass
    
    def on_train_epoch_end(self, context: 'Context'):
        """Called at the end of a training epoch."""
        pass
    
    def on_val_epoch_begin(self, context: 'Context'):
        """Called at the beginning of a validation epoch."""
        pass
    
    def on_val_epoch_end(self, context: 'Context'):
        """Called at the end of a validation epoch."""
        pass
    
    def on_test_epoch_begin(self, context: 'Context'):
        """Called at the beginning of a test epoch."""
        pass
    
    def on_test_epoch_end(self, context: 'Context'):
        """Called at the end of a test epoch."""
        pass

    # Batch-level hooks
    def on_train_batch_begin(self, context: 'Context'):
        """Called at the beginning of a training batch."""
        pass

    def on_train_batch_end(self, context: 'Context'):
        """Called at the end of a training batch."""
        pass
    
    def on_val_batch_begin(self, context: 'Context'):
        """Called at the beginning of a validation batch."""
        pass

    def on_val_batch_end(self, context: 'Context'):
        """Called at the end of a validation batch."""
        pass
    
    def on_test_batch_begin(self, context: 'Context'):
        """Called at the beginning of a test batch."""
        pass

    def on_test_batch_end(self, context: 'Context'):
        """Called at the end of a test batch."""
        pass

    # Training-specific hooks
    def on_backward_begin(self, context: 'Context'):
        """Called at the beginning of the backward pass."""
        pass

    def on_backward_end(self, context: 'Context'):
        """Called at the end of the backward pass."""
        pass

    def on_optimizer_step_begin(self, context: 'Context'):
        """Called at the beginning of an optimizer step."""
        pass

    def on_optimizer_step_end(self, context: 'Context'):
        """Called at the end of an optimizer step."""
        pass

class BaseLoggerCallback(BaseCallback, abc.ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    @abc.abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value."""
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics dictionary."""
        pass

    @abc.abstractmethod
    def log_hyperparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        pass

    @abc.abstractmethod
    def log_image(self, name: str, image, step: int):
        """Log an image."""
        pass