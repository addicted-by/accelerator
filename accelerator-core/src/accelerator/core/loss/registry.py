from functools import wraps
import inspect
from typing import Dict, Callable, Iterable, Union, List, Optional
from enum import Enum

import torch

from .base import LossAdapter, LossWrapper

from accelerator.utilities.base_registry import BaseRegistry
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


class LossType(Enum):
    """Enumeration of loss types supported by the registry."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    IMG2IMG = "image2image"
    REGULARIZATION = "regularization"
    CUSTOM = "custom"


def _adapt(obj, name: Optional[str] = None):
    if (inspect.isclass(obj) and issubclass(obj, LossWrapper)):
        return obj

    if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
        class _NNAdapter(LossAdapter):
            def __init__(self, *args, **kwargs):
                loss_module = obj(*args, **kwargs)
                super().__init__(loss_module=loss_module, **kwargs)
        _NNAdapter.__name__ = name or f"{obj.__name__}_Adapter"
        _NNAdapter.__original_cls__ = obj
        return _NNAdapter

    if inspect.isfunction(obj):
        class _FuncAdapter(LossWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._fn = obj
            def calculate_batch_loss(self, pred, gt, *a, **kw):
                return self._fn(pred, gt, *a, **kw, **self._cfg)
        _FuncAdapter.__name__ = name or f"{obj.__name__}_FuncAdapter"
        _FuncAdapter.__original_fn__ = obj
        return _FuncAdapter

    return obj



class LossRegistry(BaseRegistry):
    """A registry for managing loss functions used in model training.
    
    This registry allows for registering, retrieving, and listing loss functions
    by their type and name. It provides a centralized way to manage different
    loss implementations across the accelerator framework.
    """

    NOT_REGISTERED_MSG: str = (
        "Loss '{loss_name}' not found in registry for '{loss_type}'. "
        "Register it with 'registry.register_loss('{loss_type}')' " 
        "or use an existing loss: {available}."
    )

    def __init__(self, enable_logging: bool = True):
        """Initialize the loss registry.

        Args:
            enable_logging: Whether to wrap loss functions with logging.
        """
        super().__init__(enable_logging=enable_logging)
        
        # Initialize registry types with loss types
        for loss_type in LossType:
            self._registry_types[loss_type.value] = {}

    def register_loss(self, loss_type: Union[str, Iterable[str], LossType, Iterable[LossType]], name: Optional[str]=None):
        """Decorator for registering loss functions in the registry.
        
        Args:
            loss_type: The type(s) under which to register the loss function.
            
        Returns:
            Decorator function that registers the decorated loss function.
            
        Raises:
            TypeError: If loss_type is not a string, LossType, or iterable of these.
        """
        base_decorator = self.register_object(loss_type, name=name)
        
        @wraps(base_decorator)
        def decorator(cls_or_fn):
            return base_decorator(_adapt(cls_or_fn, name=name))

        return decorator

    def add_loss(self, loss_type: str, func: Callable, name: Optional[str] = None) -> None:
        """Dynamically add a loss function without a decorator.

        Args:
            loss_type: The type of loss (e.g., 'classification').
            func: The loss function or class to register.
            name: Optional custom name; defaults to func.__name__.
        """
        self.add_object(loss_type, _adapt(func, name=name), name)

    def get_loss(self, loss_type: str, name: str) -> Callable:
        """Retrieve a loss function by type and name.

        Args:
            loss_type: The type of loss.
            name: The name of the loss function.

        Returns:
            The registered loss function.

        Raises:
            KeyError: If the loss function is not registered.
        """
        return self.get_object(loss_type, name)

    def list_losses(self, loss_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered loss functions.

        Args:
            loss_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping loss types to lists of loss function names.
        """
        return self.list_objects(loss_type)

    def has_loss(self, loss_type: str, name: str) -> bool:
        """Check if a loss function is registered.

        Args:
            loss_type: The type of loss.
            name: The name of the loss function.

        Returns:
            True if the loss function exists, False otherwise.
        """
        return self.has_object(loss_type, name)


# Singleton instance
registry = LossRegistry(False)