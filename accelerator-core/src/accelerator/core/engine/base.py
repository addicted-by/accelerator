"""
Base distributed backend interface for the accelerator runtime engine.

This module defines the abstract base class for distributed training backends
that can be used with the accelerator framework. The DistributedBackend class
provides a unified interface for different distributed computing frameworks
(such as PyTorch's DistributedDataParallel, Accelerate, etc.) to integrate
seamlessly with the accelerator runtime system.

The module enables:
    - Framework-agnostic distributed training support
    - Model, dataloader, and optimizer preparation for distributed execution
    - Process synchronization and communication primitives
    - Device management and tensor operations across multiple processes

Classes:
    DistributedBackend: Abstract base class defining the interface for
        distributed training backends.

Example:
    >>> class MyDistributedBackend(DistributedBackend):
    ...     def setup(self):
    ...         # Initialize distributed environment
    ...         pass
    ...
    ...     def prepare_model(self, model):
    ...         # Wrap model for distributed training
    ...         return distributed_model
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from accelerator.utilities.default_config import _DefaultConfig

T = TypeVar("T", bound="_DefaultConfig")


class DistributedBackend(ABC):
    """
    Abstract base class for distributed training backends.

    This class defines the interface that all distributed training backends must
    implement to work with the accelerator framework. It provides a unified API
    for different distributed computing frameworks (PyTorch DDP, Accelerate, etc.)
    to integrate seamlessly with the accelerator runtime system.

    The backend handles:
        - Distributed environment setup and cleanup
        - Model, dataloader, and optimizer preparation for distributed execution
        - Process synchronization and communication primitives
        - Device management and tensor operations across multiple processes

    Args:
        config (Dict[str, Any]): Configuration dictionary containing backend-specific
            settings and parameters.
        default_config (Type[T], optional): Default configuration class to use for
            creating the configuration object. Defaults to _DefaultConfig.

    Attributes:
        config: The configuration object created from the provided config dictionary
            using the default_config class.

    Example:
        >>> class MyBackend(DistributedBackend):
        ...     def setup(self):
        ...         # Initialize distributed environment
        ...         pass
        ...
        ...     def prepare_model(self, model):
        ...         # Wrap model for distributed training
        ...         return wrapped_model
        ...
        >>> backend = MyBackend({"world_size": 4})
        >>> backend.setup()
    """

    def __init__(self, config: dict[str, Any], default_config: type[T] = _DefaultConfig):
        self.config = default_config.create(config)

    @property
    @abstractmethod
    def device(self) -> torch.DeviceObjType:
        """
        Get the device used by this distributed backend.

        This property returns the torch device object that represents the
        hardware device (CPU, GPU, etc.) that this backend is configured
        to use for distributed training operations.

        Returns:
            torch.DeviceObjType: The torch device object representing the
                hardware device used by this backend.
        """
        pass

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def prepare_model(self, model: Module) -> Module:
        pass

    @abstractmethod
    def prepare_dataloader(self, loader: DataLoader) -> DataLoader:
        pass

    @abstractmethod
    def prepare_optimizer(self, optimizer) -> Any:
        pass

    @abstractmethod
    def rank(self) -> int:
        pass

    @abstractmethod
    def world_size(self) -> int:
        pass

    @abstractmethod
    def is_main_process(self) -> bool:
        pass

    @abstractmethod
    def barrier(self) -> None:
        pass

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        pass

    @abstractmethod
    def gather(self, tensor: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def spawn(self, fn: Callable, *args) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def broadcast(self, obj: Any, src: int = 0) -> Any:
        pass
