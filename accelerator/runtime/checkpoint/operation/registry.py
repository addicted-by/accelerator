from typing import Dict, Callable, Iterable, Union, List, Optional
from enum import Enum
from accelerator.utilities.base_registry import BaseRegistry
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)



class OperationType(Enum):
    PRE_LOAD_OPS = "pre_load_ops"
    CKPT_TRANSFORMS = "ckpt_transforms"
    POST_LOAD_OPS = "post_load_ops"


class OperationRegistry(BaseRegistry):
    """A registry for managing operation types used in model acceleration and training."""

    NOT_REGISTERED_MSG: str = (
        "Operation '{op_name}' not found in registry for '{op_type}'. "
        "Register it with 'registry.register_operation('{op_type}')' " 
        "or use an existing operation: {available}."
    )

    def __init__(self, enable_logging: bool = True):
        """Initialize the operation registry.

        Args:
            enable_logging: Whether to wrap operations with logging.
        """
        super().__init__(enable_logging=enable_logging)
        
        for op_type in OperationType:
            self._registry_types[op_type.value] = {}
            

    def register_operation(self, operation_type: Union[str, Iterable[str], OperationType, Iterable[OperationType]]):
        """Decorator for registering operations in the registry.
        
        Args:
            operation_type: The type(s) under which to register the operation.
            
        Returns:
            Decorator function that registers the decorated operation.
            
        Raises:
            TypeError: If operation_type is not a string, OperationType, or iterable of these.
            ValueError: If operation_type is not a valid operation type.
        """
        return self.register_object(operation_type)
        
    def add_operation(self, operation_type: str, func: Callable, name: Optional[str] = None) -> None:
        """Dynamically add an operation without a decorator.

        Args:
            operation_type: The type of operation (e.g., 'pre_load_ops').
            func: The function to register.
            name: Optional custom name; defaults to func.__name__.

        Raises:
            ValueError: If operation_type is invalid.
        """
        self.add_object(operation_type, func, name)
        
    def get_operation(self, operation_type: str, name: str) -> Callable:
        """Retrieve an operation by type and name.

        Args:
            operation_type: The type of operation.
            name: The name of the operation.

        Returns:
            The registered callable.

        Raises:
            KeyError: If the operation is not registered.
        """
        return self.get_object(operation_type, name)

    def list_operations(self, operation_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered operations.

        Args:
            operation_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping operation types to lists of operation names.
        """
        return self.list_objects(operation_type)

    def has_operation(self, operation_type: str, name: str) -> bool:
        """Check if an operation is registered.

        Args:
            operation_type: The type of operation.
            name: The name of the operation.

        Returns:
            True if the operation exists, False otherwise.
        """
        return self.has_object(operation_type, name)


# Singleton instance
registry = OperationRegistry()