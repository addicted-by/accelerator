from typing import Dict, Callable, Iterable, Union, List, Optional
from enum import Enum


from accelerator.utilities.base_registry import BaseRegistry
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


class AccelerationType(Enum):
    """Enumeration of acceleration types supported by the registry."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    REPARAMETRIZATION = "reparametrization"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"


class AccelerationRegistry(BaseRegistry):
    """A registry for managing acceleration operations used in model optimization.
    
    This registry allows for registering, retrieving, and listing acceleration operations
    by their type and name. It provides a centralized way to manage different
    acceleration implementations across the `accelerator` framework.
    """

    NOT_REGISTERED_MSG: str = (
        "Acceleration operation '{op_name}' not found in registry for '{op_type}'. "
        "Register it with 'registry.register_acceleration('{op_type}')' " 
        "or use an existing operation: {available}."
    )

    def __init__(self, enable_logging: bool = True):
        """Initialize the acceleration registry.

        Args:
            enable_logging: Whether to wrap acceleration operations with logging.
        """
        super().__init__(enable_logging=enable_logging)
        
        for acc_type in AccelerationType:
            self._registry_types[acc_type.value] = {}

    def register_acceleration(self, acceleration_type: Union[str, Iterable[str], AccelerationType, Iterable[AccelerationType]]):
        """Decorator for registering acceleration operations in the registry.
        
        Args:
            acceleration_type: The type(s) under which to register the acceleration operation.
            
        Returns:
            Decorator function that registers the decorated acceleration operation.
            
        Raises:
            TypeError: If acceleration_type is not a string, AccelerationType, or iterable of these.
        """
        return self.register_object(acceleration_type)

    def add_acceleration(self, acceleration_type: str, func: Callable, name: Optional[str] = None) -> None:
        """Dynamically add an acceleration operation without a decorator.

        Args:
            acceleration_type: The type of acceleration (e.g., 'quantization').
            func: The acceleration operation or class to register.
            name: Optional custom name; defaults to func.__name__.
        """
        self.add_object(acceleration_type, func, name)

    def get_acceleration(self, acceleration_type: str, name: str) -> Callable:
        """Retrieve an acceleration operation by type and name.

        Args:
            acceleration_type: The type of acceleration.
            name: The name of the acceleration operation.

        Returns:
            The registered acceleration operation.

        Raises:
            KeyError: If the acceleration operation is not registered.
        """
        return self.get_object(acceleration_type, name)

    def list_accelerations(self, acceleration_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered acceleration operations.

        Args:
            acceleration_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping acceleration types to lists of acceleration operation names.
        """
        return self.list_objects(acceleration_type)

    def has_acceleration(self, acceleration_type: str, name: str) -> bool:
        """Check if an acceleration operation is registered.

        Args:
            acceleration_type: The type of acceleration.
            name: The name of the acceleration operation.

        Returns:
            True if the acceleration operation exists, False otherwise.
        """
        return self.has_object(acceleration_type, name)


acceleration_registry = AccelerationRegistry()