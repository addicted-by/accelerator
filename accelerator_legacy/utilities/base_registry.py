import functools
from typing import Dict, Callable, Iterable, Union, Any, List, Optional
from threading import Lock
from enum import Enum
from accelerator.utilities.logging import get_logger
from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.signatures import get_object_info


log = get_logger(__name__)


def log_fn_call(func: callable) -> callable:
    """Decorator to log function calls.
    
    Args:
        func: The function to wrap with logging.
        
    Returns:
        Wrapped function that logs when called.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log.info(f"Executing operation `{func.__qualname__}`")
        return func(*args, **kwargs)
    return wrapper


@APIDesc.developer(dev_info='Ryabykin Alexey r00926208')
@APIDesc.status(status_level='Internal use only')
class BaseRegistry:
    """Base registry class for managing registered objects.
    
    This class provides core functionality for registering, retrieving, and listing
    objects in a thread-safe manner. It serves as the foundation for specialized registries
    like OperationRegistry, LossRegistry, and AccelerationRegistry.
    """
    
    NOT_REGISTERED_MSG: str = (
        "Object '{obj_name}' not found in registry for '{reg_type}'. "
        "Register it with 'registry.register_object('{reg_type}')' " 
        "or use an existing object: {available}."
    )

    def __init__(self, enable_logging: bool = True):
        """Initialize the base registry.

        Args:
            enable_logging: Whether to wrap registered objects with logging.
        """
        self._enable_logging = enable_logging
        self._lock = Lock()
        
        self._registry_types: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def register_object(self, registry_type: Union[str, Iterable[str], Enum, Iterable[Enum]], name: Optional[str]=None):
        """Decorator for registering objects in the registry.
        
        Args:
            registry_type: The type(s) under which to register the object.
            
        Returns:
            Decorator function that registers the decorated object.
            
        Raises:
            TypeError: If registry_type is not a string, Enum, or iterable of these.
            ValueError: If registry_type is not a valid registry type.
        """
        def decorator(cls_or_func: Callable) -> Callable:
            if isinstance(registry_type, (str, Enum)):
                reg_types = [registry_type.value if isinstance(registry_type, Enum) else registry_type]
            elif isinstance(registry_type, (list, tuple)):
                reg_types = [reg.value if isinstance(reg, Enum) else reg for reg in registry_type]
            else:
                raise TypeError(f"registry_type must be str or Iterable, got {type(registry_type)}")

            setattr(cls_or_func, 'registry_type', registry_type.value if isinstance(registry_type, Enum) else registry_type)
            
            with self._lock:
                for reg_type in reg_types:
                    if reg_type not in self._registry_types:
                        # Auto-create registry type if it doesn't exist
                        self._registry_types[reg_type] = {}
                        
                    if name:
                        cls_or_func.__name__ = name
                    _name = cls_or_func.__name__
                    if _name in self._registry_types[reg_type]:
                        log.warning(f"Overwriting existing object '{_name}' in '{reg_type}'")
                    
                    self._registry_types[reg_type][_name] = log_fn_call(cls_or_func) if self._enable_logging else cls_or_func
            return cls_or_func
        return decorator

    def add_object(self, registry_type: str, func: Callable, name: Optional[str] = None) -> None:
        """Dynamically add an object without a decorator.

        Args:
            registry_type: The type of registry to add to.
            func: The function or class to register.
            name: Optional custom name; defaults to func.__name__.

        Raises:
            ValueError: If registry_type is invalid.
        """
        name = name or func.__name__
        with self._lock:
            if registry_type not in self._registry_types:
                self._registry_types[registry_type] = {}
                
            setattr(func, 'registry_type', registry_type.value if isinstance(registry_type, Enum) else registry_type)
            if name:
                func.__name__ = name
            _name = func.__name__
            if _name in self._registry_types[registry_type]:
                log.warning(f"Overwriting existing object '{_name}' in '{registry_type}'")
            self._registry_types[registry_type][_name] = log_fn_call(func) if self._enable_logging else func

    def get_object(self, registry_type: str, name: str) -> Callable:
        """Retrieve an object by type and name.

        Args:
            registry_type: The type of registry.
            name: The name of the object.

        Returns:
            The registered callable.

        Raises:
            KeyError: If the object is not registered.
        """
        if registry_type not in self._registry_types:
            raise ValueError(f"Invalid registry type '{registry_type}'")
        objects = self._registry_types[registry_type]
        if name not in objects:
            raise KeyError(
                self.NOT_REGISTERED_MSG.format(
                    obj_name=name,
                    reg_type=registry_type,
                    available=list(objects.keys())
                )
            )
        return objects[name]

    def list_objects(self, registry_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered objects.

        Args:
            registry_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping registry types to lists of object names.
        """
        if registry_type:
            if registry_type not in self._registry_types:
                raise ValueError(f"Invalid registry type '{registry_type}'")
            return {registry_type: list(self._registry_types[registry_type].keys())}
        return {reg_type: list(objs.keys()) for reg_type, objs in self._registry_types.items()}

    def has_object(self, registry_type: str, name: str) -> bool:
        """Check if an object is registered.

        Args:
            registry_type: The type of registry.
            name: The name of the object.

        Returns:
            True if the object exists, False otherwise.
        """
        return registry_type in self._registry_types and name in self._registry_types[registry_type]
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the registry.
        
        Shows all registered objects with their type (class/function), 
        parameters, and methods (for classes).
        """
        if not self._registry_types:
            return "BaseRegistry(empty)"
        
        lines = [f"{self.__class__.__name__}:"]
        
        for registry_type, objects in self._registry_types.items():
            lines.append(f"\n  [{registry_type}]:")
            
            if not objects:
                lines.append("    (no objects registered)")
                continue
                
            for name, obj in objects.items():
                # Get the original object (in case it's wrapped with logging decorator)
                original_obj = obj
                if hasattr(obj, '__wrapped__'):
                    original_obj = obj.__wrapped__
                
                obj_info = get_object_info(name, original_obj)
                lines.append(f"    {obj_info}")
        
        return "\n".join(lines)