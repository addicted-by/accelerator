"""Base registry module for managing registered objects."""

import functools
from collections.abc import Iterable
from enum import Enum
from threading import Lock
from typing import Callable, Optional, Union

from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.logging import get_logger
from accelerator.utilities.registry.domain import Domain
from accelerator.utilities.registry.registry_metadata import RegistrationMetadata
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


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Internal use only")
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

        self._registry_types: dict[str, dict[str, RegistrationMetadata]] = {}

    def register_object(
        self,
        registry_type: Union[str, Iterable[str], Enum, Iterable[Enum]],
        name: Optional[str] = None,
        domain: Domain = Domain.CROSS,
    ):
        """Decorator for registering objects in the registry.

        Args:
            registry_type: The type(s) under which to register the object.
            name: Optional custom name for the object.
            domain: The machine learning domain classification (default: Domain.CROSS).

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

            cls_or_func.registry_type = registry_type.value if isinstance(registry_type, Enum) else registry_type

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

                    # Apply logging wrapper to callable before storing in metadata
                    wrapped_obj = log_fn_call(cls_or_func) if self._enable_logging else cls_or_func

                    # Create and store metadata
                    metadata = RegistrationMetadata.create(
                        name=_name, registry_type=reg_type, domain=domain, obj=wrapped_obj
                    )
                    self._registry_types[reg_type][_name] = metadata
            return cls_or_func

        return decorator

    def add_object(
        self, registry_type: str, func: Callable, name: Optional[str] = None, domain: Domain = Domain.CROSS
    ) -> None:
        """Dynamically add an object without a decorator.

        Args:
            registry_type: The type of registry to add to.
            func: The function or class to register.
            name: Optional custom name; defaults to func.__name__.
            domain: The machine learning domain classification (default: Domain.CROSS).

        Raises:
            ValueError: If registry_type is invalid.

        """
        name = name or func.__name__
        with self._lock:
            if registry_type not in self._registry_types:
                self._registry_types[registry_type] = {}

            func.registry_type = registry_type.value if isinstance(registry_type, Enum) else registry_type
            if name:
                func.__name__ = name
            _name = func.__name__
            if _name in self._registry_types[registry_type]:
                log.warning(f"Overwriting existing object '{_name}' in '{registry_type}'")

            # Apply logging wrapper to callable before storing in metadata
            wrapped_obj = log_fn_call(func) if self._enable_logging else func

            # Create and store metadata
            metadata = RegistrationMetadata.create(
                name=_name, registry_type=registry_type, domain=domain, obj=wrapped_obj
            )
            self._registry_types[registry_type][_name] = metadata

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
                self.NOT_REGISTERED_MSG.format(obj_name=name, reg_type=registry_type, available=list(objects.keys()))
            )
        # Extract callable from metadata
        metadata = objects[name]
        return metadata.obj

    def list_objects(self, registry_type: Optional[str] = None) -> dict[str, list[str]]:
        """List all registered objects.

        Args:
            registry_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping registry types to lists of object names.

        """
        if registry_type:
            if registry_type not in self._registry_types:
                raise ValueError(f"Invalid registry type '{registry_type}'")
            # Extract object names from metadata
            return {registry_type: list(self._registry_types[registry_type].keys())}
        # Extract object names from metadata for all registry types
        return {reg_type: list(objs.keys()) for reg_type, objs in self._registry_types.items()}

    def has_object(self, registry_type: str, name: str) -> bool:
        """Check if an object is registered.

        Args:
            registry_type: The type of registry.
            name: The name of the object.

        Returns:
            True if the object exists, False otherwise.

        """
        # Check for object existence in metadata structure
        return registry_type in self._registry_types and name in self._registry_types[registry_type]

    def get_metadata(self, registry_type: str, name: str) -> RegistrationMetadata:
        """Retrieve registration metadata for a specific object.

        Args:
            registry_type: The type of registry.
            name: The name of the object.

        Returns:
            RegistrationMetadata for the specified object.

        Raises:
            KeyError: If the object is not registered.

        """
        with self._lock:
            if registry_type not in self._registry_types:
                raise KeyError(f"Registry type '{registry_type}' not found")
            if name not in self._registry_types[registry_type]:
                raise KeyError(f"Object '{name}' not found in registry for '{registry_type}'")
            return self._registry_types[registry_type][name]

    def list_objects_by_domain(
        self, domain: Union[Domain, list[Domain]], registry_type: Optional[str] = None
    ) -> dict[str, list[str]]:
        """List registered objects filtered by domain(s).

        Args:
            domain: Single Domain or list of Domains to filter by.
            registry_type: Optional specific registry type to filter; if None, search all.

        Returns:
            Dictionary mapping registry types to lists of object names matching the domain(s).

        """
        # Normalize domain to list
        domains = [domain] if isinstance(domain, Domain) else domain

        with self._lock:
            result: dict[str, list[str]] = {}

            # Determine which registry types to search
            reg_types_to_search = [registry_type] if registry_type else list(self._registry_types.keys())

            for reg_type in reg_types_to_search:
                if reg_type not in self._registry_types:
                    continue

                # Filter objects by domain
                matching_objects = [
                    name for name, metadata in self._registry_types[reg_type].items() if metadata.domain in domains
                ]

                if matching_objects:
                    result[reg_type] = matching_objects

            return result

    def get_domain_summary(self) -> dict[str, dict[str, int]]:
        """Get count of objects per domain for each registry type.

        Returns:
            Nested dictionary structure: {registry_type: {domain: count}}

        """
        with self._lock:
            summary: dict[str, dict[str, int]] = {}

            for reg_type, objects in self._registry_types.items():
                domain_counts: dict[str, int] = {}

                for metadata in objects.values():
                    domain_value = metadata.domain.value
                    domain_counts[domain_value] = domain_counts.get(domain_value, 0) + 1

                summary[reg_type] = domain_counts

            return summary

    def export_metadata(self, registry_type: Optional[str] = None) -> dict[str, list[dict]]:
        """Export all registration metadata as structured dictionary.

        Args:
            registry_type: Optional specific registry type to export; if None, export all.

        Returns:
            Dictionary mapping registry types to lists of metadata dictionaries.

        """
        with self._lock:
            result: dict[str, list[dict]] = {}

            # Determine which registry types to export
            reg_types_to_export = [registry_type] if registry_type else list(self._registry_types.keys())

            for reg_type in reg_types_to_export:
                if reg_type not in self._registry_types:
                    continue

                # Convert metadata to dictionaries
                metadata_list = [metadata.to_dict() for metadata in self._registry_types[reg_type].values()]

                result[reg_type] = metadata_list

            return result

    def __repr__(self) -> str:
        """Return a detailed string representation of the registry.

        Shows all registered objects grouped by registry type and domain,
        with their type (class/function), parameters, methods, and registration timestamp.
        """
        if not self._registry_types:
            return "BaseRegistry(empty)"

        lines = [f"{self.__class__.__name__}:"]

        for registry_type, objects in self._registry_types.items():
            lines.append(f"\n  [{registry_type}]:")

            if not objects:
                lines.append("    (no objects registered)")
                continue

            # Group objects by domain
            domain_groups: dict[Domain, list[tuple]] = {}
            for name, metadata in objects.items():
                if metadata.domain not in domain_groups:
                    domain_groups[metadata.domain] = []
                domain_groups[metadata.domain].append((name, metadata))

            # Display objects grouped by domain
            for domain in sorted(domain_groups.keys(), key=lambda d: d.value):
                lines.append(f"    Domain: {domain.value}")
                for name, metadata in domain_groups[domain]:
                    # Get the original object (in case it's wrapped with logging decorator)
                    original_obj = metadata.obj
                    if hasattr(original_obj, "__wrapped__"):
                        original_obj = original_obj.__wrapped__

                    obj_info = get_object_info(name, original_obj)
                    timestamp = metadata.registered_at
                    lines.append(f"      {obj_info} [registered: {timestamp}]")

        return "\n".join(lines)
