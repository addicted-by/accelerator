from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional, Union

from accelerator.utilities import _IS_DEBUG_LEVEL, APIDesc, BaseRegistry

from .base import BaseTransform


class TransformScopeType(Enum):
    """Enumeration of transformation types supported by the registry."""

    MODEL = "model"
    TENSOR = "tensor"
    CONTAINER = "container"
    STATE_DICT = "state_dict"


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="beta")
class TensorTransformsRegistry(BaseRegistry):
    """
    Registry for transform classes and cached instances.

    This registry provides a centralized way to register, retrieve, and manage
    tensor transformation operations. It ensures that transform instances with
    identical configurations are reused across the system.
    """

    NOT_REGISTERED_MSG: str = (
        "Transform '{op_name}' not found in registry for '{op_type}'. "
        "Register it with 'registry.register_transform('{op_type}')' "
        "or use an existing transform: {available}."
    )

    def __init__(self, enable_logging: bool = False):
        """Initialize the transform registry.

        Args:
            enable_logging: Whether to wrap transform operations with logging.
        """
        super().__init__(enable_logging=enable_logging)

        for transform_type in TransformScopeType:
            self._registry_types[transform_type.value] = {}

        self._instance_cache: list[BaseTransform] = []

    def register_transform(
        self, transform_type: Union[str, Iterable[str], TransformScopeType, Iterable[TransformScopeType]]
    ):
        """
        Decorator for registering transform classes in the registry.

        Args:
            transform_type: The type(s) under which to register the transform.

        Returns:
            Decorator function that registers the decorated transform class.

        Raises:
            TypeError: If transform_type is not a string, TransformType, or iterable of these.
        """
        return self.register_object(transform_type)

    def add_transform(
        self, transform_type: str, transform_cls: type[BaseTransform], name: Optional[str] = None
    ) -> None:
        """
        Dynamically add a transform class without a decorator.

        Args:
            transform_type: The type of transform (e.g., 'normalization').
            transform_cls: The transform class to register.
            name: Optional custom name; defaults to transform_cls.__name__.
        """
        self.add_object(transform_type, transform_cls, name)

    def get_transform(self, name: str, transform_type: Optional[str] = None) -> type[BaseTransform]:
        """
        Retrieve a transform class by name and optionally type.

        Args:
            name: The name of the transform class.
            transform_type: Optional type to narrow the search.

        Returns:
            The registered transform class.

        Raises:
            KeyError: If the transform is not registered.
        """
        if transform_type:
            return self.get_object(transform_type, name)

        for type_name in self._registry_types:
            if name in self._registry_types[type_name]:
                return self._registry_types[type_name][name]

        available = self.list_transforms()
        formatted_available = {t: list(available[t]) for t in available}
        raise KeyError(f"Transform '{name}' not found in registry. Available transforms: {formatted_available}")

    def list_transforms(self, transform_type: Optional[str] = None) -> dict[str, list[str]]:
        """
        List all registered transform classes.

        Args:
            transform_type: Optional specific type to list; if None, list all.

        Returns:
            Dictionary mapping transform types to lists of transform class names.
        """
        return self.list_objects(transform_type)

    def has_transform(self, name: str, transform_type: Optional[str] = None) -> bool:
        """
        Check if a transform class is registered.

        Args:
            name: The name of the transform class.
            transform_type: Optional type to narrow the search.

        Returns:
            True if the transform exists, False otherwise.
        """
        if transform_type:
            return self.has_object(transform_type, name)

        for type_name in self._registry_types:
            if name in self._registry_types[type_name]:
                return True
        return False

    def get_or_create_instance(
        self, name: str, config: Optional[dict[str, Any]] = None, transform_type: Optional[str] = None
    ) -> BaseTransform:
        """
        Get a cached transform instance or create a new one.

        Args:
            name: Name of the transform
            config: Configuration dictionary
            transform_type: Optional type to narrow the search

        Returns:
            A transform instance, either from cache or newly created
        """
        config = config or {}

        transform_cls = self.get_transform(name, transform_type)
        temp_instance = transform_cls(config)

        for cached_instance in self._instance_cache:
            if cached_instance == temp_instance:
                return cached_instance

        self._instance_cache.append(temp_instance)
        return temp_instance

    def instantiate_transform_pipeline(self, transforms_config: list) -> list[BaseTransform]:
        """
        Create a pipeline of transforms from configuration, reusing cached instances.

        Args:
            transforms_config: List of transform configurations

        Returns:
            List of transform instances

        Raises:
            ValueError: If an invalid transform entry is provided
        """
        transforms = []
        for entry in transforms_config:
            if isinstance(entry, str):
                key, config = entry, {}
            elif isinstance(entry, dict):
                key, config = next(iter(entry.items()))
            else:
                raise ValueError(f"Invalid transform entry: {entry}")

            transform = self.get_or_create_instance(key, config)
            transforms.append(transform)

        return transforms

    def clear_cache(self) -> None:
        """Clear the instance cache."""
        self._instance_cache.clear()


transforms_registry = TensorTransformsRegistry(enable_logging=_IS_DEBUG_LEVEL)
