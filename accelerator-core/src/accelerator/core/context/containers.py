"""Lifecycle-scoped containers for managing training data with automatic memory management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .reference_manager import ReferenceManager


class PathResolutionError(Exception):
    """Raised when a path cannot be resolved."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to resolve path '{path}': {reason}")


class BaseLifecycleContainer(ABC):
    """Base class for all lifecycle-scoped containers."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._sub_containers: List[str] = []
        self._initialize_sub_containers()

    @abstractmethod
    def _initialize_sub_containers(self) -> None:
        """Initialize sub-containers specific to this lifecycle scope."""

    def get_item(self, path: str) -> Any:
        """Get item using dot-notation path (e.g., 'input.rgb' or 'additional.key1.key2.data').

        Args:
            path: Dot-notation path to the item

        Returns:
            The value at the specified path

        Raises:
            PathResolutionError: If the path cannot be resolved
        """
        parts = path.split(".", 1)
        sub_container = parts[0]

        if sub_container not in self._sub_containers:
            raise PathResolutionError(
                path,
                f"Sub-container '{sub_container}' not found. Available: {self._sub_containers}",
            )

        if len(parts) == 1:
            # Return entire sub-container
            return self._data[sub_container]

        # Navigate nested path
        key_path = parts[1]
        try:
            return self._get_nested(self._data[sub_container], key_path)
        except KeyError as e:
            raise PathResolutionError(path, str(e)) from e
        except TypeError as e:
            raise PathResolutionError(path, str(e)) from e

    def set_item(
        self, path: str, value: Any, use_weakref: Optional[bool] = None
    ) -> None:
        """Set item using dot-notation path with automatic reference management.

        Args:
            path: Dot-notation path to set the item at
            value: The value to set
            use_weakref: If None, automatically determine; otherwise use specified value

        Raises:
            PathResolutionError: If the path cannot be resolved
        """
        parts = path.split(".", 1)
        sub_container = parts[0]

        if sub_container not in self._sub_containers:
            raise PathResolutionError(
                path,
                f"Sub-container '{sub_container}' not found. Available: {self._sub_containers}",
            )

        if len(parts) == 1:
            raise PathResolutionError(path, "Cannot replace entire sub-container")

        # Navigate and set nested path
        key_path = parts[1]
        try:
            self._set_nested(self._data[sub_container], key_path, value, use_weakref)
        except TypeError as e:
            raise PathResolutionError(path, str(e)) from e

    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot-notation path.

        Args:
            data: The dictionary to navigate
            path: Dot-notation path within the dictionary

        Returns:
            The value at the specified path

        Raises:
            KeyError: If a key in the path is not found
            TypeError: If trying to navigate through a non-dict
        """
        keys = path.split(".")
        current = data

        for i, key in enumerate(keys):
            if key not in current:
                raise KeyError(f"Key '{key}' not found in path '{path}'")

            if isinstance(current[key], tuple) and len(current[key]) == 2:
                ref, is_weakref = current[key]
                if is_weakref:
                    value = ReferenceManager.dereference(ref, is_weakref)
                    current = value
                else:
                    current = ref
            else:
                current = current[key]

            if i < len(keys) - 1 and not isinstance(current, dict):
                raise TypeError(
                    f"Cannot navigate through non-dict at '{'.'.join(keys[: i + 1])}' "
                    f"in path '{path}'"
                )

        return current

    def _set_nested(
        self, data: Dict, path: str, value: Any, use_weakref: Optional[bool]
    ) -> None:
        """Set value in nested dict using dot-notation path, creating intermediate dicts.

        Args:
            data: The dictionary to navigate
            path: Dot-notation path within the dictionary
            value: The value to set
            use_weakref: If None, automatically determine; otherwise use specified value

        Raises:
            TypeError: If trying to create nested path through non-dict
        """2
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise TypeError(
                    f"Cannot create nested path through non-dict at '{key}'"
                )
            current = current[key]

        final_key = keys[-1]
        ref, is_weakref = ReferenceManager.create_reference(value, use_weakref)
        current[final_key] = (ref, is_weakref) if is_weakref else value

    def clear(self) -> None:
        """Clear all data in this container."""
        for sub in self._sub_containers:
            self._data[sub] = {}

    def cleanup_dead_refs(self) -> None:
        """Remove dead weak references from all sub-containers."""
        for sub in self._sub_containers:
            self._cleanup_dict(self._data[sub])

    def _cleanup_dict(self, d: Dict) -> None:
        """Recursively cleanup dead refs in dict.

        Args:
            d: Dictionary to clean up
        """
        keys_to_remove = []
        for key, value in list(d.items()):
            if isinstance(value, tuple) and len(value) == 2:
                ref, is_weakref = value
                if is_weakref and ref() is None:
                    keys_to_remove.append(key)
            elif isinstance(value, dict):
                self._cleanup_dict(value)

        for key in keys_to_remove:
            del d[key]

class PerBatchContainer(BaseLifecycleContainer):
    """Container for single forward pass data."""

    def _initialize_sub_containers(self) -> None:
        self._sub_containers = ["input", "prediction", "target", "additional"]
        for sub in self._sub_containers:
            self._data[sub] = {}


class PerStepContainer(BaseLifecycleContainer):
    """Container for optimization step data."""

    def _initialize_sub_containers(self) -> None:
        self._sub_containers = [
            "gradients",
            "gradient_masks",
            "gradient_metadata",
            "loss",
            "additional",
        ]
        for sub in self._sub_containers:
            self._data[sub] = {}


class PerEpochContainer(BaseLifecycleContainer):
    """Container for epoch-level data."""

    def _initialize_sub_containers(self) -> None:
        self._sub_containers = [
            "metrics",
            "statistics",
            "validation",
            "checkpoints",
            "additional",
        ]
        for sub in self._sub_containers:
            self._data[sub] = {}


class PersistentContainer(BaseLifecycleContainer):
    """Container for training run data."""

    def _initialize_sub_containers(self) -> None:
        self._sub_containers = [
            "model",
            "optimizer",
            "scheduler",
            "config",
            "additional",
        ]
        for sub in self._sub_containers:
            self._data[sub] = {}
