"""Smart reference management for context containers.

This module provides utilities for managing weak and strong references
to objects stored in context containers, optimizing memory usage while
maintaining data integrity.
"""

import weakref
from typing import Any, Tuple, Optional


class DeadReferenceError(Exception):
    """Raised when accessing a dead weak reference."""

    def __init__(
        self, message: str = "Weak reference target has been garbage collected"
    ):
        super().__init__(message)
        self.message = message


class ReferenceManager:
    """Manages weak/strong references based on object type.

    This class provides static methods to determine whether objects should
    use weak references and to create/dereference them appropriately.
    """

    @staticmethod
    def should_use_weakref(value: Any) -> bool:
        """Determine if value should use weak reference.

        Uses type-based heuristics to decide whether an object should be
        stored as a weak reference to optimize memory usage.

        Args:
            value: The object to evaluate

        Returns:
            True if the object should use a weak reference, False otherwise

        Rules:
            - PyTorch tensors and modules: Use weak refs (large memory footprint)
            - Primitives (int, float, str, bool, None): Never use weak refs
            - Built-in collections (list, dict, tuple, set): Never use weak refs
            - Other objects: Use weak refs by default
        """
        # Import torch only when needed to avoid hard dependency
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return True
            if isinstance(value, torch.nn.Module):
                return True
        except ImportError:
            pass

        # Primitives should never use weak refs
        if isinstance(value, (int, float, str, bool, type(None))):
            return False

        # Built-in collections don't support weak refs
        if isinstance(value, (dict, list, tuple, set, frozenset)):
            return False

        # Default to weak refs for other objects
        return True

    @staticmethod
    def create_reference(
        value: Any, use_weakref: Optional[bool] = None
    ) -> Tuple[Any, bool]:
        """Create appropriate reference for value.

        Creates either a weak reference or stores the value directly based
        on the use_weakref parameter or automatic type detection.

        Args:
            value: The object to create a reference for
            use_weakref: If True, force weak ref; if False, force strong ref;
                        if None, use automatic detection

        Returns:
            Tuple of (reference, is_weakref) where:
                - reference is either a weakref.ref or the value itself
                - is_weakref is True if a weak reference was created

        Note:
            If weak reference creation fails (e.g., for objects that don't
            support weak refs), falls back to strong reference automatically.
        """
        if use_weakref is None:
            use_weakref = ReferenceManager.should_use_weakref(value)

        if use_weakref:
            try:
                return weakref.ref(value), True
            except TypeError:
                # Object doesn't support weak references, use strong ref
                return value, False
        else:
            return value, False

    @staticmethod
    def dereference(ref: Any, is_weakref: bool) -> Any:
        """Dereference a reference.

        Extracts the actual value from a reference, handling both weak
        and strong references appropriately.

        Args:
            ref: The reference (either weakref.ref or the value itself)
            is_weakref: True if ref is a weak reference

        Returns:
            The dereferenced value

        Raises:
            DeadReferenceError: If the weak reference target has been
                              garbage collected
        """
        if is_weakref:
            value = ref()
            if value is None:
                raise DeadReferenceError(
                    "Weak reference target has been garbage collected"
                )
            return value
        return ref


def dereference(ref: Any, is_weakref: bool) -> Any:
    return ReferenceManager.dereference(ref, is_weakref)