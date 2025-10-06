"""
HookRegistry class for managing hook lifecycle and state tracking.
"""

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import torch

from .types import HookInfo, HookType


class HookNotFoundError(Exception):
    """Raised when trying to operate on non-existent hooks."""


class HookStateError(Exception):
    """Raised when trying to perform invalid state transitions."""


class HookRegistrationError(Exception):
    """Raised when PyTorch hook registration fails."""


class HookRegistry:
    """
    Manages the lifecycle and state of registered hooks.
    
    The registry tracks all registered hooks, their state (enabled/disabled),
    and provides methods for hook lifecycle management including registration,
    enabling, disabling, and removal.
    """

    def __init__(self):
        """Initialize the HookRegistry."""
        self._hooks: Dict[str, HookInfo] = {}
        self._hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._enabled_hooks: Set[str] = set()
        self._disabled_hooks: Set[str] = set()

    def register_hook(
        self,
        module: torch.nn.Module,
        hook_type: HookType,
        hook_fn: Callable,
        module_name: Optional[str] = None
    ) -> str:
        """
        Register a hook on a module.

        Args:
            module: The PyTorch module to register the hook on
            hook_type: Type of hook to register (forward, backward, full_backward)
            hook_fn: The hook function to register
            module_name: Optional name for the module (auto-generated if None)

        Returns:
            Unique hook ID for the registered hook

        Raises:
            HookRegistrationError: If PyTorch hook registration fails
        """
        # Generate unique hook ID
        hook_id = str(uuid.uuid4())
        
        # Generate module name if not provided
        if module_name is None:
            module_name = f"{module.__class__.__name__}_{id(module)}"

        try:
            # Register the appropriate hook type
            if hook_type == HookType.FORWARD:
                handle = module.register_forward_hook(hook_fn)
            elif hook_type == HookType.BACKWARD:
                handle = module.register_backward_hook(hook_fn)
            elif hook_type == HookType.FULL_BACKWARD:
                handle = module.register_full_backward_hook(hook_fn)
            else:
                raise HookRegistrationError(f"Unsupported hook type: {hook_type}")

        except Exception as e:
            raise HookRegistrationError(
                f"Failed to register {hook_type.value} hook on module {module_name}: {e}"
            ) from e

        # Create hook info
        hook_info = HookInfo(
            hook_id=hook_id,
            module_name=module_name,
            module_type=module.__class__.__name__,
            hook_type=hook_type,
            is_enabled=True,
            registration_time=datetime.now(),
            execution_count=0
        )

        # Store hook information and handle
        self._hooks[hook_id] = hook_info
        self._hook_handles[hook_id] = handle
        self._enabled_hooks.add(hook_id)

        return hook_id

    def enable_hook(self, hook_id: str) -> None:
        """
        Enable a previously disabled hook.

        Args:
            hook_id: ID of the hook to enable

        Raises:
            HookNotFoundError: If hook ID doesn't exist
            HookStateError: If hook is already enabled
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        if hook_id in self._enabled_hooks:
            raise HookStateError(f"Hook '{hook_id}' is already enabled")

        # Move from disabled to enabled
        self._disabled_hooks.discard(hook_id)
        self._enabled_hooks.add(hook_id)
        self._hooks[hook_id].is_enabled = True

    def disable_hook(self, hook_id: str) -> None:
        """
        Disable a hook without removing it.

        Args:
            hook_id: ID of the hook to disable

        Raises:
            HookNotFoundError: If hook ID doesn't exist
            HookStateError: If hook is already disabled
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        if hook_id in self._disabled_hooks:
            raise HookStateError(f"Hook '{hook_id}' is already disabled")

        # Move from enabled to disabled
        self._enabled_hooks.discard(hook_id)
        self._disabled_hooks.add(hook_id)
        self._hooks[hook_id].is_enabled = False

    def remove_hook(self, hook_id: str) -> None:
        """
        Remove a hook completely and clean up resources.

        Args:
            hook_id: ID of the hook to remove

        Raises:
            HookNotFoundError: If hook ID doesn't exist
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        # Remove the PyTorch hook handle
        if hook_id in self._hook_handles:
            handle = self._hook_handles[hook_id]
            handle.remove()
            del self._hook_handles[hook_id]

        # Clean up tracking sets
        self._enabled_hooks.discard(hook_id)
        self._disabled_hooks.discard(hook_id)

        # Remove hook info
        del self._hooks[hook_id]

    def enable_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Enable multiple hooks at once.

        Args:
            hook_ids: List of hook IDs to enable. If None, enables all disabled hooks.

        Returns:
            Number of hooks that were enabled

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        if hook_ids is None:
            hook_ids = list(self._disabled_hooks)

        enabled_count = 0
        for hook_id in hook_ids:
            try:
                self.enable_hook(hook_id)
                enabled_count += 1
            except HookStateError:
                # Hook already enabled, skip
                continue

        return enabled_count

    def disable_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Disable multiple hooks at once.

        Args:
            hook_ids: List of hook IDs to disable. If None, disables all enabled hooks.

        Returns:
            Number of hooks that were disabled

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        if hook_ids is None:
            hook_ids = list(self._enabled_hooks)

        disabled_count = 0
        for hook_id in hook_ids:
            try:
                self.disable_hook(hook_id)
                disabled_count += 1
            except HookStateError:
                # Hook already disabled, skip
                continue

        return disabled_count

    def remove_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Remove multiple hooks at once.

        Args:
            hook_ids: List of hook IDs to remove. If None, removes all hooks.

        Returns:
            Number of hooks that were removed

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        if hook_ids is None:
            hook_ids = list(self._hooks.keys())

        removed_count = 0
        for hook_id in hook_ids:
            try:
                self.remove_hook(hook_id)
                removed_count += 1
            except HookNotFoundError:
                # Hook doesn't exist, skip
                continue

        return removed_count

    def get_active_hooks(self) -> List[str]:
        """
        Get list of all active (enabled) hook IDs.

        Returns:
            List of enabled hook IDs
        """
        return list(self._enabled_hooks)

    def get_inactive_hooks(self) -> List[str]:
        """
        Get list of all inactive (disabled) hook IDs.

        Returns:
            List of disabled hook IDs
        """
        return list(self._disabled_hooks)

    def get_all_hooks(self) -> List[str]:
        """
        Get list of all registered hook IDs.

        Returns:
            List of all hook IDs (enabled and disabled)
        """
        return list(self._hooks.keys())

    def get_hook_info(self, hook_id: str) -> HookInfo:
        """
        Get detailed information about a hook.

        Args:
            hook_id: ID of the hook to get information for

        Returns:
            HookInfo object containing hook details

        Raises:
            HookNotFoundError: If hook ID doesn't exist
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        return self._hooks[hook_id]

    def get_hooks_by_module(self, module_name: str) -> List[str]:
        """
        Get all hook IDs for a specific module.

        Args:
            module_name: Name of the module to get hooks for

        Returns:
            List of hook IDs registered on the specified module
        """
        return [
            hook_id for hook_id, hook_info in self._hooks.items()
            if hook_info.module_name == module_name
        ]

    def get_hooks_by_type(self, hook_type: HookType) -> List[str]:
        """
        Get all hook IDs of a specific type.

        Args:
            hook_type: Type of hooks to retrieve

        Returns:
            List of hook IDs of the specified type
        """
        return [
            hook_id for hook_id, hook_info in self._hooks.items()
            if hook_info.hook_type == hook_type
        ]

    def increment_execution_count(self, hook_id: str) -> None:
        """
        Increment the execution count for a hook.

        Args:
            hook_id: ID of the hook to increment count for

        Raises:
            HookNotFoundError: If hook ID doesn't exist
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        self._hooks[hook_id].execution_count += 1

    def is_hook_enabled(self, hook_id: str) -> bool:
        """
        Check if a hook is currently enabled.

        Args:
            hook_id: ID of the hook to check

        Returns:
            True if hook is enabled, False if disabled

        Raises:
            HookNotFoundError: If hook ID doesn't exist
        """
        if hook_id not in self._hooks:
            raise HookNotFoundError(f"Hook with ID '{hook_id}' not found")

        return hook_id in self._enabled_hooks

    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry state.

        Returns:
            Dictionary containing registry summary information
        """
        hook_types_count = {}
        for hook_info in self._hooks.values():
            hook_type = hook_info.hook_type.value
            hook_types_count[hook_type] = hook_types_count.get(hook_type, 0) + 1

        module_count = len(set(hook_info.module_name for hook_info in self._hooks.values()))

        return {
            "total_hooks": len(self._hooks),
            "enabled_hooks": len(self._enabled_hooks),
            "disabled_hooks": len(self._disabled_hooks),
            "unique_modules": module_count,
            "hook_types": hook_types_count,
            "total_executions": sum(hook_info.execution_count for hook_info in self._hooks.values())
        }

    def clear_all_hooks(self) -> int:
        """
        Remove all hooks and clean up all resources.

        Returns:
            Number of hooks that were removed
        """
        hook_count = len(self._hooks)
        
        # Remove all PyTorch hook handles
        for handle in self._hook_handles.values():
            handle.remove()

        # Clear all data structures
        self._hooks.clear()
        self._hook_handles.clear()
        self._enabled_hooks.clear()
        self._disabled_hooks.clear()

        return hook_count

    def validate_hook_state(self) -> bool:
        """
        Validate the internal consistency of the registry state.

        Returns:
            True if state is consistent, False otherwise
        """
        # Check that all hook IDs are consistent across data structures
        all_hook_ids = set(self._hooks.keys())
        handle_ids = set(self._hook_handles.keys())
        state_ids = self._enabled_hooks | self._disabled_hooks

        # All hooks should have handles
        if all_hook_ids != handle_ids:
            return False

        # All hooks should be in either enabled or disabled state
        if all_hook_ids != state_ids:
            return False

        # No hook should be both enabled and disabled
        if self._enabled_hooks & self._disabled_hooks:
            return False

        # Hook info enabled state should match tracking sets
        for hook_id, hook_info in self._hooks.items():
            expected_enabled = hook_id in self._enabled_hooks
            if hook_info.is_enabled != expected_enabled:
                return False

        return True

    def __len__(self) -> int:
        """
        Get total number of registered hooks.

        Returns:
            Number of registered hooks
        """
        return len(self._hooks)

    def __contains__(self, hook_id: str) -> bool:
        """
        Check if a hook ID is registered.

        Args:
            hook_id: Hook ID to check

        Returns:
            True if hook is registered
        """
        return hook_id in self._hooks

    def __repr__(self) -> str:
        """String representation of the HookRegistry."""
        return (f"HookRegistry(total={len(self._hooks)}, "
                f"enabled={len(self._enabled_hooks)}, "
                f"disabled={len(self._disabled_hooks)})")