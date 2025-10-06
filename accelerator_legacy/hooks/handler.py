"""
Main HooksHandler class for unified hook management.

This module provides the primary interface for registering and managing
PyTorch module hooks across neural networks.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import torch

from .types import HookType, HooksConfig
from .registry import HookRegistry
from .data_collector import DataCollector
from .module_filter import ModuleFilter
from .default_hooks import (
    default_forward_hook,
    default_backward_hook,
    default_full_backward_hook
)
from .validation import validate_hook_signature, InvalidHookSignatureError


class ModuleNotFoundError(Exception):
    """Raised when specified modules don't exist in the model."""


class HooksHandler:
    """
    Main interface for managing PyTorch module hooks.
    
    The HooksHandler provides a unified interface for registering and managing
    different types of hooks on PyTorch neural network modules. It supports
    forward hooks, backward hooks, and full backward hooks, with flexible
    filtering and bulk operations for efficient instrumentation of large networks.
    """

    def __init__(self, model: torch.nn.Module, data_mode: str = "replace"):
        """
        Initialize the HooksHandler.

        Args:
            model: The PyTorch model to manage hooks for
            data_mode: Data collection mode ("replace", "accumulate", "list")

        Raises:
            ValueError: If model is None or data_mode is invalid
        """
        if model is None:
            raise ValueError("Model cannot be None")
        
        if data_mode not in ["replace", "accumulate", "list"]:
            raise ValueError(f"Invalid data_mode '{data_mode}'. Must be one of: replace, accumulate, list")

        self.model = model
        self.config = HooksConfig(data_mode=data_mode)
        self.registry = HookRegistry()
        self.data_collector = DataCollector(self.config)

    def register_forward_hook(
        self,
        modules: Union[str, List[str], Callable],
        hook_fn: Optional[Callable] = None
    ) -> Union[str, List[str]]:
        """
        Register forward hooks on specified modules.

        Args:
            modules: Module specification (name pattern, list of names, or filter function)
            hook_fn: Custom hook function. If None, uses default forward hook.

        Returns:
            Hook ID(s) for the registered hook(s)

        Raises:
            ModuleNotFoundError: If specified modules don't exist
            InvalidHookSignatureError: If custom hook function has incorrect signature
            HookRegistrationError: If PyTorch hook registration fails
        """
        return self._register_hooks(modules, HookType.FORWARD, hook_fn)

    def register_backward_hook(
        self,
        modules: Union[str, List[str], Callable],
        hook_fn: Optional[Callable] = None
    ) -> Union[str, List[str]]:
        """
        Register backward hooks on specified modules.

        Args:
            modules: Module specification (name pattern, list of names, or filter function)
            hook_fn: Custom hook function. If None, uses default backward hook.

        Returns:
            Hook ID(s) for the registered hook(s)

        Raises:
            ModuleNotFoundError: If specified modules don't exist
            InvalidHookSignatureError: If custom hook function has incorrect signature
            HookRegistrationError: If PyTorch hook registration fails
        """
        return self._register_hooks(modules, HookType.BACKWARD, hook_fn)

    def register_full_backward_hook(
        self,
        modules: Union[str, List[str], Callable],
        hook_fn: Optional[Callable] = None
    ) -> Union[str, List[str]]:
        """
        Register full backward hooks on specified modules.

        Args:
            modules: Module specification (name pattern, list of names, or filter function)
            hook_fn: Custom hook function. If None, uses default full backward hook.

        Returns:
            Hook ID(s) for the registered hook(s)

        Raises:
            ModuleNotFoundError: If specified modules don't exist
            InvalidHookSignatureError: If custom hook function has incorrect signature
            HookRegistrationError: If PyTorch hook registration fails
        """
        return self._register_hooks(modules, HookType.FULL_BACKWARD, hook_fn)

    def enable_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Enable hooks by their IDs.

        Args:
            hook_ids: List of hook IDs to enable. If None, enables all disabled hooks.

        Returns:
            Number of hooks that were enabled

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        return self.registry.enable_hooks(hook_ids)

    def disable_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Disable hooks by their IDs.

        Args:
            hook_ids: List of hook IDs to disable. If None, disables all enabled hooks.

        Returns:
            Number of hooks that were disabled

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        return self.registry.disable_hooks(hook_ids)

    def remove_hooks(self, hook_ids: Optional[List[str]] = None) -> int:
        """
        Remove hooks by their IDs.

        Args:
            hook_ids: List of hook IDs to remove. If None, removes all hooks.

        Returns:
            Number of hooks that were removed

        Raises:
            HookNotFoundError: If any hook ID doesn't exist
        """
        return self.registry.remove_hooks(hook_ids)

    def get_activations(self, module_name: str) -> torch.Tensor:
        """
        Get activation data for a module.

        Args:
            module_name: Name of the module to get activations for

        Returns:
            Activation tensor for the module

        Raises:
            DataNotAvailableError: If no activation data exists for the module
        """
        return self.data_collector.get_activation(module_name)

    def get_gradients(self, module_name: str) -> torch.Tensor:
        """
        Get gradient data for a module.

        Args:
            module_name: Name of the module to get gradients for

        Returns:
            Gradient tensor for the module

        Raises:
            DataNotAvailableError: If no gradient data exists for the module
        """
        return self.data_collector.get_gradient(module_name)

    def clear_data(self) -> None:
        """Clear all stored hook data."""
        self.data_collector.clear_all_data()

    def get_hook_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the hooks handler state.

        Returns:
            Dictionary containing summary information about hooks and data
        """
        registry_summary = self.registry.get_registry_summary()
        data_summary = self.data_collector.get_data_summary()
        
        return {
            "model_type": self.model.__class__.__name__,
            "config": {
                "data_mode": self.config.data_mode,
                "auto_clear": self.config.auto_clear,
                "max_memory_mb": self.config.max_memory_mb,
                "device_placement": self.config.device_placement
            },
            "registry": registry_summary,
            "data": data_summary
        }

    def _register_hooks(
        self,
        modules: Union[str, List[str], Callable],
        hook_type: HookType,
        hook_fn: Optional[Callable] = None
    ) -> Union[str, List[str]]:
        """
        Internal method to register hooks on modules.

        Args:
            modules: Module specification
            hook_type: Type of hook to register
            hook_fn: Custom hook function or None for default

        Returns:
            Hook ID(s) for registered hooks

        Raises:
            ModuleNotFoundError: If specified modules don't exist
            InvalidHookSignatureError: If custom hook function has incorrect signature
            HookRegistrationError: If PyTorch hook registration fails
        """
        # Get target modules based on specification
        target_modules = self._resolve_modules(modules)
        
        if not target_modules:
            raise ModuleNotFoundError("No modules found matching the specification")

        # Validate custom hook function if provided
        if hook_fn is not None:
            validate_hook_signature(hook_fn, hook_type)

        # Register hooks on all target modules
        hook_ids = []
        for module_name, module in target_modules:
            # Create hook function (default or custom)
            if hook_fn is None:
                actual_hook_fn = self._create_default_hook(hook_type, module_name)
            else:
                actual_hook_fn = hook_fn

            # Register the hook
            hook_id = self.registry.register_hook(
                module=module,
                hook_type=hook_type,
                hook_fn=actual_hook_fn,
                module_name=module_name
            )
            hook_ids.append(hook_id)

        # Return single ID or list based on input type
        if isinstance(modules, str) and len(hook_ids) == 1:
            return hook_ids[0]
        return hook_ids

    def register_hooks_by_name_pattern(
        self,
        pattern: str,
        hook_type: HookType,
        hook_fn: Optional[Callable] = None
    ) -> List[str]:
        """
        Register hooks on modules matching a name pattern.

        Args:
            pattern: Regex pattern to match module names
            hook_type: Type of hook to register
            hook_fn: Custom hook function. If None, uses default hook.

        Returns:
            List of hook IDs for the registered hooks

        Raises:
            ModuleNotFoundError: If no modules match the pattern
            InvalidHookSignatureError: If custom hook function has incorrect signature
        """
        target_modules = ModuleFilter.filter_by_name_pattern(self.model, pattern)
        
        if not target_modules:
            raise ModuleNotFoundError(f"No modules found matching pattern '{pattern}'")

        return self._register_hooks_on_modules(target_modules, hook_type, hook_fn)

    def register_hooks_by_type(
        self,
        module_types: Union[type, List[type]],
        hook_type: HookType,
        hook_fn: Optional[Callable] = None
    ) -> List[str]:
        """
        Register hooks on modules of specified types.

        Args:
            module_types: Module type or list of module types to match
            hook_type: Type of hook to register
            hook_fn: Custom hook function. If None, uses default hook.

        Returns:
            List of hook IDs for the registered hooks

        Raises:
            ModuleNotFoundError: If no modules match the types
            InvalidHookSignatureError: If custom hook function has incorrect signature
        """
        target_modules = ModuleFilter.filter_by_type(self.model, module_types)
        
        if not target_modules:
            type_names = [t.__name__ for t in (module_types if isinstance(module_types, list) else [module_types])]
            raise ModuleNotFoundError(f"No modules found of types: {type_names}")

        return self._register_hooks_on_modules(target_modules, hook_type, hook_fn)

    def register_hooks_by_filter(
        self,
        filter_fn: Callable[[str, torch.nn.Module], bool],
        hook_type: HookType,
        hook_fn: Optional[Callable] = None
    ) -> List[str]:
        """
        Register hooks on modules selected by a custom filter function.

        Args:
            filter_fn: Function that takes (module_name, module) and returns True for modules to hook
            hook_type: Type of hook to register
            hook_fn: Custom hook function. If None, uses default hook.

        Returns:
            List of hook IDs for the registered hooks

        Raises:
            ModuleNotFoundError: If no modules match the filter
            InvalidHookSignatureError: If custom hook function has incorrect signature
        """
        target_modules = ModuleFilter.filter_by_custom_function(self.model, filter_fn)
        
        if not target_modules:
            raise ModuleNotFoundError("No modules found matching the filter function")

        return self._register_hooks_on_modules(target_modules, hook_type, hook_fn)

    def _resolve_modules(
        self,
        modules: Union[str, List[str], Callable]
    ) -> List[tuple[str, torch.nn.Module]]:
        """
        Resolve module specification to actual modules.

        Args:
            modules: Module specification

        Returns:
            List of (module_name, module) tuples

        Raises:
            ModuleNotFoundError: If specified modules don't exist
            ValueError: If module specification is invalid
        """
        if isinstance(modules, str):
            # Single module name or pattern
            if self._is_pattern(modules):
                # Treat as regex pattern
                return ModuleFilter.filter_by_name_pattern(self.model, modules)
            # Treat as exact module name
            module = self._get_module_by_name(modules)
            if module is None:
                raise ModuleNotFoundError(f"Module '{modules}' not found in model")
            return [(modules, module)]
        
        if isinstance(modules, list):
            # List of module names
            resolved_modules = []
            for module_name in modules:
                if not isinstance(module_name, str) or not module_name.strip():
                    raise ValueError(f"Module name must be a non-empty string, got: {module_name}")
                module = self._get_module_by_name(module_name)
                if module is None:
                    raise ModuleNotFoundError(f"Module '{module_name}' not found in model")
                resolved_modules.append((module_name, module))
            return resolved_modules
        
        if callable(modules):
            # Custom filter function
            return ModuleFilter.filter_by_custom_function(self.model, modules)
        
        raise ValueError(f"Invalid modules specification type: {type(modules)}")

    def _register_hooks_on_modules(
        self,
        target_modules: List[tuple[str, torch.nn.Module]],
        hook_type: HookType,
        hook_fn: Optional[Callable] = None
    ) -> List[str]:
        """
        Register hooks on a list of target modules.

        Args:
            target_modules: List of (module_name, module) tuples
            hook_type: Type of hook to register
            hook_fn: Custom hook function or None for default

        Returns:
            List of hook IDs for registered hooks

        Raises:
            InvalidHookSignatureError: If custom hook function has incorrect signature
        """
        # Validate custom hook function if provided
        if hook_fn is not None:
            validate_hook_signature(hook_fn, hook_type)

        # Register hooks on all target modules
        hook_ids = []
        for module_name, module in target_modules:
            # Create hook function (default or custom)
            if hook_fn is None:
                actual_hook_fn = self._create_default_hook(hook_type, module_name)
            else:
                actual_hook_fn = hook_fn

            # Register the hook
            hook_id = self.registry.register_hook(
                module=module,
                hook_type=hook_type,
                hook_fn=actual_hook_fn,
                module_name=module_name
            )
            hook_ids.append(hook_id)

        return hook_ids

    def _get_module_by_name(self, module_name: str) -> Optional[torch.nn.Module]:
        """
        Get a module by its name from the model.

        Args:
            module_name: Name of the module to retrieve

        Returns:
            The module if found, None otherwise
        """
        for name, module in self.model.named_modules():
            if name == module_name:
                return module
        return None

    def _is_pattern(self, module_spec: str) -> bool:
        """
        Check if a module specification is a regex pattern.

        Args:
            module_spec: Module specification string

        Returns:
            True if the specification contains regex metacharacters
        """
        # Simple heuristic: if it contains regex metacharacters, treat as pattern
        regex_chars = {'*', '+', '?', '[', ']', '(', ')', '{', '}', '^', '$', '|', '\\', '.'}
        return any(char in module_spec for char in regex_chars)

    def _create_default_hook(self, hook_type: HookType, module_name: str) -> Callable:
        """
        Create a default hook function for the specified type.

        Args:
            hook_type: Type of hook to create
            module_name: Name of the module for data storage

        Returns:
            Default hook function

        Raises:
            ValueError: If hook type is not supported
        """
        if hook_type == HookType.FORWARD:
            return default_forward_hook(self.data_collector, module_name)
        if hook_type == HookType.BACKWARD:
            return default_backward_hook(self.data_collector, module_name)
        if hook_type == HookType.FULL_BACKWARD:
            return default_full_backward_hook(self.data_collector, module_name)
        raise ValueError(f"Unsupported hook type: {hook_type}")

    def __repr__(self) -> str:
        """String representation of the HooksHandler."""
        return (f"HooksHandler(model={self.model.__class__.__name__}, "
                f"hooks={len(self.registry)}, "
                f"data_modules={len(self.data_collector)})")