"""
Module filtering utilities for flexible module selection in hooks system.
"""

import re
from typing import List, Tuple, Type, Callable, Union
import torch.nn as nn


class ModuleFilter:
    """
    Provides flexible module selection capabilities for hook registration.
    
    This class offers various filtering methods to select modules from a PyTorch model
    based on name patterns, module types, or custom filter functions.
    """
    
    @staticmethod
    def filter_by_name_pattern(model: nn.Module, pattern: str) -> List[Tuple[str, nn.Module]]:
        """
        Filter modules by name pattern using regex matching.
        
        Args:
            model: PyTorch model to filter modules from
            pattern: Regex pattern to match module names against
            
        Returns:
            List of tuples containing (module_name, module) for matching modules
            
        Raises:
            ValueError: If pattern is invalid regex
        """
        if not pattern:
            raise ValueError("Pattern cannot be empty")
            
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        matching_modules = []
        for name, module in model.named_modules():
            if compiled_pattern.search(name):
                matching_modules.append((name, module))
        
        return matching_modules
    
    @staticmethod
    def filter_by_type(model: nn.Module, module_types: Union[Type[nn.Module], List[Type[nn.Module]]]) -> List[Tuple[str, nn.Module]]:
        """
        Filter modules by their type.
        
        Args:
            model: PyTorch model to filter modules from
            module_types: Single module type or list of module types to match
            
        Returns:
            List of tuples containing (module_name, module) for matching modules
            
        Raises:
            ValueError: If module_types is empty or contains invalid types
        """
        if not module_types:
            raise ValueError("module_types cannot be empty")
        
        # Normalize to list
        if not isinstance(module_types, list):
            module_types = [module_types]
        
        # Validate all types are subclasses of nn.Module
        for module_type in module_types:
            if not (isinstance(module_type, type) and issubclass(module_type, nn.Module)):
                raise ValueError(f"Invalid module type: {module_type}. Must be a subclass of nn.Module")
        
        matching_modules = []
        for name, module in model.named_modules():
            if isinstance(module, tuple(module_types)):
                matching_modules.append((name, module))
        
        return matching_modules
    
    @staticmethod
    def filter_by_custom_function(model: nn.Module, filter_fn: Callable[[str, nn.Module], bool]) -> List[Tuple[str, nn.Module]]:
        """
        Filter modules using a custom filter function.
        
        Args:
            model: PyTorch model to filter modules from
            filter_fn: Function that takes (module_name, module) and returns True if module should be included
            
        Returns:
            List of tuples containing (module_name, module) for matching modules
            
        Raises:
            ValueError: If filter_fn is not callable
            RuntimeError: If filter_fn raises an exception during execution
        """
        if not callable(filter_fn):
            raise ValueError("filter_fn must be callable")
        
        matching_modules = []
        for name, module in model.named_modules():
            try:
                if filter_fn(name, module):
                    matching_modules.append((name, module))
            except Exception as e:
                raise RuntimeError(f"Filter function failed for module '{name}': {e}")
        
        return matching_modules
    
    @staticmethod
    def validate_filter_criteria(pattern: str = None, module_types: Union[Type[nn.Module], List[Type[nn.Module]]] = None, 
                                filter_fn: Callable[[str, nn.Module], bool] = None) -> None:
        """
        Validate filter criteria before applying filters.
        
        Args:
            pattern: Regex pattern for name-based filtering
            module_types: Module types for type-based filtering
            filter_fn: Custom filter function
            
        Raises:
            ValueError: If validation fails
        """
        criteria_count = sum(x is not None for x in [pattern, module_types, filter_fn])
        
        if criteria_count == 0:
            raise ValueError("At least one filter criterion must be provided")
        
        if criteria_count > 1:
            raise ValueError("Only one filter criterion can be used at a time")
        
        if pattern is not None:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("Pattern must be a non-empty string")
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        if module_types is not None:
            if isinstance(module_types, list):
                if not module_types:
                    raise ValueError("module_types list cannot be empty")
                for module_type in module_types:
                    if not (isinstance(module_type, type) and issubclass(module_type, nn.Module)):
                        raise ValueError(f"Invalid module type: {module_type}. Must be a subclass of nn.Module")
            else:
                if not (isinstance(module_types, type) and issubclass(module_types, nn.Module)):
                    raise ValueError(f"Invalid module type: {module_types}. Must be a subclass of nn.Module")
        
        if filter_fn is not None:
            if not callable(filter_fn):
                raise ValueError("filter_fn must be callable")