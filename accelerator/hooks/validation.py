"""
Validation functions for hook signatures and configurations.
"""

import inspect
from typing import Callable, TYPE_CHECKING
from torch import nn

from .types import HookType

if TYPE_CHECKING:
    from .types import HooksConfig


class InvalidHookSignatureError(Exception):
    """Raised when a hook function has an invalid signature."""
    pass


def validate_hook_signature(hook_fn: Callable, hook_type: HookType) -> bool:
    """
    Validate that a hook function has the correct signature for its type.
    
    Args:
        hook_fn: The hook function to validate
        hook_type: The type of hook (forward, backward, or full_backward)
        
    Returns:
        True if the signature is valid
        
    Raises:
        InvalidHookSignatureError: If the signature is invalid
    """
    sig = inspect.signature(hook_fn)
    params = list(sig.parameters.keys())
    
    if hook_type == HookType.FORWARD:
        expected_params = ["module", "input", "output"]
        if len(params) != 3:
            raise InvalidHookSignatureError(
                f"Forward hook must have exactly 3 parameters (module, input, output), "
                f"got {len(params)}: {params}"
            )
        if params != expected_params:
            raise InvalidHookSignatureError(
                f"Forward hook parameters must be {expected_params}, got {params}"
            )
            
    elif hook_type == HookType.BACKWARD:
        expected_params = ["module", "grad_input", "grad_output"]
        if len(params) != 3:
            raise InvalidHookSignatureError(
                f"Backward hook must have exactly 3 parameters (module, grad_input, grad_output), "
                f"got {len(params)}: {params}"
            )
        if params != expected_params:
            raise InvalidHookSignatureError(
                f"Backward hook parameters must be {expected_params}, got {params}"
            )
            
    elif hook_type == HookType.FULL_BACKWARD:
        expected_params = ["module", "grad_input", "grad_output"]
        if len(params) != 3:
            raise InvalidHookSignatureError(
                f"Full backward hook must have exactly 3 parameters (module, grad_input, grad_output), "
                f"got {len(params)}: {params}"
            )
        if params != expected_params:
            raise InvalidHookSignatureError(
                f"Full backward hook parameters must be {expected_params}, got {params}"
            )
    else:
        raise InvalidHookSignatureError(f"Unknown hook type: {hook_type}")
        
    return True


def validate_module_name(module_name: str) -> bool:
    """
    Validate that a module name exists in the given model.
    
    Args:
        model: The PyTorch model to check
        module_name: The name of the module to validate
        
    Returns:
        True if the module exists
        
    Raises:
        ValueError: If the module name doesn't exist
    """
    module_names = [name for name, _ in model.named_modules()]
    if module_name not in module_names:
        raise ValueError(
            f"Module '{module_name}' not found in model. "
            f"Available modules: {module_names[:10]}{'...' if len(module_names) > 10 else ''}"
        )
    return True


def validate_hooks_config(config: 'HooksConfig') -> bool:
    """
    Validate hooks configuration parameters.
    
    Args:
        config: The hooks configuration to validate
        
    Returns:
        True if the configuration is valid
        
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    valid_data_modes = ["replace", "accumulate", "list"]
    if config.data_mode not in valid_data_modes:
        raise ValueError(
            f"Invalid data_mode '{config.data_mode}'. "
            f"Must be one of: {valid_data_modes}"
        )
        
    valid_device_placements = ["same", "cpu", "cuda"]
    if config.device_placement not in valid_device_placements:
        raise ValueError(
            f"Invalid device_placement '{config.device_placement}'. "
            f"Must be one of: {valid_device_placements}"
        )
        
    if config.max_memory_mb is not None and config.max_memory_mb <= 0:
        raise ValueError(
            f"max_memory_mb must be positive, got {config.max_memory_mb}"
        )

    return True


def validate_default_hook_compatibility(hook_fn: Callable, hook_type: HookType) -> bool:
    """
    Validate that a hook function is compatible with the default hook patterns.
    
    This function checks if a custom hook function follows the same signature
    patterns as the default hooks, making it suitable for use with the hooks handler.
    
    Args:
        hook_fn: The hook function to validate
        hook_type: The type of hook (forward, backward, or full_backward)
        
    Returns:
        True if the hook is compatible with default patterns
        
    Raises:
        InvalidHookSignatureError: If the hook is not compatible
    """
    # First validate basic signature
    validate_hook_signature(hook_fn, hook_type)
    
    # Additional checks for default hook compatibility
    sig = inspect.signature(hook_fn)
    
    # Check that the function doesn't have unexpected return annotations
    if sig.return_annotation != inspect.Signature.empty and sig.return_annotation is not None:
        raise InvalidHookSignatureError(
            f"Hook functions should not return values (found return annotation: {sig.return_annotation})"
        )
    
    return True


def get_hook_signature_info(hook_type: HookType) -> dict:
    """
    Get information about the expected signature for a hook type.
    
    Args:
        hook_type: The type of hook to get signature info for
        
    Returns:
        Dictionary containing signature information
    """
    if hook_type == HookType.FORWARD:
        return {
            "parameters": ["module", "input", "output"],
            "description": "Forward hook receives the module, input tuple, and output tensor",
            "example": "def forward_hook(module, input, output): pass"
        }
    elif hook_type == HookType.BACKWARD:
        return {
            "parameters": ["module", "grad_input", "grad_output"],
            "description": "Backward hook receives the module, input gradients, and output gradients",
            "example": "def backward_hook(module, grad_input, grad_output): pass"
        }
    elif hook_type == HookType.FULL_BACKWARD:
        return {
            "parameters": ["module", "grad_input", "grad_output"],
            "description": "Full backward hook receives the module, input gradients, and output gradients",
            "example": "def full_backward_hook(module, grad_input, grad_output): pass"
        }
    else:
        raise ValueError(f"Unknown hook type: {hook_type}")


def is_valid_hook_function(hook_fn: Callable) -> tuple[bool, str]:
    """
    Check if a function can be used as any type of hook.
    
    Args:
        hook_fn: The function to check
        
    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    if not callable(hook_fn):
        return False, "Hook must be callable"
    
    sig = inspect.signature(hook_fn)
    params = list(sig.parameters.keys())
    
    if len(params) != 3:
        return False, f"Hook must have exactly 3 parameters, got {len(params)}"
    
    # Check if it matches any valid hook pattern
    valid_patterns = [
        ["module", "input", "output"],  # forward
        ["module", "grad_input", "grad_output"],  # backward/full_backward
    ]
    
    if params not in valid_patterns:
        return False, f"Invalid parameter names: {params}. Must match one of: {valid_patterns}"
    
    return True, ""