"""
PyTorch hooks handler module for neural network instrumentation and analysis.

This module provides a unified interface for managing PyTorch module hooks,
including forward hooks, backward hooks, and full backward hooks.
"""

# Core types first
from .types import HookType, HookInfo, HookData, HooksConfig

# Data collector
from .data_collector import DataCollector, DataNotAvailableError, MemoryLimitExceededError

# Validation
from .validation import (
    validate_hook_signature,
    validate_module_name,
    validate_hooks_config,
    validate_default_hook_compatibility,
    get_hook_signature_info,
    is_valid_hook_function,
    InvalidHookSignatureError
)

# Registry
from .registry import (
    HookRegistry,
    HookNotFoundError,
    HookStateError,
    HookRegistrationError
)

# Statistics
from .statistics import (
    ActivationStatisticsHook,
    StatisticsConfig,
    StatisticsCollector,
    ModuleStatistics,
    TensorStatistics,
    OnlineStatistics
)

# Module filter
from .module_filter import ModuleFilter

# Default hooks
from .default_hooks import (
    default_forward_hook,
    default_backward_hook,
    default_full_backward_hook,
    create_activation_capture_hook,
    create_gradient_capture_hook
)

# Main handler
from .handler import HooksHandler, ModuleNotFoundError

__all__ = [
    # Core types
    "HookType",
    "HookInfo",
    "HookData",
    "HooksConfig",
    # Data collector
    "DataCollector",
    "DataNotAvailableError",
    "MemoryLimitExceededError",
    # Validation
    "validate_hook_signature",
    "validate_module_name",
    "validate_hooks_config",
    "validate_default_hook_compatibility",
    "get_hook_signature_info",
    "is_valid_hook_function",
    "InvalidHookSignatureError",
    # Registry
    "HookRegistry",
    "HookNotFoundError",
    "HookStateError",
    "HookRegistrationError",
    # Statistics
    "ActivationStatisticsHook",
    "StatisticsConfig",
    "StatisticsCollector",
    "ModuleStatistics",
    "TensorStatistics",
    "OnlineStatistics",
    # Module filter
    "ModuleFilter",
    # Default hooks
    "default_forward_hook",
    "default_backward_hook",
    "default_full_backward_hook",
    "create_activation_capture_hook",
    "create_gradient_capture_hook",
    # Main handler
    "HooksHandler",
    "ModuleNotFoundError",
]