"""
Type definitions and data structures for the hooks system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List
import torch


class HookType(Enum):
    """Types of hooks supported by the system."""
    FORWARD = "forward"
    BACKWARD = "backward"
    FULL_BACKWARD = "full_backward"


@dataclass
class HookData:
    """Data structure for storing hook information."""
    module_name: str
    data_type: str  # "activation" or "gradient"
    tensor_data: torch.Tensor
    shape: Tuple[int, ...]
    device: str
    dtype: torch.dtype
    timestamp: datetime


@dataclass
class HookInfo:
    """Information about a registered hook."""
    hook_id: str
    module_name: str
    module_type: str
    hook_type: HookType
    is_enabled: bool
    registration_time: datetime
    execution_count: int = 0


@dataclass
class HooksConfig:
    """Configuration for hooks behavior."""
    data_mode: str = "replace"  # "replace", "accumulate", "list"
    auto_clear: bool = False
    max_memory_mb: Optional[int] = None
    device_placement: str = "same"  # "same", "cpu", "cuda"