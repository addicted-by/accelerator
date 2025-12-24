"""Re-export acceleration registry components from accelerator.core.

This module provides convenient import paths for acceleration registry components
within the accelerator-algorithm package.
"""

from accelerator.core.acceleration.registry import (
    AccelerationRegistry,
    AccelerationType,
    acceleration_registry,
)

__all__ = [
    "AccelerationRegistry",
    "AccelerationType",
    "acceleration_registry",
]
