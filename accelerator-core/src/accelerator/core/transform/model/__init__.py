"""Model transformation utilities for the accelerator framework.

This module provides built-in model transforms that can be applied before or after
acceleration operations like pruning. Transforms are callables that modify models
in-place.

Available transforms:
    - set_eval_mode: Set model to evaluation mode
    - set_train_mode: Set model to training mode
    - fuse_batch_norm: Fuse batch normalization layers into preceding conv/linear layers
    - unfreeze_parameters: Unfreeze all model parameters
"""

from .transforms import (
    fuse_batch_norm,
    set_eval_mode,
    set_train_mode,
    unfreeze_parameters,
)

__all__ = [
    "set_eval_mode",
    "set_train_mode",
    "fuse_batch_norm",
    "unfreeze_parameters",
]
