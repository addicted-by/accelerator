"""Factory helpers for callbacks that should always be active.

These constructors return instances of callbacks that are considered
core to the training loop and are therefore attached by default by the
:class:`ComponentManager`.
"""
from __future__ import annotations

from typing import List

from .base import BaseCallback
from .progress import (
    StepEpochTrackerCallback,
    TimeTrackingCallback,
    TqdmProgressBar,
)


def create_step_epoch_tracker_callback() -> StepEpochTrackerCallback:
    """Return the callback keeping global step/epoch counters in sync."""
    return StepEpochTrackerCallback()


def create_time_tracking_callback() -> TimeTrackingCallback:
    """Return the callback responsible for tracking elapsed time."""
    return TimeTrackingCallback()


def create_progress_bar_callback() -> TqdmProgressBar:
    """Return a default progress bar callback.

    Currently this uses :class:`~accelerator.runtime.callbacks.progress.TqdmProgressBar`.
    """
    return TqdmProgressBar()


def create_always_on_callbacks(include_progress: bool = False) -> List[BaseCallback]:
    """Create the list of callbacks that are always enabled.

    Args:
        include_progress: Whether to include a default progress bar. Defaults to False.

    Returns:
        List of instantiated callback objects.
    """
    callbacks = [
        create_step_epoch_tracker_callback(),
        create_time_tracking_callback(),
    ]
    if include_progress:
        callbacks.append(create_progress_bar_callback())
    return callbacks
