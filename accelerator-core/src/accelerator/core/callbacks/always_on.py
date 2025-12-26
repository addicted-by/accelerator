"""Factory helpers for callbacks that should always be active.

These constructors return instances of callbacks that are considered
core to the training loop and are therefore attached by default by the
:class:`ComponentManager`.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseCallback
from .progress import RichProgressBar, StepEpochTrackerCallback, TimeTrackingCallback, TqdmProgressBar


def create_step_epoch_tracker_callback() -> StepEpochTrackerCallback:
    """Return the callback keeping global step/epoch counters in sync."""
    return StepEpochTrackerCallback()


def create_time_tracking_callback() -> TimeTrackingCallback:
    """Return the callback responsible for tracking elapsed time."""
    return TimeTrackingCallback()


def create_progress_bar_callback(kind: str = "tqdm") -> BaseCallback:
    """Return a progress bar callback instance.

    Args:
        kind: Type of progress bar to create. Supported values are
            ``"tqdm"`` and ``"rich"``.

    Returns:
        Instantiated progress bar callback.

    Raises:
        ValueError: If an unsupported progress bar type is provided.

    """
    kind = kind.lower()
    if kind == "rich":
        return RichProgressBar()
    if kind == "tqdm":
        return TqdmProgressBar()
    raise ValueError(f"Unsupported progress bar type: {kind}")


def create_always_on_callbacks(progress_bar: Optional[str] = None) -> list[BaseCallback]:
    """Create the list of callbacks that are always enabled.

    Args:
        progress_bar: Which progress bar to use. Supported values are
            ``"tqdm"`` and ``"rich"``. Pass ``None`` to disable the progress
            bar entirely. Defaults to ``None``.

    Returns:
        List of instantiated callback objects.

    """
    callbacks = [
        create_step_epoch_tracker_callback(),
        create_time_tracking_callback(),
    ]
    if progress_bar:
        callbacks.append(create_progress_bar_callback(progress_bar))
    return callbacks
