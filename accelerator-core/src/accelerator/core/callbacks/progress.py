import time
from datetime import timedelta
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, TextColumn
from tqdm.auto import tqdm

from accelerator.utilities.distributed_state.state import distributed_state
from accelerator.utilities.logging import get_logger

from .base import BaseCallback

if TYPE_CHECKING:
    from accelerator.core.context import Context

logger = get_logger(__name__)


class StepEpochTrackerCallback(BaseCallback):
    """Always‑on callback that keeps the global *step* and *epoch* counters in sync with
    the :class:`~accelerator.runtime.context.context.Context`.
    """

    priority: int = 0
    critical: bool = True

    def on_train_batch_end(self, context: "Context"):
        context.update_step()

    def on_train_epoch_end(self, context: "Context"):
        context.update_epoch()


class TqdmProgressBar(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = None

    @distributed_state.on_main_process
    def on_train_epoch_begin(self, context):
        epoch = context.training_manager.state.current_epoch
        total_batches = getattr(context.data, "total_batches", 100)

        self.progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch}", dynamic_ncols=True)

    @distributed_state.on_main_process
    def on_train_batch_end(self, context):
        metrics = context.training_manager.state.training_metrics
        if self.progress_bar:
            self.progress_bar.update(1)
            if metrics:
                display_metrics = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
                self.progress_bar.set_postfix(display_metrics)

    @distributed_state.on_main_process
    def on_train_epoch_end(self, context):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None


class RichProgressBar(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = None
        self.epoch_task = None

    @distributed_state.on_main_process
    def on_train_begin(self, context):
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            auto_refresh=False,
        )
        self.progress.start()

    @distributed_state.on_main_process
    def on_train_epoch_begin(self, context):
        epoch = context.training_manager.state.current_epoch
        total_batches = getattr(context.data, "total_batches", 100)

        self.epoch_task = self.progress.add_task(f"Epoch {epoch}", total=total_batches)

    @distributed_state.on_main_process
    def on_train_batch_end(self, context):
        epoch = context.training_manager.state.current_epoch
        metrics = context.training_manager.state.training_metrics

        if self.epoch_task is not None:
            metrics_str = ""
            if metrics:
                metrics_str = " | " + " ".join(
                    [f"{k}:{v:.3f}" if isinstance(v, float) else f"{k}:{v}" for k, v in metrics.items()]
                )

            self.progress.update(self.epoch_task, advance=1, description=f"Epoch {epoch}{metrics_str}")
            self.progress.refresh()

    @distributed_state.on_main_process
    def on_train_epoch_end(self, context):
        if self.epoch_task is not None:
            self.progress.remove_task(self.epoch_task)
            self.epoch_task = None

    @distributed_state.on_main_process
    def on_train_end(self, context):
        if self.progress:
            self.progress.stop()
            self.progress = None


class TimeTrackingCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.training_start_time = None
        self.epoch_start_time = None
        self.epoch_times: list[float] = []
        self.current_epoch_time = 0.0

    @property
    def priority(self) -> int:
        return 10

    @distributed_state.on_main_process
    def on_train_begin(self, context):
        self.training_start_time = time.time()
        self.epoch_times = []
        self.total_epochs = context.training_manager.state.total_epochs

    @distributed_state.on_main_process
    def on_train_epoch_begin(self, context):
        self.epoch_start_time = time.time()

    @distributed_state.on_main_process
    def on_train_epoch_end(self, context):
        if self.epoch_start_time is None:
            return

        epoch_elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_elapsed)
        self.current_epoch_time = epoch_elapsed

        epoch = context.training_manager.state.current_epoch

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        total_elapsed = time.time() - self.training_start_time
        total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))

        if self.total_epochs and len(self.epoch_times) > 0:
            remaining_epochs = self.total_epochs - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_epoch_time
            estimated_remaining_str = str(timedelta(seconds=int(estimated_remaining)))
        else:
            estimated_remaining_str = "Unknown"

        context.training_manager.update_metrics(
            training_metrics={
                "eta": estimated_remaining_str,
                "elapsed": total_elapsed_str,
            }
        )

    @distributed_state.on_main_process
    def on_train_end(self, context):
        if self.training_start_time is None:
            return

        total_training_time = time.time() - self.training_start_time
        total_str = str(timedelta(seconds=int(total_training_time)))

        logger.info(f"Total training time: {total_str}")
        logger.info(f"Total epochs completed: {len(self.epoch_times)}")

        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            fastest_epoch = min(self.epoch_times)
            slowest_epoch = max(self.epoch_times)

            logger.info(f"Average epoch time: {str(timedelta(seconds=int(avg_epoch_time)))}")
            logger.info(f"Fastest epoch: {str(timedelta(seconds=int(fastest_epoch)))}")
            logger.info(f"Slowest epoch: {str(timedelta(seconds=int(slowest_epoch)))}")

    def get_current_stats(self) -> dict:
        """Get current timing statistics"""
        if not self.epoch_times:
            return {}

        return {
            "current_epoch_time": self.current_epoch_time,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times),
            "total_elapsed": time.time() - self.training_start_time if self.training_start_time else 0,
            "epochs_completed": len(self.epoch_times),
            "fastest_epoch": min(self.epoch_times),
            "slowest_epoch": max(self.epoch_times),
        }
