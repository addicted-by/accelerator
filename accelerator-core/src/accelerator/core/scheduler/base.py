import abc
from typing import TYPE_CHECKING, Any, Optional

from accelerator.core.callbacks import BaseCallback
from accelerator.utilities.logging import get_logger
from accelerator.utilities.typings import ConfigType

if TYPE_CHECKING:
    from accelerator.core.context import Context


logger = get_logger(__name__)


class BaseSchedulerCallback(BaseCallback, abc.ABC):
    def __init__(self, config: Optional[ConfigType], *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mode = config.get("mode", "epoch")  # 'step', 'epoch', 'metric'
        self.frequency = config.get("frequency", 1)
        self._step_count = 0
        self._epoch_count = 0

    @property
    def priority(self) -> int:
        return 90

    def get_optimizer(self, context):
        return context.optimizer

    def get_scheduler(self, context):
        return context.scheduler

    @abc.abstractmethod
    def step_scheduler(self, context: "Context"):
        pass

    def on_optimizer_step_end(self, context):
        if self.mode == "step":
            self._step_count += 1
            if self._step_count % self.frequency == 0:
                self.step_scheduler(context)

    def on_train_epoch_end(self, context):
        if self.mode == "epoch":
            self._epoch_count += 1
            if self._epoch_count % self.frequency == 0:
                self.step_scheduler(context)

    def on_val_epoch_end(self, context):
        if self.mode == "metric":
            self.step_scheduler(context)


class PyTorchSchedulerCallback(BaseSchedulerCallback):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.scheduler_class = config["scheduler_class"]
        self.scheduler_params = config.get("scheduler_params", {})
        self._scheduler = None

    def on_train_begin(self, context):
        optimizer = self.get_optimizer(context)
        if optimizer and not self._scheduler:
            self._scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            context.set_scheduler(self._scheduler)
            logger.info(f"Initialized {self.scheduler_class.__name__}")

    def step_scheduler(self, context):
        if self._scheduler:
            if self.mode == "metric":
                metric_value = self.get_metric_value(context)
                if metric_value is not None:
                    self._scheduler.step(metric_value)
            else:
                self._scheduler.step()
