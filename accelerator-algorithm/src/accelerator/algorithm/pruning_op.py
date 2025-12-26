import accelerator
from accelerator.algorithm.pruning import (
    get_pruned_dict,
    load_pruned,
    pruner as pruners,
)
from accelerator.algorithm.registry import AccelerationType, registry
from accelerator.utilities.logging import get_logger

from .base import AccelerationOperationBase

log = get_logger(__name__)


@registry.register_acceleration(AccelerationType.PRUNING)
class QaPUTPruning(AccelerationOperationBase):
    """PLACEHOLDER."""

    def apply(self, context: "accelerator.Context"):
        model = context.model
        dataloaders = context.data.dataloaders

        pruner_name = self.config.get("_target_", None)
        if pruner_name is None:
            raise ValueError("Pruning target not specified")

        pruner = getattr(pruners, pruner_name, None)
        if pruner is None:
            raise ValueError(f"Pruner {pruner_name} not found")

        pruner = pruner(**self.config)
        log.info(f"Applying pruning with {pruner.__class__.__name__}...")

        pruner(model.model_core, dataloaders[self.config.dataloader_name])
        self._meta_data = {"pruned_dict": get_pruned_dict(model.model_core)}

    def reapply(self, model):
        log.info("Reapplying the pruning...")
        if self._meta_data is not None and "pruned_dict" in self._meta_data:
            pruned_dict = self._meta_data["pruned_dict"]
            load_pruned(model.model_core, pruned_dict)
        else:
            raise ValueError(self._not_loaded_error_msg)

    def calibrate(self, context):
        return
