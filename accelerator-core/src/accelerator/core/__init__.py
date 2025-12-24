import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_lazy_map = {
    # acceleration
    "AccelerationOperationBase": ("accelerator.core.acceleration", "AccelerationOperationBase"),
    "acceleration_registry": ("accelerator.core.acceleration", "acceleration_registry"),
    "AccelerationType": ("accelerator.core.acceleration", "AccelerationType"),
    "AccelerationRegistry": ("accelerator.core.acceleration", "AccelerationRegistry"),
    # callbacks
    "create_always_on_callbacks": ("accelerator.core.callbacks", "create_always_on_callbacks"),
    "BaseCallback": ("accelerator.core.callbacks", "BaseCallback"),
    "BaseLoggerCallback": ("accelerator.core.callbacks", "BaseLoggerCallback"),
    "TensorBoardLogger": ("accelerator.core.callbacks", "TensorBoardLogger"),
    "MLflowLogger": ("accelerator.core.callbacks", "MLflowLogger"),
    "CSVLogger": ("accelerator.core.callbacks", "CSVLogger"),
    "CallbackManager": ("accelerator.core.callbacks", "CallbackManager"),
    "TimeTrackingCallback": ("accelerator.core.callbacks", "TimeTrackingCallback"),
    "StepEpochTrackerCallback": ("accelerator.core.callbacks", "StepEpochTrackerCallback"),
    "TqdmProgressBar": ("accelerator.core.callbacks", "TqdmProgressBar"),
    "RichProgressBar": ("accelerator.core.callbacks", "RichProgressBar"),
    # checkpoint
    "operation": ("accelerator.core.checkpoint", "operation"),
    "CheckpointManager": ("accelerator.core.checkpoint", "CheckpointManager"),
    # context
    "Context": ("accelerator.core.context", "Context"),
    # datamodule
    "DataModule": ("accelerator.core.datamodule", "DataModule"),
    # distillation
    "DistillationManager": ("accelerator.core.distillation", "DistillationManager"),
    # engine
    "DistributedBackend": ("accelerator.core.engine", "DistributedBackend"),
    "AccelerateEngine": ("accelerator.core.engine", "AccelerateEngine"),
    "AccelerateEngineDefaults": ("accelerator.core.engine", "AccelerateEngineDefaults"),
    "DDPEngine": ("accelerator.core.engine", "DDPEngine"),
    "DDPEngineDefaults": ("accelerator.core.engine", "DDPEngineDefaults"),
    # loop
    "LoopBase": ("accelerator.core.loop", "LoopBase"),
    "TrainLoop": ("accelerator.core.loop", "TrainLoop"),
    # loss
    "LossWrapper": ("accelerator.core.loss", "LossWrapper"),
    "LossCombiner": ("accelerator.core.loss", "LossCombiner"),
    "LossType": ("accelerator.core.loss", "LossType"),
    "registry": ("accelerator.core.loss", "registry"),
    "InputValidator": ("accelerator.core.loss", "InputValidator"),
    "ValidationConfig": ("accelerator.core.loss", "ValidationConfig"),
    "ValidationError": ("accelerator.core.loss", "ValidationError"),
    "LossAPIException": ("accelerator.core.loss", "LossAPIException"),
    "LossCalculationError": ("accelerator.core.loss", "LossCalculationError"),
    "LossConfigurationError": ("accelerator.core.loss", "LossConfigurationError"),
    "LossStatistics": ("accelerator.core.loss", "LossStatistics"),
    "GradientLogger": ("accelerator.core.loss", "GradientLogger"),
    # model
    "AcceleratedModel": ("accelerator.core.model", "AcceleratedModel"),
    # pipeline
    "resolve_checkpoint_path": ("accelerator.core.pipeline", "resolve_checkpoint_path"),
    "make_step_cfg": ("accelerator.core.pipeline", "make_step_cfg"),
    "StepConfigManager": ("accelerator.core.pipeline", "StepConfigManager"),
    # scheduler
    "BaseSchedulerCallback": ("accelerator.core.scheduler", "BaseSchedulerCallback"),
    "PyTorchSchedulerCallback": ("accelerator.core.scheduler", "PyTorchSchedulerCallback"),
    # transform
    "BaseLossTransform": ("accelerator.core.transform", "BaseLossTransform"),
    "BaseTransform": ("accelerator.core.transform", "BaseTransform"),
    "TensorTransformType": ("accelerator.core.transform", "TensorTransformType"),
    "transforms_registry": ("accelerator.core.transform", "transforms_registry"),
    "LossTransformManager": ("accelerator.core.transform", "LossTransformManager"),
}


def __getattr__(name):
    """Lazy load attributes on first access."""
    try:
        mod_path, attr = _lazy_map[name]
    except KeyError:
        raise AttributeError(f"module 'accelerator.core' has no attribute '{name}'") from None
    module = importlib.import_module(mod_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = list(_lazy_map.keys())


def __dir__():
    """Keep tab-completion & help() honest without importing heavy stuff."""
    return sorted(set(globals()) | _lazy_map.keys())
