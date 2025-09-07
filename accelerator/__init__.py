from typing import TYPE_CHECKING
from accelerator.utilities import (
    resolvers,  # noqa: F401 - side-effect import
    logging,
    patches,  # noqa: F401 - side-effect import
)
try:
    from accelerator import domain  # noqa: F401 - optional heavy import
except Exception:  # pragma: no cover - allow partial installs
    domain = None
import importlib

log = logging.get_logger()

_custom_patches = ["resolvers", "patches"]

if TYPE_CHECKING:
    from accelerator.runtime.context import Context  # noqa: F401 - used in lazy loading
    from accelerator.runtime.pipeline.manager import (  # noqa: F401 - used in lazy loading
        PipelineManager,
        PipelineStep,
    )
    from accelerator.runtime.checkpoint_manager.manager import CheckpointManager  # noqa: F401 - used in lazy loading
    from accelerator.runtime.operations.operation_registry import (
        registry as operation_registry, # noqa: F401 - used in lazy loading
    )
    from accelerator.runtime.loss.registry import registry as loss_registry  # noqa: F401 - used in lazy loading
    from accelerator.acceleration.registry import registry as acceleration_registry  # noqa: F401 - used in lazy loading


_lazy_map = {
    "Context": ("accelerator.runtime.context", "Context"),
    "PipelineManager": ("accelerator.runtime.pipeline.manager", "PipelineManager"),
    "PipelineStep": ("accelerator.runtime.pipeline.manager", "PipelineStep"),
    "CheckpointManager": (
        "accelerator.runtime.checkpoint_manager.manager",
        "CheckpointManager",
    ),
    "operation_registry": (
        "accelerator.runtime.operations.operation_registry",
        "registry",
    ),
    "loss_registry": ("accelerator.runtime.loss.registry", "registry"),
    "acceleration_registry": ("accelerator.acceleration.registry", "registry"),
}


def __getattr__(name):
    try:
        mod_path, attr = _lazy_map[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(mod_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


log.info("Framework `Accelerator` is ready to be used!")


__all__ = list(_lazy_map.keys()) + _custom_patches


def __dir__():
    """Keep tab-completion & help() honest without importing heavy stuff."""
    return sorted(set(globals()) | _lazy_map.keys())
