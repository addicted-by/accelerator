from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry
from .common import rename_item

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def remove_module_prefix(model, ckpt_model_dict):
    log.info("\tremove_module:")

    ckpt_dict_keys = set(ckpt_model_dict.keys())
    for k in ckpt_dict_keys:
        if k.startswith("module."):
            rename_item(k, k[len("module.") :], ckpt_model_dict)
            continue
