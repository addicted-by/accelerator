from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry
from .common import remove_item

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def remove_is_rep(model, ckpt_model_dict):
    log.info("\tRemove_is_rep_and_pretrain_count:")
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    for k in ckpt_dict_keys:
        if k.endswith(".is_rep"):
            remove_item(k, ckpt_model_dict)
            continue
