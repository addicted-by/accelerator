from .common import remove_item
from ..registry import registry, OperationType
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def remove_pretrain_count(model, ckpt_model_dict):
    log.info('\tRemove_pretrain_count:')
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    for k in ckpt_dict_keys:
        if k == "pretrain_count":
            remove_item(k, ckpt_model_dict)
            continue
