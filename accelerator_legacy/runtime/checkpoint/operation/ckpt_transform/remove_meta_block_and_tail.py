from .common import rename_item
from ..registry import registry, OperationType
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def remove_meta_block_and_tail(model, ckpt_model_dict):
    log.info('\tremove_meta_block_and_tail:')

    ckpt_dict_keys = set(ckpt_model_dict.keys())
    for k in ckpt_dict_keys:
        if '.meta_block.' in k:
            rename_item(k, k.replace('.meta_block.', '.'), ckpt_model_dict)
            continue

        if k.startswith('tail_jdd.'):
            rename_item(k, k[len('tail_jdd.'):], ckpt_model_dict)
            continue

    ckpt_dict_keys = set(ckpt_model_dict.keys())
    for k in ckpt_dict_keys:
        if k.startswith('conv1_out.'):
            rename_item(k, k.replace('conv1_out.', 'conv1_out.0.'), ckpt_model_dict)
            continue


