from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry
from .common import add_item

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def add_default_gamma(model, ckpt_model_dict):
    log.info("\tadd_default_gamma:")

    state_dict = model.state_dict()
    state_dict_keys = set(state_dict.keys())

    for k in state_dict_keys:
        if k.startswith("gamma."):
            add_item(k, state_dict[k], ckpt_model_dict)
