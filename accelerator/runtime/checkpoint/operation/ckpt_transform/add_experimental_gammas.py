from ..registry import registry, OperationType
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def add_experimental_gammas(model, ckpt_model_dict):
    log.info('='*50, 'ADD experimental gammas', '='*50)

    log.info(ckpt_model_dict.keys())

    state_dict_keys = set(model.state_dict().keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    log.info('diff_state:', diff_state)
    log.info('diff_ckpt:', diff_ckpt)

    ###############################################################
    ############## CREATE GAMMA PARAMETERS IN CKPT ################
    ###############################################################
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        if 'gamma' in key:
            log.info(f"Adding {key}")
            ckpt_model_dict[key] = value