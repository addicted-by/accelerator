from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def change_gamma_1430_to_143(model, ckpt_model_dict):
    log.info("===change_gamma_1430_to_143===")
    state_dict_keys = set(model.state_dict().keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    log.info("diff_state:", diff_state)
    log.info("diff_ckpt:", diff_ckpt)

    if len(diff_ckpt) == 0 and len(diff_state) == 7 and "gamma.b1" in diff_state:
        model.load_state_dict(ckpt_model_dict, strict=False)
    else:
        raise ValueError("Checkpoint does not match model description.")
