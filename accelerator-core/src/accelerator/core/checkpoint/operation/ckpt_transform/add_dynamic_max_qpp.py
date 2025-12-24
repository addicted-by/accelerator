from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def add_dynamic_max_qpp(model, ckpt_model_dict):
    log.info("===add_dynamic_max_qpp===")
    state_dict = model.state_dict()
    state_dict_keys = set(state_dict.keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    log.info("diff_state:", diff_state)
    log.info("diff_ckpt:", diff_ckpt)

    ckpt_model_dict["qpp.qpp_coefs"] = state_dict["qpp.qpp_coefs"]
    ckpt_model_dict.pop("tail_jdd.upcon_full2.weight", None)

    model.load_state_dict(ckpt_model_dict, strict=True)
