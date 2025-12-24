from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def change_gamma_1430_to_943_add_dynamic_max_qpp(model, ckpt_model_dict):
    log.info("===change_gamma_1430_to_943 and add_dynamic_max_qpp===")
    state_dict = model.state_dict()
    state_dict_keys = set(state_dict.keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    log.info("CKPT DICT KEYS")
    log.info(ckpt_dict_keys)
    log.info("STATE DICT KEYS")
    log.info(state_dict_keys)

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    log.info("diff_state:", diff_state)
    log.info("diff_ckpt:", diff_ckpt)

    ckpt_model_dict["qpp.qpp_coefs"] = state_dict["qpp.qpp_coefs"]
    ckpt_model_dict["gamma.b1_arr"] = state_dict["gamma.b1_arr"]
    ckpt_model_dict["gamma.c1_arr"] = state_dict["gamma.c1_arr"]
    ckpt_model_dict["gamma.d1_arr"] = state_dict["gamma.d1_arr"]
    ckpt_model_dict["gamma.e1_arr"] = state_dict["gamma.e1_arr"]
    ckpt_model_dict["gamma.c2_arr"] = state_dict["gamma.c2_arr"]
    ckpt_model_dict["gamma.d2_arr"] = state_dict["gamma.d2_arr"]
    ckpt_model_dict["gamma.e2_arr"] = state_dict["gamma.e2_arr"]

    ckpt_model_dict.pop("tail_jdd.upcon_full2.weight", None)

    model.load_state_dict(ckpt_model_dict, strict=True)
