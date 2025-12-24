import torch

from accelerator.runtime.model.gamma import Gamma
from accelerator.utilities.logging import get_logger

from ..registry import OperationType, registry

log = get_logger(__name__)


@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def change_gamma_inverse_to_experimental_gamma(model, ckpt_model_dict):
    ckpt_model_config = model.model_config
    print("=" * 50, "Change gamma to experimental gamma", "=" * 50)

    print(ckpt_model_dict.keys())
    print(ckpt_model_config)

    state_dict_keys = set(model.state_dict().keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    print("diff_state:", diff_state)
    print("diff_ckpt:", diff_ckpt)

    ###############################################################
    ########### CREATE GAMMA AS IN CHECKPOINTED MODEL #############
    ###############################################################
    gamma_ckpt = Gamma(ckpt_model_config)
    if ckpt_model_config["gamma"]:
        if ckpt_model_config["gamma_type"] in [1430, 1450]:
            pass
        elif ckpt_model_config["gamma_type"] in [143, 145]:
            state_dict_for_gamma_ckpt = {
                "b1": ckpt_model_dict["gamma.b1"],
                "c1": ckpt_model_dict["gamma.c1"],
                "d1": ckpt_model_dict["gamma.d1"],
                "e1": ckpt_model_dict["gamma.e1"],
                "c2": ckpt_model_dict["gamma.c2"],
                "d2": ckpt_model_dict["gamma.d2"],
                "e2": ckpt_model_dict["gamma.e2"],
            }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        elif [743, 843, 943, 945, 947]:
            state_dict_for_gamma_ckpt = {
                "b1_arr": ckpt_model_dict["gamma.b1_arr"],
                "c1_arr": ckpt_model_dict["gamma.c1_arr"],
                "d1_arr": ckpt_model_dict["gamma.d1_arr"],
                "e1_arr": ckpt_model_dict["gamma.e1_arr"],
                "c2_arr": ckpt_model_dict["gamma.c2_arr"],
                "d2_arr": ckpt_model_dict["gamma.d2_arr"],
                "e2_arr": ckpt_model_dict["gamma.e2_arr"],
            }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        elif ckpt_model_config["gamma_type"] in [1110, 1120]:
            pass
        elif ckpt_model_config["gamma_type"] in [111, 112, 911, 912]:
            state_dict_for_gamma_ckpt = {
                "p_forward_arr": ckpt_model_dict["gamma.p_forward_arr"],
                "p_inverse_arr": ckpt_model_dict["gamma.p_inverse_arr"],
            }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        else:
            raise ValueError(f"Unknown type of gamma {ckpt_model_config['gamma_type']}, please check the transform!")

    ###############################################################
    ######### GET VALUES IN QUERY POINTS FOR NEW GAMMA ############
    ###############################################################
    x_forward = torch.linspace(0, 1, model.gamma._gamma_inverse.gamma_frames.resolution)

    x_frames_inverse = x_forward.reshape(1, 1, 1, -1).repeat(
        1, model.gamma._gamma_inverse.gamma_frames.num_channels, 1, 1
    )

    y_frames_inverse = gamma_ckpt._gamma_inverse(x_frames_inverse)

    ###############################################################
    ####### Remove unnecessary keys from old gamma-degamma ########
    ###############################################################
    ckpt_model_dict.pop("gamma.c2_arr", None)
    ckpt_model_dict.pop("gamma.d2_arr", None)
    ckpt_model_dict.pop("gamma.e2_arr", None)
    ckpt_model_dict.pop("gamma.p_inverse_arr", None)

    ###############################################################
    ############## UPDATE CHECKPOINT WITH NEW VALUES ##############
    ###############################################################
    ckpt_model_dict["gamma._gamma_inverse.gamma_frames.deltas"] = model.gamma._gamma_inverse.gamma_frames.get_deltas(
        y_frames_inverse.squeeze()
    )
    if model.gamma._gamma_inverse.gamma_frames.trainable_endpoints:
        ckpt_model_dict["gamma._gamma_inverse.gamma_frames.start_points"] = torch.zeros(
            model.gamma._gamma_inverse.gamma_frames.num_channels, 1
        )

    print("GAMMA INVERSE FRAMES BEFORE:")
    print(model.gamma._gamma_inverse.gamma_frames.deltas)

    model.load_state_dict(ckpt_model_dict, strict=True)

    print("GAMMA INVERSE FRAMES AFTER:")
    print(model.gamma._gamma_inverse.gamma_frames.deltas)
