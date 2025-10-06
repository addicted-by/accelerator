import torch
from ..registry import registry, OperationType
from accelerator.runtime.model.gamma import Gamma
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)

@registry.register_operation(OperationType.CKPT_TRANSFORMS)
def change_gamma_forward_to_experimental_gamma(model, ckpt_model_dict):
    ckpt_model_config = model.model_config
    log.info('='*50, 'Change gamma to experimental gamma', '='*50)

    log.info(ckpt_model_dict.keys())
    log.info(ckpt_model_config)

    state_dict_keys = set(model.state_dict().keys())
    ckpt_dict_keys = set(ckpt_model_dict.keys())

    diff_state = state_dict_keys.difference(ckpt_dict_keys)
    diff_ckpt = ckpt_dict_keys.difference(state_dict_keys)
    log.info('diff_state:', diff_state)
    log.info('diff_ckpt:', diff_ckpt)

    ###############################################################
    ########### CREATE GAMMA AS IN CHECKPOINTED MODEL #############
    ###############################################################
    gamma_ckpt = Gamma(ckpt_model_config)
    if ckpt_model_config['gamma']:
        if ckpt_model_config['gamma_type'] in [1430, 1450]:
            pass
        elif ckpt_model_config['gamma_type'] in [143, 145]:
            state_dict_for_gamma_ckpt = {
                                        'b1': ckpt_model_dict['gamma.b1'],
                                        'c1': ckpt_model_dict['gamma.c1'],
                                        'd1': ckpt_model_dict['gamma.d1'],
                                        'e1': ckpt_model_dict['gamma.e1'],
                                        'c2': ckpt_model_dict['gamma.c2'],
                                        'd2': ckpt_model_dict['gamma.d2'],
                                        'e2': ckpt_model_dict['gamma.e2'],
                                        }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        elif [743, 843, 943, 945, 947]:
            state_dict_for_gamma_ckpt = {
                                        'b1_arr': ckpt_model_dict['gamma.b1_arr'],
                                        'c1_arr': ckpt_model_dict['gamma.c1_arr'],
                                        'd1_arr': ckpt_model_dict['gamma.d1_arr'],
                                        'e1_arr': ckpt_model_dict['gamma.e1_arr'],
                                        'c2_arr': ckpt_model_dict['gamma.c2_arr'],
                                        'd2_arr': ckpt_model_dict['gamma.d2_arr'],
                                        'e2_arr': ckpt_model_dict['gamma.e2_arr'],
                                        }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        elif ckpt_model_config['gamma_type'] in [1110, 1120]:
            pass
        elif ckpt_model_config['gamma_type'] in [111, 112, 911, 912]:
            state_dict_for_gamma_ckpt = {
                                        'p_forward_arr': ckpt_model_dict['gamma.p_forward_arr'],
                                        'p_inverse_arr': ckpt_model_dict['gamma.p_inverse_arr']
                                        }
            gamma_ckpt.load_state_dict(state_dict_for_gamma_ckpt)
        else:
            raise ValueError(f"Unknown type of gamma {ckpt_model_config['gamma_type']}, please check the transform!")

    ###############################################################
    ######### GET VALUES IN QUERY POINTS FOR NEW GAMMA ############
    ###############################################################
    x_forward = torch.linspace(0, 1, model.gamma._gamma_forward.gamma_frames.resolution)

    x_frames_forward = x_forward.reshape(1, 1, 1, -1).repeat(1, model.gamma._gamma_forward.gamma_frames.num_channels, 1, 1)
    if hasattr(model.gamma._gamma_forward, 'gamma_meta'):
        x_meta_forward = x_forward.reshape(1, 1, 1, -1).repeat(1, model.gamma._gamma_forward.gamma_meta.num_channels, 1, 1)
    else:
        x_meta_forward = torch.rand(1, 1, 1, 1)


    y_frames_forward, y_meta_forward, _ = gamma_ckpt._gamma_forward(x_frames_forward, x_meta_forward, 0)

    ###############################################################
    ####### Remove unnecessary keys from old gamma-degamma ########
    ###############################################################
    ckpt_model_dict.pop('gamma.b1_arr', None)
    ckpt_model_dict.pop('gamma.c1_arr', None)
    ckpt_model_dict.pop('gamma.d1_arr', None)
    ckpt_model_dict.pop('gamma.e1_arr', None)
    ckpt_model_dict.pop('gamma.p_forward', None)
    ckpt_model_dict.pop('gamma.p_forward_arr', None)

    ###############################################################
    ############## UPDATE CHECKPOINT WITH NEW VALUES ##############
    ###############################################################
    ckpt_model_dict['gamma._gamma_forward.gamma_frames.deltas'] = model.gamma._gamma_forward.gamma_frames.get_deltas(y_frames_forward.squeeze())
    if model.gamma._gamma_forward.gamma_frames.trainable_endpoints:
        ckpt_model_dict['gamma._gamma_forward.gamma_frames.start_points'] = torch.zeros(model.gamma._gamma_forward.gamma_frames.num_channels, 1)
    if hasattr(model.gamma._gamma_forward, 'gamma_meta'):
        log.info('!'*100)
        log.info('!'*20, 'NOT WORKING RIGHT NOW','!'*20)
        log.info('!'*100)
        assert False, "Dude, not again!"
        ckpt_model_dict['gamma._gamma_forward.gamma_meta.deltas'] = model.gamma._gamma_forward.gamma_meta.get_deltas(y_meta_forward.squeeze())
        if model.gamma._gamma_forward.gamma_meta.trainable_endpoints:
            ckpt_model_dict['gamma._gamma_forward.gamma_meta.start_points'] = torch.zeros(model.gamma._gamma_forward.gamma_meta.num_channels, 1)
    
    log.info("GAMMA FORWARD FRAMES BEFORE:")
    log.info(model.gamma._gamma_forward.gamma_frames.deltas)
    if hasattr(model.gamma._gamma_forward, 'gamma_meta'):
        log.info("GAMMA FORWARD META BEFORE:")
        log.info(model.gamma._gamma_forward.gamma_meta.deltas)
  
    model.load_state_dict(ckpt_model_dict, strict=True)

    log.info("GAMMA FORWARD FRAMES AFTER:")
    log.info(model.gamma._gamma_forward.gamma_frames.deltas)
    if hasattr(model.gamma._gamma_forward, 'gamma_meta'):
        log.info("GAMMA FORWARD META AFTER:")
        log.info(model.gamma._gamma_forward.gamma_meta.deltas)