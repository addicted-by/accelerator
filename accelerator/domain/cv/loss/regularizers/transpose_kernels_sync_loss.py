from accelerator.runtime.loss import LossWrapper
import torch 


def activate_conv_transpose_input_norm(model, model_config):
    if model_config['net_version'] in [
        'still_fov_fused', 
        'still_fov_fused_no_double_relu', 
        'still_fov_pura', 
        'BTM_for_still_fov', 
        'BTM_for_still_fov_v2', 
        'still_fov_fused_reduced_input'
    ]:
        if model_config['d2s']:
            convTs = [model.model_core.Conv2dTranspose_new1, model.model_core.Conv2dTranspose_new2, model.model_core.Conv2dTranspose_new3]
        else:
            convTs = [model.model_core.Conv2dTranspose_new1, model.model_core.Conv2dTranspose_new2, model.model_core.Conv2dTranspose_new3, model.model_core.Conv2dTranspose_new4]
    else:
        convTs = [model.model_core.upconv2, model.model_core.upconv1, model.model_core.upcon_full]

    for ct in convTs:
        ct.calculate_input_norm = True
    
    return convTs

def get_transp_kernel_sync_loss(input_norm, weight, loss_type):
    assert input_norm is not None
    if loss_type == 1:
        w = torch.unsqueeze(torch.transpose(weight, 0, 1), 1)  # F, 1, C , k, k
        n = torch.unsqueeze(input_norm, 0)  # l1 norm with shape: 1, N, C, 1, 1
        z = n * w  # F, N, C, k, k
        z = torch.norm(z, dim=2)  # F, N, k, k
        return torch.mean(torch.std(z, dim=(2, 3)))
    else:
        raise NotImplementedError()

class TransposeKernelsSyncLoss(LossWrapper):
    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth, 
        *args, 
        **kwargs
    ):
        convTs = kwargs.get('convTs', None)
        if convTs is None:
            raise ValueError("TransposeKernelsSyncLoss requires convTs as input")
    
        convT_loss = 0
        for ct in convTs:
            convT_loss += get_transp_kernel_sync_loss(
                ct.input_norm, 
                ct.dconv.weight,
                self._cfg['type']
            )
        return self.grad_variance_loss(net_result, ground_truth)

