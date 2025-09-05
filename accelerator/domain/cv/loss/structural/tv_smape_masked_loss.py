from accelerator.domain.cv.utils import tv
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='tv_smape_masked_loss')
def tv_smape_masked_loss(
    net_result,
    ground_truth,
    *_,
    **kwargs
):
    targets_tv = tv(ground_truth)
    model_outputs_tv = tv(net_result)
    
    smape = (targets_tv - model_outputs_tv).abs() / (targets_tv.abs() + model_outputs_tv.abs() + 1e-7)
    tv_smape_masked = ((targets_tv - model_outputs_tv).abs() * smape.detach()).mean()
    return tv_smape_masked