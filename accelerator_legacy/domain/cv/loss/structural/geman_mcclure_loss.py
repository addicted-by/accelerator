from accelerator.domain.cv.utils import tv
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='geman_mcclure_extended')
def geman_mcclure_extended(net_result, ground_truth, eps=0.25, *_, **kwargs):
    return (((net_result - ground_truth).pow(2)) / ((net_result - ground_truth).pow(2) + ground_truth.pow(2) + net_result.pow(2) + eps).mean()).mean()
@registry.register_loss(LossType.IMG2IMG, name='tv_geman_mcclure_extended')
def tv_geman_mcclure_extended(net_result, ground_truth, eps=0.25, *_, **kwargs):
    img1, img2 = tv(net_result), tv(ground_truth)
    return (((img1 - img2).pow(2)) / ((img1 - img2).pow(2) + img2.pow(2) + img1.pow(2) + eps).mean()).mean()