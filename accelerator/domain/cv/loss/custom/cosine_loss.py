import torch
from accelerator.runtime.loss import registry, LossType, LossWrapper


@registry.register_loss(LossType.CUSTOM, name='cosine_loss')
class CosineLoss(LossWrapper):
    def __init__(
        self, 
        prediction_key = None,
        target_key = None,
        loss_coefficient = 1,
        num_layer=3,
        **kwargs
    ):
        super().__init__(prediction_key, target_key, loss_coefficient, **kwargs)
        # super(CosineLoss, self).__init__()
        self.num_layer = num_layer
        # self._transform_pipeline.append() # build pyramid

    def forward(
        self, 
        net_result, 
        ground_truth, 
        *args, 
        **kwargs
    ):
        net_pyr = net_result #self.pyr_builder.build_gauss_pyr(net_result, self.num_layer) 
        label_pyr = ground_truth
        x_norm = net_pyr[0] / (net_pyr[0].norm(dim=1, keepdim=True) + 1e-3)
        y_norm = label_pyr[0] / (label_pyr[0].norm(dim=1, keepdim=True) + 1e-3)
        cos_sim = torch.cosine_similarity(x_norm, y_norm, dim=1)
        cosine_loss = torch.mean(1. - cos_sim)

        return cosine_loss
    