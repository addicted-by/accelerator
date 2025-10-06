import torch

from accelerator.runtime.loss import LossWrapper, registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='gauss_pyramid_loss')
class GaussPyramidLoss(LossWrapper):
    def __init__(
        self, 
        prediction_key=None, 
        target_key=None, 
        loss_coefficient=1,
        num_layer: int=5,
        lf_lvl: int=3,
        **kwargs
    ):
        super().__init__(prediction_key, target_key, loss_coefficient, **kwargs)

        # self._transform_pipeline.append() #: rgb2yuv+buildpyramid

        self.num_layer = num_layer
        self.lf_lvl = lf_lvl

    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth, 
        *args, 
        **kwargs
    ):
        net_yuv_pyr = net_result #self.pyr_builder.build_gauss_pyr(rgb2yuv(net_result, self.device), self.num_layer)
        label_yuv_pyr = ground_truth #self.pyr_builder.build_gauss_pyr(rgb2yuv(ground_truth, self.device), self.num_layer)

        gauss_pyramid_loss = 0.

        for layer in range(self.lf_lvl):
            net_uv = net_yuv_pyr[layer][:, 1:, ...]
            label_uv = label_yuv_pyr[layer][:, 1:, ...]

            gauss_pyramid_loss += torch.mean(torch.abs(net_uv - label_uv))

        return gauss_pyramid_loss
    