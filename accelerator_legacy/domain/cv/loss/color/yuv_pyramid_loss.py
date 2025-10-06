import torch


from accelerator.domain.cv.utils import tv
from accelerator.runtime.loss import LossWrapper, registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='yuv_pyramid_loss')
class YUVLoss(LossWrapper):
    def __init__(
        self, 
        prediction_key=None, 
        target_key=None, 
        loss_coefficient=1, 
        num_layer: int=6,
        lf_lvl: int=4,
        mf_lvl: int=3,
        divide_on_label_tv: bool = False,
        **kwargs
    ):
        super().__init__(prediction_key, target_key, loss_coefficient, **kwargs)

        
        # self._transform_pipeline.append() #: rgb2yuv+buildpyramid

        self.num_layer = num_layer
        self.lf_lvl = lf_lvl
        self.mf_lvl = mf_lvl
        self.divide_on_label_tv = divide_on_label_tv


    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth, 
        *args, 
        **kwargs
    ):
        net_yuv_pyr = net_result #self.pyr_builder.build_gauss_pyr(rgb2yuv(net_result, self.device), self.num_layer)
        label_yuv_pyr = ground_truth #self.pyr_builder.build_gauss_pyr(rgb2yuv(ground_truth, self.device), self.num_layer)
        
        lf_uv_loss = 0.0
        for layer in range(self.lf_lvl):  # 0 1 2 3
            grad_net_yuv = tv(net_yuv_pyr[layer])
            grad_label_yuv = tv(label_yuv_pyr[layer])

            grad_net_uv = grad_net_yuv[:, 1:, :, :]
            grad_label_uv = grad_label_yuv[:, 1:, :, :]
            grad_label_y = grad_label_yuv[:, :1, :, :]
            if self.divide_on_label_tv:
                lf_perc = torch.abs(grad_label_y) + float(1e-5)
            else:
                lf_perc = torch.abs(label_yuv_pyr[layer][:, :1, :, :]) + float(1e-3)
                lf_perc = lf_perc[..., None]
            if layer < self.mf_lvl:  # 0 1 2
                lf_uv_loss += torch.mean((torch.abs(grad_net_uv - grad_label_uv)) / lf_perc)
            else:  # 3
                tv_net_lf = torch.mean(torch.abs(grad_net_uv), dim=-1)
                tv_label_lf = torch.mean(torch.abs(grad_label_uv), dim=-1)
                if self.divide_on_label_tv:
                    lf_uv_loss += torch.mean(torch.abs(tv_net_lf - tv_label_lf) / torch.mean(lf_perc, dim=-1))
                else:
                    lf_uv_loss += torch.mean(torch.abs(tv_net_lf - tv_label_lf) / lf_perc.squeeze(-1))

        return lf_uv_loss