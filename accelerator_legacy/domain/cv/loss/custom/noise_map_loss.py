import torch

from accelerator.typings.base import _DEVICE, BatchTensorType
from accelerator.domain.cv.utils import rggb_to_bayer
from accelerator.runtime.loss.registry import registry, LossType
from ..structural.ssim import ssim


def calculate_noise_map(net_result, frame, device, apply_downsample: bool=False, downsample_type: str="bilinear", p: int=2):
    bayer = rggb_to_bayer(frame)
    
    if apply_downsample:
        downsample = torch.nn.Upsample(scale_factor=0.5, mode=downsample_type, align_corners=None)
        net_result_downsampled = downsample(net_result)
        noise_map_final = torch.zeros_like(bayer).to(device)
        noise_map_final[..., 0::2, 0::2] = bayer[..., 0::2, 0::2] - net_result_downsampled[:, 0, :, :].unsqueeze(1)
        noise_map_final[..., 0::2, 1::2] = bayer[..., 0::2, 1::2] - net_result_downsampled[:, 1, :, :].unsqueeze(1)
        noise_map_final[..., 1::2, 0::2] = bayer[..., 1::2, 0::2] - net_result_downsampled[:, 1, :, :].unsqueeze(1)
        noise_map_final[..., 1::2, 1::2] = bayer[..., 1::2, 1::2] - net_result_downsampled[:, 2, :, :].unsqueeze(1)
        return noise_map_final ** 2 if p == 2 else noise_map_final

    output_R = bayer - net_result[:, 0, :, :].unsqueeze(1)
    output_G = bayer - net_result[:, 1, :, :].unsqueeze(1)
    output_B = bayer - net_result[:, 2, :, :].unsqueeze(1)


    noise_map_final = torch.zeros_like(bayer).to(device)

    noise_map_final[..., 0::2, 0::2] = output_R[:, :, 0::2, 0::2]
    noise_map_final[..., 0::2, 1::2] = output_G[:, :, 0::2, 1::2]
    noise_map_final[..., 1::2, 0::2] = output_G[:, :, 1::2, 0::2]
    noise_map_final[..., 1::2, 1::2] = output_B[:, :, 1::2, 1::2]

    return noise_map_final


@registry.register_loss(LossType.CUSTOM, 'noise_map_loss')
class NoiseMapLoss(torch.nn.Module):
    def __init__(
        self, 
        *,
        device: _DEVICE,
        apply_downsample: bool = False,
        downsample_type: str = 'bilinear',
        p: int = 2, 
        ssim: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.device = device
        self.apply_downsample = apply_downsample
        self.downsample_type = downsample_type
        self.p = p
        self.ssim = ssim
        
        
    def forward(
        self, 
        net_result, 
        ground_truth, 
        inputs: BatchTensorType,
        *args, 
        **kwargs
    ):
        frames, _, *_ = inputs
        if frames is None:
            raise ValueError(
                """
                NoiseMapLoss requires `inputs` to calculate
                the value! Please, provide it!
                """
            )
        noise_map_net = calculate_noise_map(
            net_result, 
            frames[:, :4, ...], 
            self.device,
            apply_downsample=self.apply_downsample,
            downsample_type=self.downsample_type,
            p=self.p
        ) # only normal frame currently
        noise_map_gt = calculate_noise_map(
            ground_truth, 
            frames[:, :4, ...], 
            self.device,
            apply_downsample=self.apply_downsample,
            downsample_type=self.downsample_type,
            p=self.p
        ) # only normal frame currently
                
        
        if self.ssim:
            ssim_loss = ssim(noise_map_net, noise_map_gt, window_size=9, reduction='none', max_val=1.0)
            noise_map_loss = torch.mean(ssim_loss)
        else:
            noise_map_loss = torch.sqrt((noise_map_net - noise_map_gt) ** 2 + 1e-16).mean()
        
        return noise_map_loss