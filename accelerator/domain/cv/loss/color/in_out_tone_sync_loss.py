from typing import Dict, Any
import torch
import numpy as np
import torch.nn.functional as F


from accelerator.runtime.loss import LossWrapper, registry, LossType
from accelerator.typings.base import _DEVICE, BatchTensorType


@registry.register_loss(LossType.IMG2IMG, name='InOutToneSyncLoss')
class InOutToneSyncLoss(LossWrapper):
    def __init__(
        self, 
        device: _DEVICE,
        prediction_key=None, 
        target_key=None, 
        loss_coefficient=1, 
        params: Dict[str, Any]={
            'type': 0,
            'lambda': 0.1,
            'gauss_kernel_sigma': 3.0,
            'gauss_kernel_size': 21,
            'clip_ratio': 0.05,
            'downsample_mode': 'nearest',
        },
        **kwargs
    ):
        super().__init__(prediction_key, target_key, loss_coefficient, **kwargs)
        self.in_out_tone_sync = InOutToneSync(params, device)
        
    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth, 
        inputs: BatchTensorType,
        *args, 
        **kwargs
    ):
        img_in, meta, *_ = inputs
        if img_in is None or meta is None:
            raise ValueError(
                """
                InOutToneSyncLoss requires inputs to calculate
                the value! Please, provide it!
                """
            )
        return self.in_out_tone_sync(img_in, meta, net_result)

class InOutToneSync(torch.nn.Module):
    def __init__(self, params, device=torch.device('cpu')):
        super(InOutToneSync, self).__init__()

        self.device = device
        self.params = params
        self.kernel = self.gauss_kernel(params['gauss_kernel_size'], params['gauss_kernel_sigma'], device)
        self.downsample = torch.nn.Upsample(scale_factor=0.5, mode=params['downsample_mode'], align_corners=None)
        self.padding = [params['gauss_kernel_size'] // 2, params['gauss_kernel_size'] // 2,
                        params['gauss_kernel_size'] // 2, params['gauss_kernel_size'] // 2]

    def get_reference_frame(self, in_):
        return torch.cat(
            (in_[:, 0].unsqueeze(1),
             (in_[:, 1] + in_[:, 2]).unsqueeze(1) / 2.0,
             in_[:, 3].unsqueeze(1)), 1)

    @staticmethod
    def gauss_kernel(size=5, sigma=1.0, device=torch.device('cpu')):
        import cv2
        kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = torch.from_numpy(kernel / np.sum(kernel))
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel.requires_grad = False
        kernel = kernel.to(device)
        return kernel

    def gaussian_blur(self, img):
        img = F.pad(img, self.padding, mode="reflect")
        return F.conv2d(img, self.kernel, groups=3)

    def forward(self, in_, in_meta, out_):
        clip_val = in_meta[:, 0].unsqueeze(1)  # N111

        rf = self.get_reference_frame(in_)  #N3HW
        rf = torch.clamp(rf, torch.zeros_like(clip_val, device=self.device), clip_val)
        rf = self.gaussian_blur(rf)

        out_ = torch.clamp(out_, torch.zeros_like(clip_val, device=self.device), clip_val)  # N3HW
        out_ = self.downsample(out_)
        out_ = self.gaussian_blur(out_)


        diff = rf - out_
        diff = torch.clamp(diff, - clip_val * self.params['clip_ratio'], clip_val * self.params['clip_ratio'])

        if self.params['type'] == 2:
            return torch.mean(diff ** 2.0)
        else:
            raise NotImplementedError(f"type is unknown: {self.params['type']}")

        return None