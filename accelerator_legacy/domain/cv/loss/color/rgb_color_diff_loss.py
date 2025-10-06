import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


from accelerator.runtime.loss import registry, LossType
from accelerator.typings.base import _DEVICE


@registry.register_loss(LossType.IMG2IMG, name='rgb_color_diff_loss')
class rgbColordiff_v1(nn.Module):
    def __init__(
        self, 
        *,
        device: _DEVICE = 'cuda', 
        eps: float = 1e-4, 
        loss_mode: str = "l1", 
        reduction: str = "mean", 
        gaussian_blur: bool = False,
        mask: float = 1.0,
        diff_map_out: bool = False,
        RB_diff_flag: bool = True,
        **kwargs
    ):
        super().__init__()

        self.eps = eps
        if loss_mode == "l1":
            self.Loss_fn = nn.L1Loss(reduction=reduction)
        else:
            self.Loss_fn = nn.MSELoss(reduction=reduction)

        self.gaussian_blur = gaussian_blur
        # print(f"rgb color diff loss, loss mode: {loss_mode}")
        self.mask = mask
        self.diff_map_out = diff_map_out
        self.RB_diff_flag = RB_diff_flag
        
        if gaussian_blur:
            sigma = 0.5
            self.kernel = self.gauss_kernel(sigma=sigma, device=device)

    @ staticmethod
    def gauss_kernel(size=5, sigma=1.0, device=torch.device('cpu')):
        import cv2
        kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = torch.from_numpy(kernel / np.sum(kernel))
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def convolve(self, img, scale=1.0):
        pad = F.pad(img, (2, 2, 2, 2), mode='constant')#mode='reflect')
        return F.conv2d(pad, self.kernel * scale, groups=3)

    def apply_gaussian_blur(self, img):
        return self.convolve(img)

    def forward(self, out, gt, *_, **kwargs):
        # NCHW format, should use orig data input
        if self.gaussian_blur:
            gt = self.apply_gaussian_blur(gt)
            out = self.apply_gaussian_blur(out)

        gt = gt * self.mask
        out = out * self.mask

        outRG_diff = out[:, 0:1, ...] - out[:, 1:2, ...]
        outBG_diff = out[:, 2:3, ...] - out[:, 1:2, ...]
        outRB_diff = out[:, 0:1, ...] - out[:, 2:3, ...]

        gtRG_diff = gt[:, 0:1, ...] - gt[:, 1:2, ...]
        gtBG_diff = gt[:, 2:3, ...] - gt[:, 1:2, ...]
        gtRB_diff = gt[:, 0:1, ...] - gt[:, 2:3, ...]

        if self.RB_diff_flag:
            consistencyLoss = (
                self.Loss_fn(outRG_diff, gtRG_diff) + self.Loss_fn(outBG_diff, gtBG_diff) + self.Loss_fn(outRB_diff, gtRB_diff)
            )
        else:
            consistencyLoss = self.Loss_fn(outRG_diff, gtRG_diff) + self.Loss_fn(outBG_diff, gtBG_diff)

        if self.diff_map_out:
            RG_consistency_mask = torch.abs(outRG_diff - gtRG_diff)
            BG_consistency_mask = torch.abs(outBG_diff - gtBG_diff)
            RB_consistency_mask = torch.abs(outRB_diff - gtRB_diff)
            return consistencyLoss, RG_consistency_mask, BG_consistency_mask, RB_consistency_mask
        else:
            return consistencyLoss       