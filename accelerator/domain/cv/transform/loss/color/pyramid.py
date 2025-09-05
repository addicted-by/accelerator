import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerator.runtime.transform import (
    BaseLossTransform, 
    transforms_registry, 
    TensorTransformType
)


@transforms_registry.register_transform(TensorTransformType.LOSS_TRANSFORM)
class BuildPyramid(BaseLossTransform):
    def __init__(
        self, 
        *_,
        pyramid_type: str = 'gauss',
        num_levels: int = 3,
        sigma: float = 0.5,
        **kwargs
    ):
        super().__init__(
            pyramid_type=pyramid_type, 
            num_levels=num_levels,
            sigma=sigma,
            **kwargs
        )
        self.pyramid_type = pyramid_type
        self.num_levels = num_levels
        self.sigma = sigma
        self.kernel = self.gauss_kernel(sigma=self.sigma, device=self.device)


    @staticmethod
    def gauss_kernel(size=5, sigma=1.0, device=torch.device('cpu')):
        kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma) # pylint: disable=no-member
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = torch.from_numpy(kernel / np.sum(kernel))
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def convolve(self, img, scale=1.0):
        pad = F.pad(img, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(pad, self.kernel * scale, groups=3) # pylint: disable=not-callable

    def downsample(self, img):
        conv = self.convolve(img)
        ds = conv[:, :, ::2, ::2]
        return ds

    def upsample(self, img):
        up = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        conv = self.convolve(up)
        return conv

    def build_gauss_pyr(self, img, num_levels):
        gauss_pyr = [img]
        for _ in range(num_levels - 1):
            gauss_pyr.append(self.downsample(gauss_pyr[-1]))
        return gauss_pyr[::-1]

    def build_laplac_pyr(self, img, num_levels):
        gauss_pyr = self.build_gauss_pyr(img, num_levels)
        laplac_pyr = [gauss_pyr[0]]
        for lvl in range(num_levels - 1):
            hf = gauss_pyr[lvl + 1] - self.upsample(gauss_pyr[lvl])
            laplac_pyr.append(hf)
        return laplac_pyr

    def blend_laplac_pyr(self, laplac_pyr):
        num_levels = len(laplac_pyr)
        img = laplac_pyr[0]
        for lvl in range(num_levels - 1):
            img = laplac_pyr[lvl + 1] + self.upsample(img)
        return img
    
    def _apply_single(self, tensor, **kwargs):
        if self.pyramid_type == 'gauss':
            return self.build_gauss_pyr(tensor, self.num_levels), {}
        elif self.pyramid_type == 'laplac':
            return self.build_laplac_pyr(tensor, self.num_levels), {}
        else:
            raise NotImplementedError(f'Pyramid type {self.pyramid_type} is not implemented')