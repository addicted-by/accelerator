import cv2
import torch
import numpy as np
import torch.nn.functional as F


class BuildPyramid(object):
    def __init__(self, device='cuda', interp_downsample=False):
        sigma = 0.5
        self.kernel = self.gauss_kernel(sigma=sigma, device=device)
        self.interp_downsample = interp_downsample

    @ staticmethod
    def gauss_kernel(size=5, sigma=1.0, device=torch.device('cpu')):
        kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = torch.from_numpy(kernel / np.sum(kernel))
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def convolve(self, img, scale=1.0):
        pad = F.pad(img, (2, 2, 2, 2), 'constant') #mode='reflect')
        return F.conv2d(pad, self.kernel * scale, groups=3)

    def downsample(self, img):
        conv = self.convolve(img)
        if self.interp_downsample:
            ds = F.interpolate(conv, scale_factor=0.5, mode='bilinear', align_corners=False)
            # print('ds.shape', ds.shape)
            # print('conv[:, :, ::2, ::2].shape', conv[:, :, ::2, ::2].shape)
        else:
            ds = conv[:, :, ::2, ::2]
        return ds

    def upsample(self, img):
        up = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
        conv = self.convolve(up)
        return conv

    def build_gauss_pyr(self, img, num_levels):
        gauss_pyr = [img]
        for lvl in range(num_levels - 1):
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