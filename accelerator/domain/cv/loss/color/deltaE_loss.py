from typing import Iterable
import torch
import torch.nn.functional as F


from accelerator.runtime.loss import registry, LossType
from accelerator.domain.cv.utils import gaussian_blur



@registry.register_loss(LossType.IMG2IMG, name='color_difference')
class ColorDiffLoss(torch.nn.Module):
    """
    """
    def __init__(
            self, 
            *,
            device, 
            mode: str='lab', 
            scales=1, 
            kernels=1, 
            weights=1, 
            coef={'l': 1, 'a': 1, 'b': 1}, 
            use_abs=True,
            use_mask=True, 
            use_l_smape=False, 
            eps=1e-16,
            **kwargs
        ):
        
        super().__init__()
        # super(ColorDiffLoss, self).__init__()

        self.device = device
        self.mode = mode
        self.eps = eps
        self.coef = coef
        self.use_abs = use_abs
        self.use_mask = use_mask
        self.use_l_smape = use_l_smape
      
        if not isinstance(weights, Iterable) and not isinstance(kernels, Iterable) and isinstance(scales, Iterable):
            
            self.weights = [weights] * len(scales)
            self.kernels = [kernels] * len(scales)
            self.scales = scales
            
        elif not isinstance(weights, Iterable) and isinstance(kernels, Iterable) and not isinstance(scales, Iterable):
            
            self.weights = [weights] * len(kernels)
            self.scales = [scales] * len(kernels)
            self.kernels = kernels
            
        elif isinstance(weights, Iterable) and not isinstance(kernels, Iterable) and not isinstance(scales, Iterable):
            
            self.weights = weights
            self.kernels = [kernels] * len(weights)
            self.scales = [scales] * len(weights)
            
        elif not isinstance(weights, Iterable) and isinstance(kernels, Iterable) and isinstance(scales, Iterable):
            
            if len(kernels) != len(scales):
                raise ValueError('"kernels" and "scales" must have the same length')
            
            self.weights = [weights] * len(kernels)
            self.kernels = kernels
            self.scales = scales
            
        elif isinstance(weights, Iterable) and not isinstance(kernels, Iterable) and isinstance(scales, Iterable):
            
            if len(weights) != len(scales):
                raise ValueError('"weights" and "scales" must have the same length')
            
            self.weights = weights
            self.kernels = [kernels] * len(weights)
            self.scales = scales
            
        elif isinstance(weights, Iterable) and isinstance(kernels, Iterable) and not isinstance(scales, Iterable):
            
            if len(weights) != len(kernels):
                raise ValueError('"weights" and "kernels" must have the same length')
            
            self.weights = weights
            self.kernels = kernels
            self.scales = [scales] * len(weights)
            
        elif not isinstance(weights, Iterable) and not isinstance(kernels, Iterable) and not isinstance(scales, Iterable):
            self.weights = [weights]
            self.kernels = [kernels]
            self.scales = [scales]
            
        else:
            
            if len(weights) != len(kernels) or len(scales) != len(kernels):
                raise ValueError('"weights", "scales" and "kernels" must have the same length')
            
            self.weights = weights
            self.kernels = kernels
            self.scales = scales
            
    def rgb2xyz(self, rgb):  # rgb from [0,1]
        # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

        mask = (rgb > .04045).type(torch.FloatTensor)
        mask = mask.to(self.device)

        rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

        x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
        y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
        z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
        out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

        return out

    def xyz2lab(self, xyz):
        # 0.95047, 1., 1.08883 # white
        sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
        sc = sc.to(self.device)

        xyz_scale = xyz / sc

        mask = (xyz_scale > .008856).type(torch.FloatTensor)
        mask = mask.to(self.device)

        xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

        L = 116. * xyz_int[:, 1, :, :] - 16.
        a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
        b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
        out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

        return out

    def rgb2lab(self, rgb):
        xyz = self.rgb2xyz(rgb)
        lab = self.xyz2lab(xyz)
        l_rs = (lab[:, [0], :, :] - 50.) / 100.
        ab_rs = lab[:, 1:, :, :] / 110.
        out = torch.cat((l_rs, ab_rs), dim=1)
        return out
    
    def l_smape(self, net_result_l, ground_truth_l):
        smape = 1 / (torch.abs(net_result_l) + torch.abs(ground_truth_l) + 1e-6).detach()
        return smape

    def forward(
        self, 
        net_result, 
        ground_truth,
        *_,
        **kwargs    
    ):
        mask1 = torch.where(net_result >= 0.0, torch.ones_like(net_result), -torch.ones_like(net_result))
        mask2 = torch.where(ground_truth >= 0.0, torch.ones_like(ground_truth), -torch.ones_like(ground_truth))
        
        if self.use_abs:
            net_result = torch.clamp(torch.abs(net_result), min=1e-15)
            ground_truth = torch.clamp(torch.abs(ground_truth), min=1e-15)
        
        if self.use_mask:
            net_result_lab = mask1 * self.rgb2lab(net_result)
            ground_truth_lab = mask2 * self.rgb2lab(ground_truth)
        else:
            net_result_lab = self.rgb2lab(net_result)
            ground_truth_lab = self.rgb2lab(ground_truth)

        errs = 0
        for weight, kernel_size, scale_factor in zip(self.weights, self.kernels, self.scales):
            
            if scale_factor != 1:
                scale_factor = 1 / scale_factor
                net_result_lab = F.interpolate(net_result_lab, scale_factor=scale_factor)
                ground_truth_lab = F.interpolate(ground_truth_lab, scale_factor=scale_factor)
                
            if kernel_size > 1:
                net_result_lab = gaussian_blur(net_result_lab, kernel_size=kernel_size, sigma=1)
                ground_truth_lab = gaussian_blur(ground_truth_lab, kernel_size=kernel_size, sigma=1)
            
            if self.mode == 'lab':
                err = torch.sqrt(
                    self.coef['l'] * (net_result_lab[:, 0, ...] - ground_truth_lab[:, 0, ...]) ** 2 + \
                    self.coef['a'] * (net_result_lab[:, 1, ...] - ground_truth_lab[:, 1, ...]) ** 2 + \
                    self.coef['b'] * (net_result_lab[:, 2, ...] - ground_truth_lab[:, 2, ...]) ** 2 + self.eps
                )
                
            elif self.mode  == 'ab':
                err = torch.sqrt(
                    self.coef['a'] * (net_result_lab[:, 1, ...] - ground_truth_lab[:, 1, ...]) ** 2 + \
                    self.coef['b'] * (net_result_lab[:, 2, ...] - ground_truth_lab[:, 2, ...]) ** 2 + self.eps
                )

            elif self.mode == 'no_sqrt_lab':
                err = (
                    self.coef['l'] * (net_result_lab[:, 0, ...] - ground_truth_lab[:, 0, ...]) ** 2 + \
                    self.coef['a'] * (net_result_lab[:, 1, ...] - ground_truth_lab[:, 1, ...]) ** 2 + \
                    self.coef['b'] * (net_result_lab[:, 2, ...] - ground_truth_lab[:, 2, ...]) ** 2
                )

            elif self.mode == 'no_sqrt_ab':
                err = (
                    self.coef['a'] * (net_result_lab[:, 1, ...] - ground_truth_lab[:, 1, ...]) ** 2 + \
                    self.coef['b'] * (net_result_lab[:, 2, ...] - ground_truth_lab[:, 2, ...]) ** 2
                )

            else:
                raise ValueError(f'The mode {self.mode} is not supported in color loss!')
            
            if self.use_l_smape:
                smape = self.l_smape(net_result_lab[:, 0], ground_truth_lab[:, 0])
                err = err * smape
            
            errs = errs + torch.mean(err) * weight

             
        # error = torch.mean(errs,dim=[*range(1,len(errs.shape))])
        error = torch.mean(errs)

        return error
    
