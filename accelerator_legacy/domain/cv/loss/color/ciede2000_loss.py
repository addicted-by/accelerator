import torch
import numpy as np
from accelerator.runtime.loss.registry import registry, LossType


@registry.register_loss(LossType.IMG2IMG, name='ciede2000')
class CIEDE2000(torch.nn.Module):
    """
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    """

    def __init__(
        self, 
        *,
        device='cpu', 
        eps: float=1e-16, 
        kL: float=1, 
        kC: float=1, 
        kH: float=1,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.eps = eps
        self.kL = kL
        self.kC = kC
        self.kH = kH
        

    def rgb2xyz(self, rgb):  
        # rgb from [0,1]
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
        return lab
    
    def rgb2lab_normalized(self, rgb):
        lab = self.rgb2lab(rgb)
        l_rs = (lab[:, [0], :, :] - 50) / 100.
        ab_rs = lab[:, 1:, :, :] / 110.
        out = torch.cat((l_rs, ab_rs), dim=1)
        return out

    def _cart2polar_2pi(self, x, y):
        """convert cartesian coordinates to polar (uses non-standard theta range!)

        NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
        """
        r, t = torch.hypot(x, y + self.eps), torch.atan2(y, x)
        t = t + torch.where(t < 0., 2 * torch.pi, 0.0)
        return r, t 

    def forward(
        self, 
        net_result, 
        ground_truth,
        *args,
        **kwargs
    ):
        # print(f'net_result min={net_result.min()}, max={net_result.max()}')
        # print(f'ground_truth min={ground_truth.min()}, max={ground_truth.max()}')

        mask = torch.where(net_result >= 0.0, torch.ones_like(net_result), -torch.ones_like(net_result))
        net_result_lab = mask * self.rgb2lab(torch.clamp(torch.abs(net_result), min=1e-15))

        mask = torch.where(ground_truth >= 0.0, torch.ones_like(ground_truth), -torch.ones_like(ground_truth))
        ground_truth_lab = mask * self.rgb2lab(torch.clamp(torch.abs(ground_truth), min=1e-15))
        
        L1, a1, b1 = net_result_lab[:, 0, ...], net_result_lab[:, 1, ...], net_result_lab[:, 2, ...]
        L2, a2, b2 = ground_truth_lab[:, 0, ...], ground_truth_lab[:, 1, ...], ground_truth_lab[:, 2, ...]

        # Cbar = 0.5 * (torch.hypot(a1, b1) + torch.hypot(a2, b2))
        Cbar = 0.5 * (torch.sqrt(a1 ** 2 + b1 ** 2 + self.eps) + torch.sqrt(a2 ** 2 + b2 ** 2 + self.eps))

        c7 = Cbar ** 7
        G = 0.5 * (1 - torch.sqrt(c7 / (c7 + 25 ** 7 + self.eps) + self.eps))
        scale = 1 + G
        C1, h1 = self._cart2polar_2pi(a1 * scale, b1)
        C2, h2 = self._cart2polar_2pi(a2 * scale, b2)
        
        # recall that c, h are polar coordinates.  c==r, h==theta

        # cide2000 has four terms to delta_e:
        # 1) Luminance term
        # 2) Hue term
        # 3) Chroma term
        # 4) hue Rotation term

        # lightness term
        Lbar = 0.5 * (L1 + L2)
        tmp = (Lbar - 50) ** 2
        SL = 1 + 0.015 * tmp / torch.sqrt(20 + tmp + self.eps)
        L_term = (L2 - L1) / (self.kL * SL + self.eps)

        # chroma term
        Cbar = 0.5 * (C1 + C2)  # new coordinates
        SC = 1 + 0.045 * Cbar
        C_term = (C2 - C1) / (self.kC * SC + self.eps)

        # hue term
        h_diff = h2 - h1
        h_sum = h1 + h2
        CC = C1 * C2

        dH = h_diff.clone().detach()
        dH[h_diff > torch.pi] -= 2 * torch.pi
        dH[h_diff < -torch.pi] += 2 * torch.pi
        dH[CC == 0.] = 0.  # if r == 0, dtheta == 0
        dH_term = 2 * torch.sqrt(CC + self.eps) * torch.sin(dH / 2)

        Hbar = h_sum.clone().detach()
        mask = (CC != 0.) & (torch.abs(h_diff) > torch.pi)
        Hbar[mask * (h_sum < 2 * torch.pi)] += 2 * torch.pi
        Hbar[mask * (h_sum >= 2 * torch.pi)] -= 2 * torch.pi
        Hbar[CC == 0.] *= 2
        Hbar *= 0.5

        T = (1 -
            0.17 * torch.cos(Hbar - np.deg2rad(30)) +
            0.24 * torch.cos(2 * Hbar) +
            0.32 * torch.cos(3 * Hbar + np.deg2rad(6)) -
            0.20 * torch.cos(4 * Hbar - np.deg2rad(63))
            )
        SH = 1 + 0.015 * Cbar * T

        H_term = dH_term / (self.kH * SH + self.eps)

        # hue rotation
        c7 = Cbar ** 7
        Rc = 2 * torch.sqrt(c7 / (c7 + 25 ** 7))
        dtheta = np.deg2rad(30) * torch.exp(-((torch.rad2deg(Hbar) - 275) / 25) ** 2)
        R_term = -torch.sin(2 * dtheta) * Rc * C_term * H_term

        # put it all together
        dE2 = L_term ** 2
        dE2 += C_term ** 2
        dE2 += H_term ** 2
        dE2 += R_term
        ans = torch.sqrt(torch.maximum(dE2, torch.zeros_like(dE2)) + self.eps)
        ans = ans.mean()
        return ans