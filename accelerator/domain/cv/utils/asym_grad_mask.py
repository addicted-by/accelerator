import torch


def calc_asym_gradient_mask(gxy0, gxy1):
    flag = gxy0 * gxy1
    mask = torch.zeros_like(gxy0)
    mask = torch.where(flag > 0.0, 0.2, 4.0)
    return mask