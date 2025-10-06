import torch


def smooth_loss_fn(smooth_loss_type, beta: float=1):
    smooth_loss_type = smooth_loss_type
    if smooth_loss_type == 'l1':
        return torch.nn.L1Loss()
    elif smooth_loss_type == 'l2':
        return torch.nn.MSELoss()
    elif smooth_loss_type == 'smooth':
        return torch.nn.SmoothL1Loss(beta=beta)
    else:
        raise NotImplementedError(f"Loss type {smooth_loss_type} does not implemented!")