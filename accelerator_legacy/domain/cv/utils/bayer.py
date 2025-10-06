import torch

def rggb_to_bayer(frame):
    bs, ch, height, width = frame.shape
    device = frame.device
    bayer = torch.zeros((bs, 1, height * 2, width * 2), device=device)

    bayer[:, :, 0::2, 0::2] = frame[:, 0, ...].unsqueeze(1)
    bayer[:, :, 0::2, 1::2] = frame[:, 1, ...].unsqueeze(1)
    bayer[:, :, 1::2, 0::2] = frame[:, 2, ...].unsqueeze(1)
    bayer[:, :, 1::2, 1::2] = frame[:, 3, ...].unsqueeze(1)
    return bayer


def rgb_to_bayer(batch: torch.Tensor, per_pixel: bool=False):
    b, _, h, w = batch.shape
    device = batch.device
    if per_pixel:
        bayer = torch.zeros((b, 2*h, 2*w), device=device)
        bayer[..., 0::2, 0::2] = batch[:, 0, ...]
        bayer[..., 1::2, 0::2] = batch[:, 1, ...]
        bayer[..., 0::2, 1::2] = batch[:, 1, ...]
        bayer[..., 1::2, 1::2] = batch[:, 2, ...]
    else:
        bayer = torch.zeros((b, h, w), device=device)

        bayer[..., 0::2, 0::2] = batch[:, 0, 0::2, 0::2]
        bayer[..., 1::2, 0::2] = batch[:, 1, 1::2, 0::2]
        bayer[..., 0::2, 1::2] = batch[:, 1, 0::2, 1::2]
        bayer[..., 1::2, 1::2] = batch[:, 2, 1::2, 1::2]

    return bayer.unsqueeze(1)