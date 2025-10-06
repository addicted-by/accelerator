import torch

def rgb2yuv(rgb):
    rgb_ = rgb.permute(0, 2, 3, 1)                              # input is 3*n*n   default
    mat = torch.tensor([[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]])   # from  Wikipedia
    yuv = torch.tensordot(rgb_, mat.to(rgb.device), 1).permute(0, 3, 1, 2)
    return yuv