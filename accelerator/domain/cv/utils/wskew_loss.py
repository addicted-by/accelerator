import torch


def calc_wskew_loss(model):
    w_counter = 0
    w_skew_loss = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # print(name)
            weight = module.weight.view(-1)
            diffs = weight - torch.mean(weight)
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            weights_skew_loss = torch.abs(torch.mean(torch.pow(zscores, 3.0)))
            w_skew_loss = w_skew_loss + weights_skew_loss

            w_counter += 1

    if isinstance(w_skew_loss, torch.Tensor):
        w_skew_loss = w_skew_loss.item() / w_counter

    return w_skew_loss