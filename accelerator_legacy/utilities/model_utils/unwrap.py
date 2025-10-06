import torch


def unwrap_model(model):
    """
    Unwraps a model from `torch.nn.parallel.DistributedDataParallel`
    or compiled model to access its underlying model.
    """
    true_model = model
    if isinstance(true_model, torch.nn.parallel.DistributedDataParallel):
        true_model = true_model.module

    if hasattr(true_model, "_orig_mod"):
        true_model = true_model._orig_mod

    return true_model