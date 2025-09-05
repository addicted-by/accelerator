import torch

def custom_collate_fn(batch):
    batch = torch.utils.data._utils.collate.default_collate(batch)
    input_, gt = batch[0], batch[1]
    return (input_, ), gt