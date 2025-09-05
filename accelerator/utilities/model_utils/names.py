import torch


def get_parent_name(model: torch.nn.Module, module_name: str):
    splitted = module_name.rsplit('.', 1)
    if len(splitted) == 1:
        parent_name = None
        attribute_name = parent_name
    else:
        parent_name, attribute_name = splitted
    return parent_name, attribute_name

def get_parent_module(model: torch.nn.Module, module_name: str):
    parent_name, attribute_name = get_parent_name(model, module_name)
    if parent_name is None:
        return model, attribute_name
    return model.get_submodule(parent_name), attribute_name