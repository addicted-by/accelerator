from collections import defaultdict

import torch

from .utils import deprecated

_attributes_to_migrate = [
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "n_head",
]


def get_pruned_dict(model):
    pruned_dict = defaultdict(dict)

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            pruned_dict["_attributes"][name]["weight"] = module.weight.shape

        if hasattr(module, "bias"):
            pruned_dict["_attributes"][name]["bias"] = module.bias.shape

        for attr in _attributes_to_migrate:
            if hasattr(module, attr):
                pruned_dict["_attributes"][name][attr] = getattr(module, attr)

    if hasattr(model, "continuous_dims"):
        integral_dict = defaultdict(dict)
        integral_dict["continuous_dims"] = model.continuous_dims
        integral_dict["discrete_dims"] = getattr(model, "discrete_dims", None)
        pruned_dict['integral_dict'] = integral_dict

    if hasattr(model, "pruning_dims"):
        merging_dict = {
            "pruning_dims": model.pruning_dims,
            "ignored_dims": model.ignored_dims,
            "merging_dict": model.pruning_dict,
            "parametrized_modules": model.parametrized_modules,
        }
        pruned_dict["merging_dict"] = merging_dict
        

    return pruned_dict


def load_pruned(model, pruned_dict, example_input=None):
    def is_same(value_before, value_after):
        if (value_before is None) ^ (value_after is None):
            return False

        if isinstance(value_before, torch.Tensor):
            return (
                torch.allclose(value_before, value_after)
                if value_before.size() == value_after.size()
                else False
            )
        if isinstance(value_before, dict):
            if set(value_before.keys()).difference(set(value_after.keys())):
                return False
            return all(
                [is_same(value_before[k], value_after[k]) for k in value_before.keys()]
            )

        return value_before == value_after

    if pruned_dict.get(
        "continuous_dims", None
    ):  # if the pruned checkpoint is integral one
        assert (
            example_input
        ), "To load the integral model you need to specify example input"
        from torch_integral import IntegralWrapper

        wrapper = IntegralWrapper(init_from_discrete=True)
        model = wrapper(
            model=model,
            example_input=example_input,
            continuous_dims=model["continuous_dims"],
            discrete_dims=pruned_dict.get("discrete_dims", None),
            related_groups=pruned_dict.get("related_groups", None),
        )
    
    if pruned_dict.get(
        "pruning_dims", None
    ): # if the pruned checkpoint is after Linear Merging
        assert (
            example_input
        ), "To load the merged model you need to specify example input"
        ### ! TODO: FILL IT []


    for module_name, values in pruned_dict.items():
        print(f"Setting {module_name}")
        module = model.get_submodule(module_name)
        weight_shape = values.pop("weight", None)
        same = True
        if weight_shape:
            is_same_ = is_same(weight_shape, module.weight.shape)
            if is_same_:
                message = f"\tWeights tensors have same shapes {weight_shape, module.weight.shape}. Skip..."
            else:
                message = (
                    f"\tSetting weights shape: {module.weight.shape} -> {weight_shape}"
                )
                module.weight.data = torch.rand(
                    weight_shape
                )  # module.weight.data.resize(weight_shape)
            print(message)
            same &= is_same_

        bias_shape = values.pop("bias", None)
        if bias_shape:
            is_same_ = is_same(bias_shape, module.bias.shape)
            if is_same_:
                message = "\tBiases tensors have same shapes. Skip..."
            else:
                message = f"\tSetting bias shape: {module.bias.shape} -> {bias_shape}"
                module.bias.data = torch.rand(
                    bias_shape
                )  # module.bias.data.resize(bias_shape)

            print(message)
            same &= is_same_

        if same:
            print(
                "\tWeights and biases shapes are the same. So no need to change other attributes!"
            )
        else:
            print(
                "\tWeights or biases shapes were changed. Need to update other attributes..."
            )
            for attr, value in values.items():
                module_attribute = getattr(module, attr)
                if is_same(module_attribute, value):
                    message = f"\t\t{attr} preserved same: {value}"
                else:
                    message = (
                        f"\t\t{attr}: {module_attribute} [Before]  -> {value} [After]"
                    )
                    setattr(module, attr, value)
                print(message)

    


@deprecated
def load_pruned_(model, pruned_state_dict):
    def is_same(value_before, value_after):
        if (value_before is None) ^ (value_after is None):
            return False

        if isinstance(value_before, torch.Tensor):
            return (
                torch.allclose(value_before, value_after)
                if value_before.size() == value_after.size()
                else False
            )
        if isinstance(value_before, dict):
            if set(value_before.keys()).difference(set(value_after.keys())):
                return False
            return all(
                [is_same(value_before[k], value_after[k]) for k in value_before.keys()]
            )

        return value_before == value_after

    for name, values in pruned_state_dict.items():
        print(f"Setting: {name}")
        for attr, value in values.items():
            value_before = getattr(model.get_submodule(name), attr)
            if "hooks" in attr:
                if len(value):
                    print(f"\t\tLoading {attr}: {value}")
                    setattr(model.get_submodule(name), attr, value)
                continue

            if is_same(value_before, value):
                continue
            else:
                print(f"\tSetting {attr}: ")

                if isinstance(value, dict):
                    for k in value_before:
                        print(
                            f"\t\tLoading {k}: {value_before[k].shape} -> {value[k].shape}"
                        )

                else:
                    print(f"\t\tLoading: {value_before} -> {value}")
                setattr(model.get_submodule(name), attr, value)


@deprecated
def get_pruned_dict_(model):
    save_dict = {}
    for name, module in model.named_modules():
        if hasattr(torch.nn.modules, module.__class__.__name__) and hasattr(
            module, "weight"
        ):
            save_dict[name] = vars(module)

    return save_dict
