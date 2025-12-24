import csv
import functools
import pathlib
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parametrize


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            f"Call to deprecated function {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def reapply_parametrizations(model, parametrized_modules, unsafe=True):
    """Function to reapply parameterizations to a model."""

    for name, params in parametrized_modules.items():
        module = dict(model.named_modules())[name]

        for p_name, parametrizations, orig_parameter in params:
            for parametrization in parametrizations:
                parametrize.register_parametrization(module, p_name, parametrization, unsafe=unsafe)
            getattr(module.parametrizations, p_name).original.data = orig_parameter


def save_report(pruning_record, experiment_path, ratio):
    """Saves a pruning report as CSV files, both in raw and
    human-readable formats, in a specified experiment path folder."""
    if experiment_path:
        experiment_path = pathlib.Path(experiment_path)
        pruning_reports_path = experiment_path / "pruning_records"
        pruning_reports_path.mkdir(parents=True, exist_ok=True)

        csv_file = pruning_reports_path / "pruning_record.csv"
        with open(csv_file, "w") as test_file:
            file_writer = csv.writer(test_file)

            for i in range(len(pruning_record[0])):
                file_writer.writerow([x[i] for x in pruning_record])
        pd.read_csv(csv_file, header=None).T.to_csv(
            pruning_reports_path / f"pruning_record_human_readable_{ratio:.4f}.csv"
        )


def remove_parametrizations(model):
    """Function to remove parameterizations from a model."""

    parametrized_modules = {}

    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            parametrized_modules[name] = []

            for p_name in list(module.parametrizations.keys()):
                orig_parameter = getattr(module.parametrizations, p_name)
                orig_parameter = orig_parameter.original.data.detach().clone()
                parametrized_modules[name].append((p_name, module.parametrizations[p_name], orig_parameter))
                parametrize.remove_parametrizations(module, p_name, True)

    return parametrized_modules


def get_attr_by_name(module, name):
    """ """
    for s in name.split("."):
        module = getattr(module, s)

    return module


def get_parent_name(qualname: str) -> tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = qualname.rsplit(".", 1)
    return parent[0] if parent else "", name


def get_parent_module(module, attr_path):
    """
    Returns parent module of module.attr_path.

    Parameters
    ----------
    module: torch.nn.Module.
    attr_path: str.
    """
    parent_name, _ = get_parent_name(attr_path)

    if parent_name != "":
        parent = get_attr_by_name(module, parent_name)
    else:
        parent = module

    return parent


def remove_all_hooks(model: torch.nn.Module) -> None:
    """ """
    for _, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_hooks(child)


def fuse_batchnorm(model, fx_model=None, convs=None):
    """
    Fuse conv and bn only if conv is in convs argument.

    Parameters
    ----------
    model: torch.nn.Module.
    fx_model: torch.fx.GraphModule.
    convs: List[torch.nn.ConvNd].
    """
    if fx_model is None:
        fx_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != "call_module":
            continue
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            to_fuse = True if convs is None else node.args[0].target in convs
            if to_fuse:
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                _inplace_conv_bn_fusion(conv, bn)
                parent_name, attr_name = get_parent_name(node.target)
                parent = get_parent_module(model, node.target)
                setattr(parent, attr_name, torch.nn.Identity())


def _inplace_conv_bn_fusion(conv, bn):
    """ """
    assert not (conv.training or bn.training), "Fusion only for eval!"
    conv.weight.data, bias = _fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    if conv.bias is None:
        conv.bias = torch.nn.Parameter(bias).to(conv.weight.device)
    else:
        conv.bias.data = bias


def _fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """ """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def to_device(x: Union[torch.Tensor, Iterable], device: torch.device) -> Union[torch.Tensor, Iterable]:
    """Recursively moves tensors in data structures (list, tuples, single tensors)
    to a specified device, making it handly for device-agnostic execution
    of model operations."""

    def _to_device(x, device):
        if isinstance(x, (list, tuple)):
            for v in x:
                _to_device(v, device)
        elif isinstance(x, torch.Tensor):
            x.data = x.data.to(device)

    _to_device(x, device)
    return x


def run_model(model, inputs, req_grad=False, device=None):
    if device is None:
        device = next(iter(model.parameters())).device
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    to_device(inputs, device)
    with torch.set_grad_enabled(mode=req_grad):
        outputs = model(*inputs)

    return outputs


def get_model(model):
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
