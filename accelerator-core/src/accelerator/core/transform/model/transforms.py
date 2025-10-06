"""Built-in model transforms for the accelerator framework.

This module provides common model transformations that can be applied
before or after acceleration operations.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from collections import OrderedDict


def set_eval_mode(model: nn.Module) -> None:
    """Set model to evaluation mode.
    
    This transform disables dropout and batch normalization training behavior,
    making the model suitable for inference or pruning operations.
    
    Parameters
    ----------
    model : nn.Module
        The model to set to evaluation mode.
        
    Examples
    --------
    >>> model = MyModel()
    >>> set_eval_mode(model)
    >>> assert not model.training
    """
    model.eval()


def set_train_mode(model: nn.Module) -> None:
    """Set model to training mode.
    
    This transform enables dropout and batch normalization training behavior.
    
    Parameters
    ----------
    model : nn.Module
        The model to set to training mode.
        
    Examples
    --------
    >>> model = MyModel()
    >>> set_train_mode(model)
    >>> assert model.training
    """
    model.train()


def unfreeze_parameters(model: nn.Module) -> None:
    """Unfreeze all model parameters.
    
    This transform enables gradient computation for all parameters in the model,
    making them trainable.
    
    Parameters
    ----------
    model : nn.Module
        The model whose parameters should be unfrozen.
        
    Examples
    --------
    >>> model = MyModel()
    >>> # Freeze all parameters
    >>> for param in model.parameters():
    ...     param.requires_grad = False
    >>> # Unfreeze them
    >>> unfreeze_parameters(model)
    >>> assert all(p.requires_grad for p in model.parameters())
    """
    for param in model.parameters():
        param.requires_grad = True


def fuse_batch_norm(
    model: nn.Module,
    fx_model: Optional[torch.fx.GraphModule] = None,
    convs: Optional[List[str]] = None
) -> None:
    """Fuse batch normalization layers into preceding convolutional layers.
    
    This transform combines Conv2d and BatchNorm2d layers into a single Conv2d
    layer by folding the batch normalization parameters into the convolution
    weights and biases. This is only valid when the model is in evaluation mode.
    
    The fusion reduces computational overhead and simplifies the model structure,
    which is particularly useful before pruning operations.
    
    Parameters
    ----------
    model : nn.Module
        The model containing layers to fuse. Must be in evaluation mode.
    fx_model : Optional[torch.fx.GraphModule], default=None
        Pre-traced FX graph module. If None, the model will be traced automatically.
    convs : Optional[List[str]], default=None
        List of convolutional layer names to fuse. If None, all eligible
        conv-bn pairs will be fused.
        
    Raises
    ------
    AssertionError
        If the model or batch norm layers are in training mode.
        
    Notes
    -----
    - Only fuses Conv2d -> BatchNorm2d sequences
    - Requires the model to be in evaluation mode
    - Replaces fused BatchNorm2d layers with Identity layers
    - Only fuses when the conv layer has a single user
    
    Examples
    --------
    >>> model = MyModel()
    >>> model.eval()  # Must be in eval mode
    >>> fuse_batch_norm(model)
    """
    if fx_model is None:
        fx_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        if node.op != "call_module":
            continue
        if (
            type(modules[node.target]) is nn.BatchNorm2d
            and type(modules[node.args[0].target]) is nn.Conv2d
        ):
            to_fuse = True if convs is None else node.args[0].target in convs
            if to_fuse:
                if len(node.args[0].users) > 1:
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                _inplace_conv_bn_fusion(conv, bn)
                parent_name, attr_name = _get_parent_name(node.target)
                parent = _get_parent_module(model, node.target)
                setattr(parent, attr_name, nn.Identity())


def _inplace_conv_bn_fusion(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fuse conv and batch norm layers in-place.
    
    Parameters
    ----------
    conv : nn.Conv2d
        Convolutional layer to fuse into.
    bn : nn.BatchNorm2d
        Batch normalization layer to fuse.
    """
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
        conv.bias = nn.Parameter(bias).to(conv.weight.device)
    else:
        conv.bias.data = bias


def _fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: Optional[torch.Tensor],
    bn_b: Optional[torch.Tensor]
) -> tuple:
    """Compute fused weights and biases for conv-bn fusion.
    
    Parameters
    ----------
    conv_w : torch.Tensor
        Convolution weights.
    conv_b : Optional[torch.Tensor]
        Convolution bias.
    bn_rm : torch.Tensor
        Batch norm running mean.
    bn_rv : torch.Tensor
        Batch norm running variance.
    bn_eps : float
        Batch norm epsilon.
    bn_w : Optional[torch.Tensor]
        Batch norm weight (gamma).
    bn_b : Optional[torch.Tensor]
        Batch norm bias (beta).
        
    Returns
    -------
    tuple
        Fused weights and bias tensors.
    """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def _get_parent_name(qualname: str) -> tuple:
    """Split a qualified name into parent path and last atom.
    
    Parameters
    ----------
    qualname : str
        Qualified name (e.g., 'foo.bar.baz').
        
    Returns
    -------
    tuple
        Parent path and name (e.g., ('foo.bar', 'baz')).
    """
    *parent, name = qualname.rsplit(".", 1)
    return parent[0] if parent else "", name


def _get_parent_module(module: nn.Module, attr_path: str) -> nn.Module:
    """Get the parent module of a nested attribute.
    
    Parameters
    ----------
    module : nn.Module
        Root module.
    attr_path : str
        Attribute path (e.g., 'layer1.conv').
        
    Returns
    -------
    nn.Module
        Parent module containing the attribute.
    """
    parent_name, _ = _get_parent_name(attr_path)

    if parent_name != "":
        parent = _get_attr_by_name(module, parent_name)
    else:
        parent = module

    return parent


def _get_attr_by_name(module: nn.Module, name: str) -> nn.Module:
    """Get a nested attribute by name.
    
    Parameters
    ----------
    module : nn.Module
        Root module.
    name : str
        Attribute name (e.g., 'layer1.conv').
        
    Returns
    -------
    nn.Module
        The nested module.
    """
    for s in name.split("."):
        module = getattr(module, s)
    return module
