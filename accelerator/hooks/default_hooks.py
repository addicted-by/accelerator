"""
Default hook functions for common use cases.

This module provides pre-implemented hook functions that can be used
for common scenarios like data collection and analysis.
"""

from typing import Tuple, Union, Optional
import torch
from torch import nn

from .data_collector import DataCollector


def default_forward_hook(
    data_collector: DataCollector,
    module_name: str
) -> callable:
    """
    Create a default forward hook function that stores activations.
    
    Args:
        data_collector: DataCollector instance to store the activation data
        module_name: Name of the module for data storage
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """
        Default forward hook implementation.
        
        Args:
            module: The module that produced the activation
            input: Input tensors to the module (tuple)
            output: Output tensor from the module
        """
        # Store the output activation
        # Handle both single tensor and tuple outputs
        if isinstance(output, tuple):
            # For modules that return multiple outputs, store the first one
            activation_data = output[0]
        else:
            activation_data = output
            
        data_collector.store_activation(module_name, activation_data)
    
    return hook_fn


def default_backward_hook(
    data_collector: DataCollector,
    module_name: str
) -> callable:
    """
    Create a default backward hook function that stores gradients.
    
    Args:
        data_collector: DataCollector instance to store the gradient data
        module_name: Name of the module for data storage
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def hook_fn(
        module: nn.Module, 
        grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor], 
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor]
    ) -> None:
        """
        Default backward hook implementation.
        
        Args:
            module: The module that produced the gradient
            grad_input: Gradient with respect to the input
            grad_output: Gradient with respect to the output
        """
        # Store the output gradient (more commonly used)
        # Handle both single tensor and tuple gradients
        if isinstance(grad_output, tuple):
            # For modules with multiple outputs, store the first gradient
            if grad_output[0] is not None:
                gradient_data = grad_output[0]
            else:
                # Find the first non-None gradient
                gradient_data = next((g for g in grad_output if g is not None), None)
                if gradient_data is None:
                    return  # No gradients to store
        else:
            gradient_data = grad_output
            
        if gradient_data is not None:
            data_collector.store_gradient(module_name, gradient_data)
    
    return hook_fn


def default_full_backward_hook(
    data_collector: DataCollector,
    module_name: str,
    store_input_grad: bool = True,
    store_output_grad: bool = True
) -> callable:
    """
    Create a default full backward hook function that stores both input and output gradients.
    
    Args:
        data_collector: DataCollector instance to store the gradient data
        module_name: Name of the module for data storage
        store_input_grad: Whether to store input gradients
        store_output_grad: Whether to store output gradients
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def hook_fn(
        module: nn.Module,
        grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor]
    ) -> None:
        """
        Default full backward hook implementation.
        
        Args:
            module: The module that produced the gradients
            grad_input: Gradient with respect to the input
            grad_output: Gradient with respect to the output
        """
        # Store output gradient if requested
        if store_output_grad:
            if isinstance(grad_output, tuple):
                if grad_output[0] is not None:
                    gradient_data = grad_output[0]
                else:
                    gradient_data = next((g for g in grad_output if g is not None), None)
                    if gradient_data is None:
                        gradient_data = None
            else:
                gradient_data = grad_output
                
            if gradient_data is not None:
                data_collector.store_gradient(f"{module_name}_output_grad", gradient_data)
        
        # Store input gradient if requested
        if store_input_grad:
            if isinstance(grad_input, tuple):
                if grad_input[0] is not None:
                    input_gradient_data = grad_input[0]
                else:
                    input_gradient_data = next((g for g in grad_input if g is not None), None)
                    if input_gradient_data is None:
                        input_gradient_data = None
            else:
                input_gradient_data = grad_input
                
            if input_gradient_data is not None:
                data_collector.store_gradient(f"{module_name}_input_grad", input_gradient_data)
    
    return hook_fn


def create_activation_capture_hook(
    data_collector: DataCollector,
    module_name: str,
    capture_input: bool = False,
    capture_output: bool = True
) -> callable:
    """
    Create a forward hook that can capture both inputs and outputs.
    
    Args:
        data_collector: DataCollector instance to store the data
        module_name: Name of the module for data storage
        capture_input: Whether to capture input activations
        capture_output: Whether to capture output activations
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """
        Activation capture hook implementation.
        
        Args:
            module: The module that produced the activation
            input: Input tensors to the module (tuple)
            output: Output tensor from the module
        """
        # Capture input if requested
        if capture_input:
            if isinstance(input, tuple) and len(input) > 0:
                input_data = input[0]  # Take the first input tensor
                data_collector.store_activation(f"{module_name}_input", input_data)
        
        # Capture output if requested
        if capture_output:
            if isinstance(output, tuple):
                output_data = output[0]  # Take the first output tensor
            else:
                output_data = output
            data_collector.store_activation(f"{module_name}_output", output_data)
    
    return hook_fn


def create_gradient_capture_hook(
    data_collector: DataCollector,
    module_name: str,
    capture_input_grad: bool = True,
    capture_output_grad: bool = True
) -> callable:
    """
    Create a backward hook that can capture both input and output gradients.
    
    Args:
        data_collector: DataCollector instance to store the gradient data
        module_name: Name of the module for data storage
        capture_input_grad: Whether to capture input gradients
        capture_output_grad: Whether to capture output gradients
        
    Returns:
        Hook function that can be registered with PyTorch modules
    """
    def hook_fn(
        module: nn.Module,
        grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor]
    ) -> None:
        """
        Gradient capture hook implementation.
        
        Args:
            module: The module that produced the gradients
            grad_input: Gradient with respect to the input
            grad_output: Gradient with respect to the output
        """
        # Capture output gradient if requested
        if capture_output_grad:
            if isinstance(grad_output, tuple):
                grad_data = next((g for g in grad_output if g is not None), None)
            else:
                grad_data = grad_output
                
            if grad_data is not None:
                data_collector.store_gradient(f"{module_name}_output_grad", grad_data)
        
        # Capture input gradient if requested
        if capture_input_grad:
            if isinstance(grad_input, tuple):
                input_grad_data = next((g for g in grad_input if g is not None), None)
            else:
                input_grad_data = grad_input
                
            if input_grad_data is not None:
                data_collector.store_gradient(f"{module_name}_input_grad", input_grad_data)
    
    return hook_fn