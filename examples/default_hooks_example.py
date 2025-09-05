"""
Example demonstrating how to use default hook functions.

This example shows how to use the default hook functions provided by the
accelerator hooks system to capture activations and gradients from a neural network.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from accelerator.hooks import (
    DataCollector,
    HooksConfig,
    default_forward_hook,
    default_backward_hook,
    default_full_backward_hook,
    create_activation_capture_hook,
    create_gradient_capture_hook,
)


def main():
    print("Default Hooks Example")
    print("=" * 50)

    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 5), nn.ReLU(), nn.Linear(5, 1)
    )

    print(f"Model architecture:")
    print(model)
    print()

    # Create data collector with replace mode
    config = HooksConfig(data_mode="replace", device_placement="cpu")
    data_collector = DataCollector(config)

    # Example 1: Basic forward hook
    print("1. Basic Forward Hook Example")
    print("-" * 30)

    # Register forward hook on the first linear layer
    forward_hook = default_forward_hook(data_collector, "layer_0")
    handle1 = model[0].register_forward_hook(forward_hook)

    # Run forward pass
    x = torch.randn(3, 10)
    output = model(x)

    # Check captured activation
    activation = data_collector.get_activation("layer_0")
    print(f"Input shape: {x.shape}")
    print(f"Captured activation shape: {activation.shape}")
    print(f"Activation mean: {activation.mean().item():.4f}")
    print()

    # Clean up
    handle1.remove()
    data_collector.clear_all_data()

    # Example 2: Basic backward hook
    print("2. Basic Backward Hook Example")
    print("-" * 30)

    # Register backward hook on the first linear layer
    backward_hook = default_backward_hook(data_collector, "layer_0")
    handle2 = model[0].register_full_backward_hook(
        backward_hook
    )  # Use full backward hook

    # Run forward and backward pass
    x = torch.randn(3, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check captured gradient
    gradient = data_collector.get_gradient("layer_0")
    print(f"Captured gradient shape: {gradient.shape}")
    print(f"Gradient mean: {gradient.mean().item():.4f}")
    print()

    # Clean up
    handle2.remove()
    data_collector.clear_all_data()

    # Example 3: Full backward hook with both input and output gradients
    print("3. Full Backward Hook Example")
    print("-" * 30)

    # Register full backward hook
    full_backward_hook = default_full_backward_hook(
        data_collector, "layer_0", store_input_grad=True, store_output_grad=True
    )
    handle3 = model[0].register_full_backward_hook(full_backward_hook)

    # Run forward and backward pass
    x = torch.randn(3, 10, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check captured gradients
    output_grad = data_collector.get_gradient("layer_0_output_grad")
    input_grad = data_collector.get_gradient("layer_0_input_grad")
    print(f"Output gradient shape: {output_grad.shape}")
    print(f"Input gradient shape: {input_grad.shape}")
    print(f"Output gradient mean: {output_grad.mean().item():.4f}")
    print(f"Input gradient mean: {input_grad.mean().item():.4f}")
    print()

    # Clean up
    handle3.remove()
    data_collector.clear_all_data()

    # Example 4: Advanced activation capture
    print("4. Advanced Activation Capture Example")
    print("-" * 30)

    # Register activation capture hook that captures both input and output
    activation_hook = create_activation_capture_hook(
        data_collector, "layer_0", capture_input=True, capture_output=True
    )
    handle4 = model[0].register_forward_hook(activation_hook)

    # Run forward pass
    x = torch.randn(3, 10)
    output = model(x)

    # Check captured activations
    input_activation = data_collector.get_activation("layer_0_input")
    output_activation = data_collector.get_activation("layer_0_output")
    print(f"Input activation shape: {input_activation.shape}")
    print(f"Output activation shape: {output_activation.shape}")
    print(f"Input activation mean: {input_activation.mean().item():.4f}")
    print(f"Output activation mean: {output_activation.mean().item():.4f}")
    print()

    # Clean up
    handle4.remove()
    data_collector.clear_all_data()

    # Example 5: Multiple hooks on different layers
    print("5. Multiple Hooks Example")
    print("-" * 30)

    # Register hooks on multiple layers
    hooks = []
    layer_names = ["linear1", "linear2", "linear3"]
    linear_layers = [model[0], model[2], model[4]]  # Skip ReLU layers

    for name, layer in zip(layer_names, linear_layers):
        hook = default_forward_hook(data_collector, name)
        handle = layer.register_forward_hook(hook)
        hooks.append(handle)

    # Run forward pass
    x = torch.randn(2, 10)
    output = model(x)

    # Check all captured activations
    for name in layer_names:
        activation = data_collector.get_activation(name)
        print(f"{name} activation shape: {activation.shape}")

    # Get data summary
    summary = data_collector.get_data_summary()
    print(f"\nData summary:")
    print(
        f"Total modules with activations: {summary['total_modules_with_activations']}"
    )
    print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")

    # Clean up
    for handle in hooks:
        handle.remove()

    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
