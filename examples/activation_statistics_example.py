"""
Example demonstrating how to use activation statistics hooks for model analysis.

This example shows:
1. Basic statistics collection
2. Quantization calibration data collection
3. Training monitoring with statistics
"""

import json
import os

# Import the statistics hook (adjust path as needed)
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from accelerator.hooks.statistics import ActivationStatisticsHook, StatisticsCollector, StatisticsConfig


class SimpleModel(nn.Module):
    """Simple CNN model for demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_dummy_data(batch_size=32, num_batches=10):
    """Create dummy data for demonstration."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        data.append((x, y))
    return data


def example_basic_statistics():
    """Example 1: Basic activation statistics collection."""
    print("=== Example 1: Basic Statistics Collection ===")

    # Create model and data
    model = SimpleModel()
    data = create_dummy_data(batch_size=16, num_batches=5)

    # Configure statistics collection
    config = StatisticsConfig(
        collect_mean=True, collect_std=True, collect_min_max=True, collect_percentiles=[95, 99], collect_sparsity=True
    )

    # Create statistics hook
    stats_hook = ActivationStatisticsHook(config)

    # Register hooks on specific layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Set module name for the hook
            module._hook_name = name
            hook_handle = module.register_forward_hook(stats_hook)
            hooks.append(hook_handle)
            print(f"Registered statistics hook on {name}")

    # Run inference to collect statistics
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data):
            output = model(x)
            print(f"Processed batch {batch_idx + 1}")

    # Get and display statistics
    all_stats = stats_hook.get_all_statistics()
    for module_name, stats in all_stats.items():
        print(f"\n--- Statistics for {module_name} ---")
        print(f"Total samples: {stats.total_samples}")
        print(f"Output shape: {stats.output_stats.shape}")
        print(f"Output mean: {stats.output_stats.mean:.4f}")
        print(f"Output std: {stats.output_stats.std:.4f}")
        print(f"Output range: [{stats.output_stats.min:.4f}, {stats.output_stats.max:.4f}]")
        print(f"Output sparsity: {stats.output_stats.sparsity_ratio:.2%}")

        if stats.output_stats.percentiles:
            percentiles_str = ", ".join([f"{p}%: {v:.4f}" for p, v in stats.output_stats.percentiles.items()])
            print(f"Output percentiles: {percentiles_str}")

    # Export statistics
    stats_data = stats_hook.export_statistics(format="json")
    with open("basic_statistics.json", "w") as f:
        json.dump(stats_data, f, indent=2)
    print("\nStatistics exported to basic_statistics.json")

    # Clean up hooks
    for hook in hooks:
        hook.remove()


def example_quantization_calibration():
    """Example 2: Collecting statistics for quantization calibration."""
    print("\n=== Example 2: Quantization Calibration ===")

    # Create model and calibration data
    model = SimpleModel()
    calibration_data = create_dummy_data(batch_size=32, num_batches=20)

    # Configure for quantization calibration
    config = StatisticsConfig(
        collect_mean=False,  # Not needed for quantization
        collect_std=False,  # Not needed for quantization
        collect_min_max=True,  # Essential for quantization
        collect_percentiles=[99.9],  # For outlier handling
        max_samples=5000,  # Limit memory usage
        update_frequency=1,  # Collect from every forward pass
    )

    # Create statistics hook
    stats_hook = ActivationStatisticsHook(config)

    # Register hooks on all layers
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module._hook_name = name
            hook_handle = module.register_forward_hook(stats_hook)
            hooks.append(hook_handle)

    # Calibration loop
    model.eval()
    print("Running calibration...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(calibration_data):
            output = model(x)
            if (batch_idx + 1) % 5 == 0:
                print(f"Calibration batch {batch_idx + 1}/{len(calibration_data)}")

    # Export quantization data
    quant_data = stats_hook.export_statistics(format="quantization")
    with open("quantization_calibration.json", "w") as f:
        json.dump(quant_data, f, indent=2)

    print("Quantization calibration data:")
    for module_name, data in quant_data.items():
        if "min" in data and "max" in data:
            print(f"{module_name}: range=[{data['min']:.4f}, {data['max']:.4f}]")
            if "percentiles" in data and "99.9" in data["percentiles"]:
                print(f"  99.9th percentile: {data['percentiles']['99.9']:.4f}")

    # Clean up hooks
    for hook in hooks:
        hook.remove()


def example_training_monitoring():
    """Example 3: Monitoring activations during training."""
    print("\n=== Example 3: Training Monitoring ===")

    # Create model, data, and training setup
    model = SimpleModel()
    train_data = create_dummy_data(batch_size=16, num_batches=20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Configure for training monitoring
    config = StatisticsConfig(
        collect_mean=True,
        collect_std=True,
        collect_sparsity=True,
        collect_percentiles=[],  # Skip percentiles for performance
        update_frequency=5,  # Update every 5 forward passes
        max_samples=1000,
    )

    # Create statistics hook
    stats_hook = ActivationStatisticsHook(config)

    # Register hooks on key layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module._hook_name = name
            hook_handle = module.register_forward_hook(stats_hook)
            hooks.append(hook_handle)

    # Training loop with monitoring
    model.train()
    for epoch in range(2):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_data):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Monitor statistics every few batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

                # Check for potential issues
                all_stats = stats_hook.get_all_statistics()
                for module_name, stats in all_stats.items():
                    if stats.output_stats.sparsity_ratio and stats.output_stats.sparsity_ratio > 0.8:
                        print(f"  Warning: {module_name} has high sparsity ({stats.output_stats.sparsity_ratio:.2%})")

                    if stats.output_stats.std and stats.output_stats.std < 0.01:
                        print(
                            f"  Warning: {module_name} has very low activation variance ({stats.output_stats.std:.6f})"
                        )

        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

    # Export final training statistics
    training_stats = stats_hook.export_statistics(format="json")
    with open("training_statistics.json", "w") as f:
        json.dump(training_stats, f, indent=2)

    print("Training statistics exported to training_statistics.json")

    # Clean up hooks
    for hook in hooks:
        hook.remove()


def example_custom_analysis():
    """Example 4: Custom analysis using the statistics collector directly."""
    print("\n=== Example 4: Custom Analysis ===")

    # Create model and data
    model = SimpleModel()
    data = create_dummy_data(batch_size=8, num_batches=10)

    # Create custom statistics collector
    config = StatisticsConfig(
        collect_mean=True,
        collect_std=True,
        collect_min_max=True,
        collect_percentiles=[50, 90, 95, 99],
        collect_sparsity=True,
    )

    collector = StatisticsCollector(config)

    # Manual statistics collection
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data):
            # Forward pass
            conv1_out = model.relu(model.conv1(x))
            conv2_out = model.relu(model.conv2(conv1_out))
            pooled = model.pool(conv2_out)
            flattened = pooled.view(pooled.size(0), -1)
            fc1_out = model.relu(model.fc1(flattened))
            output = model.fc2(fc1_out)

            # Manually collect statistics for specific layers
            collector.update_statistics("conv1", x, conv1_out)
            collector.update_statistics("conv2", conv1_out, conv2_out)
            collector.update_statistics("fc1", flattened, fc1_out)
            collector.update_statistics("fc2", fc1_out, output)

    # Custom analysis
    print("Custom Analysis Results:")
    all_stats = collector.get_all_statistics()

    # Find layers with highest/lowest activation ranges
    ranges = {}
    for name, stats in all_stats.items():
        if stats.output_stats.min is not None and stats.output_stats.max is not None:
            range_val = stats.output_stats.max.item() - stats.output_stats.min.item()
            ranges[name] = range_val

    if ranges:
        max_range_layer = max(ranges, key=ranges.get)
        min_range_layer = min(ranges, key=ranges.get)

        print(f"Layer with highest activation range: {max_range_layer} ({ranges[max_range_layer]:.4f})")
        print(f"Layer with lowest activation range: {min_range_layer} ({ranges[min_range_layer]:.4f})")

    # Find layers with highest sparsity
    sparsities = {}
    for name, stats in all_stats.items():
        if stats.output_stats.sparsity_ratio is not None:
            sparsities[name] = stats.output_stats.sparsity_ratio

    if sparsities:
        most_sparse_layer = max(sparsities, key=sparsities.get)
        print(f"Most sparse layer: {most_sparse_layer} ({sparsities[most_sparse_layer]:.2%} zeros)")

    # Export custom analysis
    custom_analysis = {
        "activation_ranges": ranges,
        "sparsity_ratios": sparsities,
        "detailed_stats": collector.export_statistics(format="json"),
    }

    with open("custom_analysis.json", "w") as f:
        json.dump(custom_analysis, f, indent=2)

    print("Custom analysis exported to custom_analysis.json")


if __name__ == "__main__":
    print("Activation Statistics Hook Examples")
    print("=" * 50)

    # Run all examples
    example_basic_statistics()
    example_quantization_calibration()
    example_training_monitoring()
    example_custom_analysis()

    print("\n" + "=" * 50)
    print("All examples completed! Check the generated JSON files for detailed results.")
