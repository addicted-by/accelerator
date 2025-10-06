"""Test imports for accelerator-algorithm package."""

import pytest


def test_base_imports():
    """Test that base acceleration classes can be imported."""
    from accelerator.algorithm.base import AccelerationOperationBase
    from accelerator.algorithm.registry import AccelerationRegistry, AccelerationType, registry
    
    assert AccelerationOperationBase is not None
    assert AccelerationRegistry is not None
    assert AccelerationType is not None
    assert registry is not None


def test_pruning_imports():
    """Test that pruning operations can be imported."""
    from accelerator.algorithm.pruning_op import QaPUTPruning
    from accelerator.algorithm.pruning import get_pruned_dict, load_pruned
    
    assert QaPUTPruning is not None
    assert get_pruned_dict is not None
    assert load_pruned is not None


def test_optimization_imports():
    """Test that optimization operations can be imported."""
    from accelerator.algorithm.optimization import SmoothlyRemoveLayer
    
    assert SmoothlyRemoveLayer is not None


def test_main_package_imports():
    """Test that main package exports work correctly."""
    from accelerator.algorithm import (
        AccelerationOperationBase,
        acceleration_registry,
        AccelerationType,
        AccelerationRegistry,
        QaPUTPruning,
        SmoothlyRemoveLayer,
    )
    
    assert AccelerationOperationBase is not None
    assert acceleration_registry is not None
    assert AccelerationType is not None
    assert AccelerationRegistry is not None
    assert QaPUTPruning is not None
    assert SmoothlyRemoveLayer is not None


def test_registry_functionality():
    """Test that the acceleration registry works correctly."""
    from accelerator.algorithm.registry import AccelerationType, registry
    
    # Test that registry has the expected acceleration types
    assert AccelerationType.PRUNING in AccelerationType
    assert AccelerationType.OPTIMIZATION in AccelerationType
    assert AccelerationType.QUANTIZATION in AccelerationType
    
    # Test that registry is properly initialized
    assert hasattr(registry, 'register_acceleration')
    assert hasattr(registry, 'get_acceleration')
    assert hasattr(registry, 'list_accelerations')


if __name__ == "__main__":
    pytest.main([__file__])