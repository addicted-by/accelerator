"""Tests for base infrastructure re-exports."""

import pytest


def test_base_imports():
    """Test that AccelerationOperationBase can be imported from algorithm.base."""
    from accelerator.algorithm.base import AccelerationOperationBase

    assert AccelerationOperationBase is not None
    assert hasattr(AccelerationOperationBase, "apply")
    assert hasattr(AccelerationOperationBase, "reapply")
    assert hasattr(AccelerationOperationBase, "calibrate")


def test_registry_imports():
    """Test that registry components can be imported from algorithm.registry."""
    from accelerator.algorithm.registry import (
        AccelerationRegistry,
        AccelerationType,
        acceleration_registry,
        registry,
    )

    assert AccelerationRegistry is not None
    assert AccelerationType is not None
    assert acceleration_registry is not None
    assert registry is not None

    # Verify they are the same instance
    assert acceleration_registry is registry


def test_acceleration_types():
    """Test that AccelerationType enum has expected values."""
    from accelerator.algorithm.registry import AccelerationType

    assert hasattr(AccelerationType, "PRUNING")
    assert hasattr(AccelerationType, "QUANTIZATION")
    assert hasattr(AccelerationType, "OPTIMIZATION")
    assert hasattr(AccelerationType, "REPARAMETRIZATION")
    assert hasattr(AccelerationType, "CUSTOM")

    assert AccelerationType.PRUNING.value == "pruning"
    assert AccelerationType.QUANTIZATION.value == "quantization"


def test_main_init_exports():
    """Test that components are exported from main __init__.py."""
    from accelerator.algorithm import (
        AccelerationOperationBase,
        AccelerationRegistry,
        AccelerationType,
        acceleration_registry,
    )

    assert AccelerationOperationBase is not None
    assert acceleration_registry is not None
    assert AccelerationType is not None
    assert AccelerationRegistry is not None


def test_registry_functionality():
    """Test basic registry functionality."""
    from accelerator.algorithm.registry import acceleration_registry

    # Test that registry has expected methods
    assert hasattr(acceleration_registry, "register_acceleration")
    assert hasattr(acceleration_registry, "get_acceleration")
    assert hasattr(acceleration_registry, "list_accelerations")
    assert hasattr(acceleration_registry, "has_acceleration")

    # Test listing accelerations
    accelerations = acceleration_registry.list_accelerations()
    assert isinstance(accelerations, dict)
    assert "pruning" in accelerations
    assert "quantization" in accelerations


def test_base_class_structure():
    """Test that AccelerationOperationBase has expected structure."""
    import inspect

    from accelerator.algorithm.base import AccelerationOperationBase

    # Check that it's an abstract base class
    assert inspect.isabstract(AccelerationOperationBase)

    # Check abstract methods
    abstract_methods = {
        name
        for name, method in inspect.getmembers(AccelerationOperationBase)
        if getattr(method, "__isabstractmethod__", False)
    }

    assert "apply" in abstract_methods
    assert "reapply" in abstract_methods
    assert "calibrate" in abstract_methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
