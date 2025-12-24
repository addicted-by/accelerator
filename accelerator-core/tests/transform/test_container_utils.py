"""Unit tests for container access utilities."""

import pytest
import torch

from accelerator.core.context import Context
from accelerator.core.transform.container_utils import ContainerGetItem, ContainerSetItem


class TestContainerGetItem:
    """Test suite for ContainerGetItem class."""

    def test_init_with_args_only(self):
        """Test initialization with only positional arguments."""
        getter = ContainerGetItem(args=["per_batch.input.rgb", "per_batch.target.label"])
        assert getter.args_paths == ["per_batch.input.rgb", "per_batch.target.label"]
        assert getter.kwargs_paths == {}

    def test_init_with_kwargs_only(self):
        """Test initialization with only keyword arguments."""
        getter = ContainerGetItem(kwargs={"image": "per_batch.input.rgb", "label": "per_batch.target.label"})
        assert getter.args_paths == []
        assert getter.kwargs_paths == {"image": "per_batch.input.rgb", "label": "per_batch.target.label"}

    def test_init_with_mixed_args_kwargs(self):
        """Test initialization with both args and kwargs."""
        getter = ContainerGetItem(args=["per_batch.input.rgb"], kwargs={"pca_mat": "persistent.additional.pca_mat"})
        assert getter.args_paths == ["per_batch.input.rgb"]
        assert getter.kwargs_paths == {"pca_mat": "persistent.additional.pca_mat"}

    def test_init_with_no_args_raises_error(self):
        """Test that initialization without args or kwargs raises ValueError."""
        with pytest.raises(ValueError, match="At least one of 'args' or 'kwargs' must be provided"):
            ContainerGetItem()

    def test_init_with_empty_args_and_kwargs_raises_error(self):
        """Test that initialization with empty args and kwargs raises ValueError."""
        with pytest.raises(ValueError, match="At least one of 'args' or 'kwargs' must be provided"):
            ContainerGetItem(args=[], kwargs={})

    def test_extract_single_arg(self):
        """Test extracting a single positional argument."""
        context = Context()
        test_tensor = torch.randn(3, 224, 224)
        context.set_item("per_batch.input.rgb", test_tensor)

        getter = ContainerGetItem(args=["per_batch.input.rgb"])
        args, kwargs = getter(context)

        assert len(args) == 1
        assert torch.equal(args[0], test_tensor)
        assert kwargs == {}

    def test_extract_multiple_args(self):
        """Test extracting multiple positional arguments."""
        context = Context()
        rgb_tensor = torch.randn(3, 224, 224)
        depth_tensor = torch.randn(1, 224, 224)
        context.set_item("per_batch.input.rgb", rgb_tensor)
        context.set_item("per_batch.input.depth", depth_tensor)

        getter = ContainerGetItem(args=["per_batch.input.rgb", "per_batch.input.depth"])
        args, kwargs = getter(context)

        assert len(args) == 2
        assert torch.equal(args[0], rgb_tensor)
        assert torch.equal(args[1], depth_tensor)
        assert kwargs == {}

    def test_extract_single_kwarg(self):
        """Test extracting a single keyword argument."""
        context = Context()
        test_tensor = torch.randn(3, 224, 224)
        context.set_item("per_batch.input.rgb", test_tensor)

        getter = ContainerGetItem(kwargs={"image": "per_batch.input.rgb"})
        args, kwargs = getter(context)

        assert args == []
        assert len(kwargs) == 1
        assert "image" in kwargs
        assert torch.equal(kwargs["image"], test_tensor)

    def test_extract_multiple_kwargs(self):
        """Test extracting multiple keyword arguments."""
        context = Context()
        rgb_tensor = torch.randn(3, 224, 224)
        label_tensor = torch.tensor([1, 0, 1])
        context.set_item("per_batch.input.rgb", rgb_tensor)
        context.set_item("per_batch.target.label", label_tensor)

        getter = ContainerGetItem(kwargs={"image": "per_batch.input.rgb", "label": "per_batch.target.label"})
        args, kwargs = getter(context)

        assert args == []
        assert len(kwargs) == 2
        assert torch.equal(kwargs["image"], rgb_tensor)
        assert torch.equal(kwargs["label"], label_tensor)

    def test_extract_mixed_args_and_kwargs(self):
        """Test extracting both positional and keyword arguments."""
        context = Context()
        rgb_tensor = torch.randn(3, 224, 224)
        pca_matrix = torch.randn(50, 100)
        label_tensor = torch.tensor([1, 0, 1])

        context.set_item("per_batch.input.rgb", rgb_tensor)
        context.set_item("persistent.additional.pca_mat", pca_matrix)
        context.set_item("per_batch.target.label", label_tensor)

        getter = ContainerGetItem(
            args=["per_batch.input.rgb", "per_batch.target.label"], kwargs={"pca_mat": "persistent.additional.pca_mat"}
        )
        args, kwargs = getter(context)

        assert len(args) == 2
        assert torch.equal(args[0], rgb_tensor)
        assert torch.equal(args[1], label_tensor)
        assert len(kwargs) == 1
        assert torch.equal(kwargs["pca_mat"], pca_matrix)

    def test_extract_nested_path(self):
        """Test extracting values from deeply nested paths."""
        context = Context()
        nested_value = {"level2": {"level3": 42}}
        context.set_item("persistent.additional.level1", nested_value)

        getter = ContainerGetItem(args=["persistent.additional.level1"])
        args, kwargs = getter(context)

        assert len(args) == 1
        assert args[0] == nested_value

    def test_extract_from_different_lifecycle_scopes(self):
        """Test extracting values from different lifecycle containers."""
        context = Context()
        batch_data = torch.randn(32, 3, 224, 224)
        step_loss = 0.5
        epoch_metric = 0.95
        persistent_config = {"lr": 0.001}

        context.set_item("per_batch.input.data", batch_data)
        context.set_item("per_step.loss.total", step_loss)
        context.set_item("per_epoch.metrics.accuracy", epoch_metric)
        context.set_item("persistent.config.hyperparams", persistent_config)

        getter = ContainerGetItem(
            args=[
                "per_batch.input.data",
                "per_step.loss.total",
                "per_epoch.metrics.accuracy",
                "persistent.config.hyperparams",
            ]
        )
        args, kwargs = getter(context)

        assert len(args) == 4
        assert torch.equal(args[0], batch_data)
        assert args[1] == step_loss
        assert args[2] == epoch_metric
        assert args[3] == persistent_config

    def test_extract_nonexistent_path_raises_error(self):
        """Test that extracting from nonexistent path raises ValueError."""
        context = Context()
        getter = ContainerGetItem(args=["per_batch.input.nonexistent"])

        with pytest.raises(ValueError, match="Failed to extract value from path"):
            getter(context)

    def test_extract_invalid_path_raises_error(self):
        """Test that extracting with invalid path format raises ValueError."""
        context = Context()
        getter = ContainerGetItem(args=["invalid_path"])

        with pytest.raises(ValueError, match="Failed to extract value from path"):
            getter(context)

    def test_repr(self):
        """Test string representation of ContainerGetItem."""
        getter = ContainerGetItem(args=["per_batch.input.rgb"], kwargs={"label": "per_batch.target.label"})
        repr_str = repr(getter)
        assert "ContainerGetItem" in repr_str
        assert "per_batch.input.rgb" in repr_str
        assert "label" in repr_str


class TestContainerSetItem:
    """Test suite for ContainerSetItem class."""

    def test_init_with_single_path(self):
        """Test initialization with a single path string."""
        setter = ContainerSetItem("per_batch.prediction.output")
        assert setter.items == ["per_batch.prediction.output"]

    def test_init_with_multiple_paths(self):
        """Test initialization with a list of paths."""
        setter = ContainerSetItem(["per_step.loss.ce", "per_step.loss.reg"])
        assert setter.items == ["per_step.loss.ce", "per_step.loss.reg"]

    def test_init_with_empty_list_raises_error(self):
        """Test that initialization with empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one item path must be provided"):
            ContainerSetItem([])

    def test_set_single_value(self):
        """Test setting a single value."""
        context = Context()
        test_tensor = torch.randn(3, 224, 224)

        setter = ContainerSetItem("per_batch.prediction.output")
        result = setter(context, test_tensor)

        assert result is context  # Check method chaining
        retrieved = context.get_item("per_batch.prediction.output")
        assert torch.equal(retrieved, test_tensor)

    def test_set_multiple_values_with_list(self):
        """Test setting multiple values using a list."""
        context = Context()
        ce_loss = 0.3
        reg_loss = 0.2

        setter = ContainerSetItem(["per_step.loss.ce", "per_step.loss.reg"])
        setter(context, [ce_loss, reg_loss])

        assert context.get_item("per_step.loss.ce") == ce_loss
        assert context.get_item("per_step.loss.reg") == reg_loss

    def test_set_multiple_values_with_tuple(self):
        """Test setting multiple values using a tuple."""
        context = Context()
        ce_loss = 0.3
        reg_loss = 0.2

        setter = ContainerSetItem(["per_step.loss.ce", "per_step.loss.reg"])
        setter(context, (ce_loss, reg_loss))

        assert context.get_item("per_step.loss.ce") == ce_loss
        assert context.get_item("per_step.loss.reg") == reg_loss

    def test_set_value_with_weakref_true(self):
        """Test setting value with forced weak reference."""
        context = Context()
        test_tensor = torch.randn(3, 224, 224)

        setter = ContainerSetItem("per_batch.input.rgb")
        setter(context, test_tensor, use_weakref=True)

        # Value should still be retrievable
        retrieved = context.get_item("per_batch.input.rgb")
        assert torch.equal(retrieved, test_tensor)

    def test_set_value_with_weakref_false(self):
        """Test setting value with forced strong reference."""
        context = Context()
        test_value = 42

        setter = ContainerSetItem("per_step.additional.counter")
        setter(context, test_value, use_weakref=False)

        retrieved = context.get_item("per_step.additional.counter")
        assert retrieved == test_value

    def test_set_nested_path(self):
        """Test setting values at deeply nested paths."""
        context = Context()
        nested_value = {"key": "value"}

        setter = ContainerSetItem("persistent.additional.level1.level2.data")
        setter(context, nested_value)

        retrieved = context.get_item("persistent.additional.level1.level2.data")
        assert retrieved == nested_value

    def test_set_values_in_different_lifecycle_scopes(self):
        """Test setting values in different lifecycle containers."""
        context = Context()
        batch_data = torch.randn(32, 3, 224, 224)
        step_loss = 0.5
        epoch_metric = 0.95

        setter = ContainerSetItem(["per_batch.input.data", "per_step.loss.total", "per_epoch.metrics.accuracy"])
        setter(context, [batch_data, step_loss, epoch_metric])

        assert torch.equal(context.get_item("per_batch.input.data"), batch_data)
        assert context.get_item("per_step.loss.total") == step_loss
        assert context.get_item("per_epoch.metrics.accuracy") == epoch_metric

    def test_set_mismatched_values_count_raises_error(self):
        """Test that mismatched number of values and paths raises ValueError."""
        context = Context()
        setter = ContainerSetItem(["per_step.loss.ce", "per_step.loss.reg"])

        with pytest.raises(ValueError, match="Number of values .* doesn't match"):
            setter(context, [0.3])  # Only 1 value for 2 paths

    def test_set_too_many_values_raises_error(self):
        """Test that too many values raises ValueError."""
        context = Context()
        setter = ContainerSetItem("per_step.loss.total")

        with pytest.raises(ValueError, match="Number of values .* doesn't match"):
            setter(context, [0.3, 0.2])  # 2 values for 1 path

    def test_set_invalid_path_raises_error(self):
        """Test that setting with invalid path format raises ValueError."""
        context = Context()
        setter = ContainerSetItem("invalid_path")

        with pytest.raises(ValueError, match="Failed to set value at path"):
            setter(context, 42)

    def test_method_chaining(self):
        """Test that setter returns context for method chaining."""
        context = Context()
        setter1 = ContainerSetItem("per_batch.input.rgb")
        setter2 = ContainerSetItem("per_batch.target.label")

        tensor1 = torch.randn(3, 224, 224)
        tensor2 = torch.tensor([1, 0, 1])

        result = setter1(context, tensor1)
        assert result is context

        result = setter2(result, tensor2)
        assert result is context

        # Verify both values were set
        assert torch.equal(context.get_item("per_batch.input.rgb"), tensor1)
        assert torch.equal(context.get_item("per_batch.target.label"), tensor2)

    def test_overwrite_existing_value(self):
        """Test that setting overwrites existing values."""
        context = Context()
        setter = ContainerSetItem("per_step.loss.total")

        # Set initial value
        setter(context, 0.5)
        assert context.get_item("per_step.loss.total") == 0.5

        # Overwrite with new value
        setter(context, 0.3)
        assert context.get_item("per_step.loss.total") == 0.3

    def test_repr(self):
        """Test string representation of ContainerSetItem."""
        setter = ContainerSetItem(["per_step.loss.ce", "per_step.loss.reg"])
        repr_str = repr(setter)
        assert "ContainerSetItem" in repr_str
        assert "per_step.loss.ce" in repr_str
        assert "per_step.loss.reg" in repr_str


class TestIntegration:
    """Integration tests for ContainerGetItem and ContainerSetItem."""

    def test_get_and_set_workflow(self):
        """Test typical workflow of getting, transforming, and setting values."""
        context = Context()

        # Setup initial data
        input_tensor = torch.randn(32, 3, 224, 224)
        context.set_item("per_batch.input.rgb", input_tensor)

        # Get the input
        getter = ContainerGetItem(args=["per_batch.input.rgb"])
        args, _ = getter(context)

        # Transform (simple example: multiply by 2)
        transformed = args[0] * 2

        # Set the output
        setter = ContainerSetItem("per_batch.prediction.output")
        setter(context, transformed)

        # Verify
        output = context.get_item("per_batch.prediction.output")
        assert torch.equal(output, input_tensor * 2)

    def test_pipeline_with_multiple_transforms(self):
        """Test a pipeline with multiple get/transform/set operations."""
        context = Context()

        # Initial setup
        rgb = torch.randn(3, 224, 224)
        depth = torch.randn(1, 224, 224)
        context.set_item("per_batch.input.rgb", rgb)
        context.set_item("per_batch.input.depth", depth)

        # First transform: concatenate inputs
        getter1 = ContainerGetItem(args=["per_batch.input.rgb", "per_batch.input.depth"])
        args, _ = getter1(context)
        concatenated = torch.cat(args, dim=0)

        setter1 = ContainerSetItem("per_batch.prediction.features")
        setter1(context, concatenated)

        # Second transform: compute mean
        getter2 = ContainerGetItem(args=["per_batch.prediction.features"])
        args, _ = getter2(context)
        mean_value = args[0].mean().item()

        setter2 = ContainerSetItem("per_step.additional.mean_activation")
        setter2(context, mean_value)

        # Verify
        features = context.get_item("per_batch.prediction.features")
        assert features.shape[0] == 4  # 3 + 1 channels

        mean = context.get_item("per_step.additional.mean_activation")
        assert isinstance(mean, float)

    def test_complex_kwargs_workflow(self):
        """Test workflow with complex keyword argument mappings."""
        context = Context()

        # Setup
        net_output = torch.randn(32, 10)
        ground_truth = torch.randint(0, 10, (32,))
        pca_matrix = torch.randn(50, 10)

        context.set_item("per_batch.prediction.net_output", net_output)
        context.set_item("per_batch.target.ground_truth", ground_truth)
        context.set_item("persistent.additional.pca_mat", pca_matrix)

        # Get with named parameters
        getter = ContainerGetItem(
            kwargs={
                "predictions": "per_batch.prediction.net_output",
                "targets": "per_batch.target.ground_truth",
                "transform_matrix": "persistent.additional.pca_mat",
            }
        )
        _, kwargs = getter(context)

        # Verify we got the right parameters
        assert "predictions" in kwargs
        assert "targets" in kwargs
        assert "transform_matrix" in kwargs
        assert torch.equal(kwargs["predictions"], net_output)
        assert torch.equal(kwargs["targets"], ground_truth)
        assert torch.equal(kwargs["transform_matrix"], pca_matrix)
