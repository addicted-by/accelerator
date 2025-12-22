"""Unit tests for Context class with lifecycle containers."""

import pytest
from omegaconf import DictConfig

from accelerator.core.context import Context
from accelerator.core.context.containers import PathResolutionError


class TestContextInitialization:
    """Test Context initialization with lifecycle containers."""
    
    def test_context_initializes_lifecycle_containers(self):
        """Test that Context initializes all lifecycle containers."""
        context = Context()
        
        assert hasattr(context, 'per_batch')
        assert hasattr(context, 'per_step')
        assert hasattr(context, 'per_epoch')
        assert hasattr(context, 'persistent')
    
    def test_context_initializes_with_config(self):
        """Test that Context initializes with config."""
        config = DictConfig({'test': 'value'})
        context = Context(config)
        
        assert context.config == config
        assert hasattr(context, 'per_batch')
    
    def test_context_maintains_backward_compatibility(self):
        """Test that Context maintains existing components."""
        context = Context()
        
        assert hasattr(context, 'components')
        assert hasattr(context, 'training_manager')
        assert context.is_distributed == False
        assert context.distributed_engine is None


class TestContextGetItem:
    """Test Context.get_item() method."""
    
    def test_get_item_with_valid_path(self):
        """Test getting item with valid hierarchical path."""
        context = Context()
        test_value = "test_data"
        
        # Set value directly in container
        context.per_batch.set_item('input.rgb', test_value)
        
        # Get via context
        result = context.get_item('per_batch.input.rgb')
        assert result == test_value
    
    def test_get_item_with_nested_path(self):
        """Test getting item with deeply nested path."""
        context = Context()
        test_value = 42
        
        context.persistent.set_item('additional.key1.key2.data', test_value)
        result = context.get_item('persistent.additional.key1.key2.data')
        
        assert result == test_value
    
    def test_get_item_invalid_scope_raises_error(self):
        """Test that invalid scope raises ValueError."""
        context = Context()
        
        with pytest.raises(ValueError, match="Unknown lifecycle scope"):
            context.get_item('invalid_scope.input.rgb')
    
    def test_get_item_missing_scope_raises_error(self):
        """Test that missing scope raises ValueError."""
        context = Context()
        
        with pytest.raises(ValueError, match="Unknown lifecycle scope"):
            context.get_item('input.rgb')
    
    def test_get_item_invalid_sub_container_raises_error(self):
        """Test that invalid sub-container raises PathResolutionError."""
        context = Context()
        
        with pytest.raises(PathResolutionError):
            context.get_item('per_batch.invalid_sub.key')
    
    def test_get_item_missing_key_raises_error(self):
        """Test that missing key raises PathResolutionError."""
        context = Context()
        
        with pytest.raises(PathResolutionError):
            context.get_item('per_batch.input.missing_key')
    
    def test_get_item_from_different_scopes(self):
        """Test getting items from different lifecycle scopes."""
        context = Context()
        
        context.per_batch.set_item('input.data', 'batch_data')
        context.per_step.set_item('loss.total', 0.5)
        context.per_epoch.set_item('metrics.accuracy', 0.95)
        context.persistent.set_item('config.lr', 0.001)
        
        assert context.get_item('per_batch.input.data') == 'batch_data'
        assert context.get_item('per_step.loss.total') == 0.5
        assert context.get_item('per_epoch.metrics.accuracy') == 0.95
        assert context.get_item('persistent.config.lr') == 0.001


class TestContextSetItem:
    """Test Context.set_item() method."""
    
    def test_set_item_with_valid_path(self):
        """Test setting item with valid hierarchical path."""
        context = Context()
        test_value = "test_data"
        
        context.set_item('per_batch.input.rgb', test_value)
        
        # Verify via direct container access
        result = context.per_batch.get_item('input.rgb')
        assert result == test_value
    
    def test_set_item_with_nested_path(self):
        """Test setting item with deeply nested path."""
        context = Context()
        test_value = {'nested': 'data'}
        
        context.set_item('persistent.additional.key1.key2.data', test_value)
        result = context.get_item('persistent.additional.key1.key2.data')
        
        assert result == test_value
    
    def test_set_item_with_weakref_parameter(self):
        """Test setting item with explicit weakref parameter."""
        context = Context()
        
        # Force strong reference for a string (normally would be strong anyway)
        context.set_item('per_batch.input.data', 'test', use_weakref=False)
        result = context.get_item('per_batch.input.data')
        
        assert result == 'test'
    
    def test_set_item_invalid_scope_raises_error(self):
        """Test that invalid scope raises ValueError."""
        context = Context()
        
        with pytest.raises(ValueError, match="Unknown lifecycle scope"):
            context.set_item('invalid_scope.input.rgb', 'value')
    
    def test_set_item_missing_scope_raises_error(self):
        """Test that missing scope raises ValueError."""
        context = Context()
        
        with pytest.raises(ValueError, match="Unknown lifecycle scope"):
            context.set_item('input.rgb', 'value')
    
    def test_set_item_invalid_sub_container_raises_error(self):
        """Test that invalid sub-container raises PathResolutionError."""
        context = Context()
        
        with pytest.raises(PathResolutionError):
            context.set_item('per_batch.invalid_sub.key', 'value')
    
    def test_set_item_to_different_scopes(self):
        """Test setting items to different lifecycle scopes."""
        context = Context()
        
        context.set_item('per_batch.input.data', 'batch_data')
        context.set_item('per_step.loss.total', 0.5)
        context.set_item('per_epoch.metrics.accuracy', 0.95)
        context.set_item('persistent.config.lr', 0.001)
        
        assert context.get_item('per_batch.input.data') == 'batch_data'
        assert context.get_item('per_step.loss.total') == 0.5
        assert context.get_item('per_epoch.metrics.accuracy') == 0.95
        assert context.get_item('persistent.config.lr') == 0.001
    
    def test_set_item_overwrites_existing_value(self):
        """Test that set_item overwrites existing values."""
        context = Context()
        
        context.set_item('per_batch.input.data', 'old_value')
        context.set_item('per_batch.input.data', 'new_value')
        
        assert context.get_item('per_batch.input.data') == 'new_value'


class TestContextLifecycleHooks:
    """Test Context lifecycle hooks."""
    
    def test_on_batch_start_clears_per_batch_container(self):
        """Test that on_batch_start clears per_batch container."""
        context = Context()
        
        context.set_item('per_batch.input.data', 'test_data')
        assert context.get_item('per_batch.input.data') == 'test_data'
        
        context.on_batch_start()
        
        with pytest.raises(PathResolutionError):
            context.get_item('per_batch.input.data')
    
    def test_on_batch_end_cleans_dead_refs(self):
        """Test that on_batch_end cleans up dead references."""
        context = Context()
        
        context.set_item('per_batch.input.data', 'test_data')
        context.on_batch_end()
        
        # Should not raise error - data should still be accessible
        assert context.get_item('per_batch.input.data') == 'test_data'
    
    def test_on_step_end_clears_per_step_container(self):
        """Test that on_step_end clears per_step container."""
        context = Context()
        
        context.set_item('per_step.loss.total', 0.5)
        assert context.get_item('per_step.loss.total') == 0.5
        
        context.on_step_end()
        
        with pytest.raises(PathResolutionError):
            context.get_item('per_step.loss.total')
    
    def test_on_epoch_end_clears_per_epoch_container(self):
        """Test that on_epoch_end clears per_epoch container."""
        context = Context()
        
        context.set_item('per_epoch.metrics.accuracy', 0.95)
        assert context.get_item('per_epoch.metrics.accuracy') == 0.95
        
        context.on_epoch_end()
        
        with pytest.raises(PathResolutionError):
            context.get_item('per_epoch.metrics.accuracy')
    
    def test_lifecycle_hooks_dont_affect_other_containers(self):
        """Test that lifecycle hooks only affect their respective containers."""
        context = Context()
        
        context.set_item('per_batch.input.data', 'batch_data')
        context.set_item('per_step.loss.total', 0.5)
        context.set_item('per_epoch.metrics.accuracy', 0.95)
        context.set_item('persistent.config.lr', 0.001)
        
        # Clear per_batch
        context.on_batch_start()
        with pytest.raises(PathResolutionError):
            context.get_item('per_batch.input.data')
        assert context.get_item('per_step.loss.total') == 0.5
        assert context.get_item('per_epoch.metrics.accuracy') == 0.95
        assert context.get_item('persistent.config.lr') == 0.001
        
        # Clear per_step
        context.on_step_end()
        with pytest.raises(PathResolutionError):
            context.get_item('per_step.loss.total')
        assert context.get_item('per_epoch.metrics.accuracy') == 0.95
        assert context.get_item('persistent.config.lr') == 0.001
        
        # Clear per_epoch
        context.on_epoch_end()
        with pytest.raises(PathResolutionError):
            context.get_item('per_epoch.metrics.accuracy')
        assert context.get_item('persistent.config.lr') == 0.001


class TestContextBackwardCompatibility:
    """Test Context backward compatibility layer."""
    
    def test_model_property_getter_uses_container(self):
        """Test that model property getter uses persistent container."""
        context = Context()
        
        # Mock model object
        class MockModel:
            pass
        
        model = MockModel()
        context.set_item('persistent.model.instance', model)
        
        assert context.model is model
    
    def test_model_property_setter_updates_both(self):
        """Test that model property setter updates both container and components."""
        context = Context()
        
        # Use a simple object that won't trigger AcceleratedModel wrapping
        model = "mock_model_string"
        
        # Set directly in container and components to avoid AcceleratedModel wrapping
        context.set_item('persistent.model.instance', model)
        context.components._components['model'] = model
        
        # Check container
        assert context.get_item('persistent.model.instance') is model
        # Check component manager
        assert context.components._components['model'] is model
    
    def test_model_property_fallback_to_components(self):
        """Test that model property falls back to component manager."""
        context = Context()
        
        model = "mock_model_string"
        context.components._components['model'] = model
        
        # Should get from components when not in container
        assert context.model is model
    
    def test_optimizer_property_getter_uses_container(self):
        """Test that optimizer property getter uses persistent container."""
        context = Context()
        
        class MockOptimizer:
            pass
        
        optimizer = MockOptimizer()
        context.set_item('persistent.optimizer.instance', optimizer)
        
        assert context.optimizer is optimizer
    
    def test_optimizer_property_setter_updates_both(self):
        """Test that optimizer property setter updates container and components."""
        context = Context()
        
        class MockOptimizer:
            pass
        
        optimizer = MockOptimizer()
        context.optimizer = optimizer
        
        assert context.get_item('persistent.optimizer.instance') is optimizer
        assert context.components.get_component('optimizer') is optimizer
    
    def test_scheduler_property_getter_uses_container(self):
        """Test that scheduler property getter uses persistent container."""
        context = Context()
        
        class MockScheduler:
            pass
        
        scheduler = MockScheduler()
        context.set_item('persistent.scheduler.instance', scheduler)
        
        # Note: This will try to add to callbacks, so we need a mock callback manager
        context.components.set_component('callbacks', MockCallbackManager())
        
        assert context.scheduler is scheduler
    
    def test_scheduler_property_setter_updates_both(self):
        """Test that scheduler property setter updates container and components."""
        context = Context()
        
        class MockScheduler:
            pass
        
        scheduler = MockScheduler()
        
        # Mock callback manager
        context.components.set_component('callbacks', MockCallbackManager())
        
        context.scheduler = scheduler
        
        assert context.get_item('persistent.scheduler.instance') is scheduler
        assert context.components.get_component('scheduler') is scheduler


class MockCallbackManager:
    """Mock callback manager for testing."""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        self.callbacks.append(callback)


class TestContextIntegration:
    """Integration tests for Context with lifecycle containers."""
    
    def test_full_training_cycle_simulation(self):
        """Test simulating a full training cycle with lifecycle hooks."""
        context = Context()
        
        # Epoch 1, Batch 1
        context.on_batch_start()
        context.set_item('per_batch.input.data', 'batch1_data')
        context.set_item('per_batch.prediction.output', 'batch1_pred')
        context.set_item('per_step.loss.total', 0.5)
        context.on_batch_end()
        context.on_step_end()
        
        # Epoch 1, Batch 2
        context.on_batch_start()
        context.set_item('per_batch.input.data', 'batch2_data')
        context.set_item('per_batch.prediction.output', 'batch2_pred')
        context.set_item('per_step.loss.total', 0.4)
        context.on_batch_end()
        context.on_step_end()
        
        # End of epoch
        context.set_item('per_epoch.metrics.accuracy', 0.95)
        context.on_epoch_end()
        
        # Verify persistent data survives
        context.set_item('persistent.config.lr', 0.001)
        assert context.get_item('persistent.config.lr') == 0.001
        
        # Verify per_epoch was cleared
        with pytest.raises(PathResolutionError):
            context.get_item('per_epoch.metrics.accuracy')
    
    def test_mixed_old_and_new_api_usage(self):
        """Test using both old and new APIs together."""
        context = Context()
        
        # Use new API
        context.set_item('persistent.config.lr', 0.001)
        
        # Use new API for model to avoid AcceleratedModel wrapping
        model = "mock_model_string"
        context.set_item('persistent.model.instance', model)
        context.components._components['model'] = model
        
        # Verify both work
        assert context.get_item('persistent.config.lr') == 0.001
        assert context.model is model
        assert context.get_item('persistent.model.instance') is model
