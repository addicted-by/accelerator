"""Unit tests for lifecycle containers."""

import pytest
import weakref
from accelerator.core.context.containers import (
    BaseLifecycleContainer,
    PerBatchContainer,
    PerStepContainer,
    PerEpochContainer,
    PersistentContainer,
    ReferenceManager,
    PathResolutionError,
    DeadReferenceError,
)


class TestReferenceManager:
    """Tests for ReferenceManager."""
    
    def test_should_use_weakref_primitives(self):
        """Test that primitives use strong references."""
        assert not ReferenceManager.should_use_weakref(42)
        assert not ReferenceManager.should_use_weakref(3.14)
        assert not ReferenceManager.should_use_weakref("string")
        assert not ReferenceManager.should_use_weakref(True)
        assert not ReferenceManager.should_use_weakref(None)
    
    def test_should_use_weakref_collections(self):
        """Test that collections use strong references (they don't support weakref)."""
        assert not ReferenceManager.should_use_weakref([1, 2, 3])
        assert not ReferenceManager.should_use_weakref({"a": 1, "b": 2})
        assert not ReferenceManager.should_use_weakref((1, 2, 3))
        assert not ReferenceManager.should_use_weakref({1, 2, 3})
        
        # Even large collections don't support weakref
        large_list = list(range(100))
        large_dict = {i: i for i in range(100)}
        assert not ReferenceManager.should_use_weakref(large_list)
        assert not ReferenceManager.should_use_weakref(large_dict)
    
    def test_should_use_weakref_custom_objects(self):
        """Test that custom objects default to weak references."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        assert ReferenceManager.should_use_weakref(obj)
    
    def test_create_reference_strong(self):
        """Test creating strong references."""
        value = 42
        ref, is_weakref = ReferenceManager.create_reference(value, use_weakref=False)
        assert ref == 42
        assert not is_weakref
    
    def test_create_reference_weak(self):
        """Test creating weak references."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj, use_weakref=True)
        assert isinstance(ref, weakref.ref)
        assert is_weakref
        assert ref() is obj
    
    def test_create_reference_weak_unsupported(self):
        """Test creating weak reference for unsupported type falls back to strong."""
        value = 42
        ref, is_weakref = ReferenceManager.create_reference(value, use_weakref=True)
        assert ref == 42
        assert not is_weakref
    
    def test_create_reference_auto(self):
        """Test automatic reference type selection."""
        # Primitive should be strong
        ref, is_weakref = ReferenceManager.create_reference(42)
        assert not is_weakref
        
        # Custom object should be weak
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj)
        assert is_weakref
    
    def test_dereference_strong(self):
        """Test dereferencing strong references."""
        value = 42
        result = ReferenceManager.dereference(value, is_weakref=False)
        assert result == 42
    
    def test_dereference_weak(self):
        """Test dereferencing weak references."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref = weakref.ref(obj)
        result = ReferenceManager.dereference(ref, is_weakref=True)
        assert result is obj
    
    def test_dereference_dead_weak(self):
        """Test dereferencing dead weak reference raises error."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref = weakref.ref(obj)
        del obj  # Delete the object
        
        with pytest.raises(DeadReferenceError):
            ReferenceManager.dereference(ref, is_weakref=True)


class TestPerBatchContainer:
    """Tests for PerBatchContainer."""
    
    def test_initialization(self):
        """Test container initializes with correct sub-containers."""
        container = PerBatchContainer()
        assert container._sub_containers == ['input', 'prediction', 'target', 'additional']
        assert all(sub in container._data for sub in container._sub_containers)
    
    def test_set_and_get_simple(self):
        """Test setting and getting simple values."""
        container = PerBatchContainer()
        container.set_item('input.rgb', [1, 2, 3])
        result = container.get_item('input.rgb')
        assert result == [1, 2, 3]
    
    def test_set_and_get_nested(self):
        """Test setting and getting nested values."""
        container = PerBatchContainer()
        container.set_item('additional.key1.key2.data', 'value')
        result = container.get_item('additional.key1.key2.data')
        assert result == 'value'
    
    def test_get_entire_subcontainer(self):
        """Test getting entire sub-container."""
        container = PerBatchContainer()
        container.set_item('input.rgb', [1, 2, 3])
        container.set_item('input.depth', [4, 5, 6])
        result = container.get_item('input')
        assert 'rgb' in result
        assert 'depth' in result
    
    def test_invalid_subcontainer(self):
        """Test accessing invalid sub-container raises error."""
        container = PerBatchContainer()
        with pytest.raises(PathResolutionError) as exc_info:
            container.get_item('invalid.key')
        assert 'invalid' in str(exc_info.value)
        assert 'Sub-container' in str(exc_info.value)
    
    def test_missing_key(self):
        """Test accessing missing key raises error."""
        container = PerBatchContainer()
        with pytest.raises(PathResolutionError) as exc_info:
            container.get_item('input.missing')
        assert 'missing' in str(exc_info.value)
    
    def test_navigate_through_non_dict(self):
        """Test navigating through non-dict raises error."""
        container = PerBatchContainer()
        container.set_item('input.value', 42)
        with pytest.raises(PathResolutionError) as exc_info:
            container.get_item('input.value.nested')
        assert 'non-dict' in str(exc_info.value)
    
    def test_cannot_replace_subcontainer(self):
        """Test that replacing entire sub-container raises error."""
        container = PerBatchContainer()
        with pytest.raises(PathResolutionError) as exc_info:
            container.set_item('input', {'new': 'data'})
        assert 'Cannot replace entire sub-container' in str(exc_info.value)
    
    def test_clear(self):
        """Test clearing container."""
        container = PerBatchContainer()
        container.set_item('input.rgb', [1, 2, 3])
        container.set_item('prediction.output', [4, 5, 6])
        container.clear()
        
        # Should be able to set again after clear
        container.set_item('input.rgb', [7, 8, 9])
        assert container.get_item('input.rgb') == [7, 8, 9]
    
    def test_weakref_management(self):
        """Test weak reference management."""
        import gc
        
        container = PerBatchContainer()
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj = CustomObject(42)
        container.set_item('additional.obj', obj, use_weakref=True)
        
        # Should be able to retrieve
        result = container.get_item('additional.obj')
        assert result.value == 42
        
        # Delete original object and force garbage collection
        del obj
        del result  # Also delete the retrieved reference
        gc.collect()  # Force garbage collection
        
        # Should raise DeadReferenceError
        with pytest.raises(DeadReferenceError):
            container.get_item('additional.obj')
    
    def test_cleanup_dead_refs(self):
        """Test cleanup of dead weak references."""
        container = PerBatchContainer()
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj1 = CustomObject(1)
        obj2 = CustomObject(2)
        
        container.set_item('additional.obj1', obj1, use_weakref=True)
        container.set_item('additional.obj2', obj2, use_weakref=True)
        container.set_item('additional.value', 42, use_weakref=False)
        
        # Delete one object
        del obj1
        
        # Cleanup should remove dead ref
        container.cleanup_dead_refs()
        
        # obj2 should still be accessible
        assert container.get_item('additional.obj2').value == 2
        
        # value should still be accessible
        assert container.get_item('additional.value') == 42
    
    def test_nested_dict_creation(self):
        """Test that intermediate dicts are created automatically."""
        container = PerBatchContainer()
        container.set_item('additional.level1.level2.level3', 'deep')
        result = container.get_item('additional.level1.level2.level3')
        assert result == 'deep'
    
    def test_cannot_create_path_through_non_dict(self):
        """Test that creating path through non-dict raises error."""
        container = PerBatchContainer()
        container.set_item('additional.value', 42)
        with pytest.raises(PathResolutionError) as exc_info:
            container.set_item('additional.value.nested', 'fail')
        assert 'non-dict' in str(exc_info.value)


class TestPerStepContainer:
    """Tests for PerStepContainer."""
    
    def test_initialization(self):
        """Test container initializes with correct sub-containers."""
        container = PerStepContainer()
        expected = ['gradients', 'gradient_masks', 'gradient_metadata', 'loss', 'additional']
        assert container._sub_containers == expected
        assert all(sub in container._data for sub in container._sub_containers)
    
    def test_basic_operations(self):
        """Test basic set and get operations."""
        container = PerStepContainer()
        container.set_item('loss.total', 0.5)
        container.set_item('loss.ce', 0.3)
        assert container.get_item('loss.total') == 0.5
        assert container.get_item('loss.ce') == 0.3


class TestPerEpochContainer:
    """Tests for PerEpochContainer."""
    
    def test_initialization(self):
        """Test container initializes with correct sub-containers."""
        container = PerEpochContainer()
        expected = ['metrics', 'statistics', 'validation', 'checkpoints', 'additional']
        assert container._sub_containers == expected
        assert all(sub in container._data for sub in container._sub_containers)
    
    def test_basic_operations(self):
        """Test basic set and get operations."""
        container = PerEpochContainer()
        container.set_item('metrics.accuracy', 0.95)
        container.set_item('validation.val_loss', 0.1)
        assert container.get_item('metrics.accuracy') == 0.95
        assert container.get_item('validation.val_loss') == 0.1


class TestPersistentContainer:
    """Tests for PersistentContainer."""
    
    def test_initialization(self):
        """Test container initializes with correct sub-containers."""
        container = PersistentContainer()
        expected = ['model', 'optimizer', 'scheduler', 'config', 'additional']
        assert container._sub_containers == expected
        assert all(sub in container._data for sub in container._sub_containers)
    
    def test_basic_operations(self):
        """Test basic set and get operations."""
        container = PersistentContainer()
        config = {'learning_rate': 0.001}
        container.set_item('config.training', config)
        assert container.get_item('config.training') == config


class TestContainerUpdate:
    """Tests for container update method."""
    
    def test_update_with_dict(self):
        """Test updating container with dict source."""
        container = PerBatchContainer()
        container.update({
            'input': {'rgb': [1, 2, 3], 'depth': [4, 5, 6]},
            'target': {'labels': [0, 1, 2]}
        })
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('input.depth') == [4, 5, 6]
        assert container.get_item('target.labels') == [0, 1, 2]
    
    def test_update_with_nested_dict(self):
        """Test updating container with deeply nested dict."""
        container = PerBatchContainer()
        container.update({
            'additional': {
                'metadata': {
                    'level1': {
                        'level2': {
                            'level3': 'deep_value'
                        }
                    }
                }
            }
        })
        
        assert container.get_item('additional.metadata.level1.level2.level3') == 'deep_value'
    
    def test_update_with_object(self):
        """Test updating container with object source."""
        class DataSource:
            def __init__(self):
                self.input = {'rgb': [1, 2, 3]}
                self.target = {'labels': [0, 1]}
        
        container = PerBatchContainer()
        source = DataSource()
        container.update(source)
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('target.labels') == [0, 1]
    
    def test_update_with_nested_objects(self):
        """Test updating container with nested objects."""
        class NestedData:
            def __init__(self):
                self.value = 42
        
        class DataSource:
            def __init__(self):
                self.additional = {'nested': NestedData()}
        
        container = PerBatchContainer()
        source = DataSource()
        container.update(source)
        
        # The nested object should be converted to dict
        result = container.get_item('additional.nested')
        assert isinstance(result, dict)
        assert result['value'] == 42
    
    def test_update_merges_existing_data(self):
        """Test that update merges with existing data."""
        container = PerBatchContainer()
        container.set_item('input.rgb', [1, 2, 3])
        container.set_item('input.depth', [4, 5, 6])
        
        # Update with new data
        container.update({
            'input': {'rgb': [7, 8, 9], 'ir': [10, 11, 12]}
        })
        
        # rgb should be updated, depth should remain, ir should be added
        assert container.get_item('input.rgb') == [7, 8, 9]
        assert container.get_item('input.depth') == [4, 5, 6]
        assert container.get_item('input.ir') == [10, 11, 12]
    
    def test_update_with_weakref_control(self):
        """Test update with explicit weakref control."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        container = PerBatchContainer()
        obj1 = CustomObject(1)
        obj2 = CustomObject(2)
        
        # Update with forced weak references
        container.update({
            'additional': {'obj1': obj1, 'obj2': obj2}
        }, use_weakref=True)
        
        # Should be able to retrieve
        assert container.get_item('additional.obj1').value == 1
        assert container.get_item('additional.obj2').value == 2
    
    def test_update_invalid_subcontainer(self):
        """Test update with invalid sub-container raises error."""
        container = PerBatchContainer()
        with pytest.raises(PathResolutionError) as exc_info:
            container.update({'invalid': {'key': 'value'}})
        assert 'invalid' in str(exc_info.value)
        assert 'not found' in str(exc_info.value)
    
    def test_update_with_non_dict_subcontainer_value(self):
        """Test update with non-dict sub-container value raises error."""
        container = PerBatchContainer()
        with pytest.raises(PathResolutionError) as exc_info:
            container.update({'input': 'not_a_dict'})
        assert 'dict-like' in str(exc_info.value)
    
    def test_update_with_unsupported_type(self):
        """Test update with unsupported source type raises error."""
        container = PerBatchContainer()
        with pytest.raises(TypeError) as exc_info:
            container.update(42)
        assert 'Cannot update from source' in str(exc_info.value)
    
    def test_update_with_dict_like_object(self):
        """Test update with dict-like object (has items method)."""
        from collections import OrderedDict
        
        container = PerBatchContainer()
        source = OrderedDict([
            ('input', {'rgb': [1, 2, 3]}),
            ('target', {'labels': [0, 1]})
        ])
        container.update(source)
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('target.labels') == [0, 1]
    
    def test_update_replaces_non_dict_with_dict(self):
        """Test that update replaces non-dict values with dicts when needed."""
        container = PerBatchContainer()
        container.set_item('additional.value', 42)
        
        # Update with nested dict at same path
        container.update({
            'additional': {'value': {'nested': 'data'}}
        })
        
        # Should replace the int with dict
        assert container.get_item('additional.value.nested') == 'data'
    
    def test_update_multiple_subcontainers(self):
        """Test updating multiple sub-containers at once."""
        container = PerBatchContainer()
        container.update({
            'input': {'rgb': [1, 2, 3]},
            'prediction': {'output': [4, 5, 6]},
            'target': {'labels': [0, 1, 2]},
            'additional': {'metadata': {'key': 'value'}}
        })
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('prediction.output') == [4, 5, 6]
        assert container.get_item('target.labels') == [0, 1, 2]
        assert container.get_item('additional.metadata.key') == 'value'
    
    def test_update_with_list_of_pairs(self):
        """Test updating container with list of (key, value) pairs."""
        container = PerBatchContainer()
        container.update([
            ('input', {'rgb': [1, 2, 3]}),
            ('target', {'labels': [0, 1, 2]})
        ])
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('target.labels') == [0, 1, 2]
    
    def test_update_with_tuple_of_pairs(self):
        """Test updating container with tuple of (key, value) pairs."""
        container = PerBatchContainer()
        container.update((
            ('input', {'rgb': [1, 2, 3]}),
            ('target', {'labels': [0, 1, 2]})
        ))
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('target.labels') == [0, 1, 2]
    
    def test_update_with_nested_list_pairs(self):
        """Test updating with nested list of pairs."""
        container = PerBatchContainer()
        container.update({
            'additional': [
                ('key1', 'value1'),
                ('key2', 'value2'),
                ('nested', [('deep_key', 'deep_value')])
            ]
        })
        
        assert container.get_item('additional.key1') == 'value1'
        assert container.get_item('additional.key2') == 'value2'
        assert container.get_item('additional.nested.deep_key') == 'deep_value'
    
    def test_update_with_nested_tuple_pairs(self):
        """Test updating with nested tuple of pairs."""
        container = PerBatchContainer()
        container.update({
            'additional': (
                ('key1', 'value1'),
                ('nested', (('deep_key', 'deep_value'),))
            )
        })
        
        assert container.get_item('additional.key1') == 'value1'
        assert container.get_item('additional.nested.deep_key') == 'deep_value'
    
    def test_update_with_mixed_types(self):
        """Test updating with mixed dict, list, tuple, and object types."""
        class NestedData:
            def __init__(self):
                self.obj_value = 'from_object'
        
        container = PerBatchContainer()
        container.update({
            'input': {'rgb': [1, 2, 3]},
            'target': [('labels', [0, 1, 2])],
            'prediction': (('output', [4, 5, 6]),),
            'additional': {'nested_obj': NestedData()}
        })
        
        assert container.get_item('input.rgb') == [1, 2, 3]
        assert container.get_item('target.labels') == [0, 1, 2]
        assert container.get_item('prediction.output') == [4, 5, 6]
        assert container.get_item('additional.nested_obj.obj_value') == 'from_object'
    
    def test_update_with_invalid_list_format(self):
        """Test that invalid list format raises appropriate error."""
        container = PerBatchContainer()
        # List that's not pairs should fail
        with pytest.raises(TypeError):
            container.update([1, 2, 3])
    
    def test_update_preserves_list_values(self):
        """Test that list values (not pairs) are stored as-is."""
        container = PerBatchContainer()
        container.update({
            'input': {'data': [1, 2, 3, 4, 5]}
        })
        
        result = container.get_item('input.data')
        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)
    
    def test_update_with_single_tensor(self):
        """Test updating with a single tensor wraps it as 'original'."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        container = PerBatchContainer()
        tensor = torch.tensor([1, 2, 3])
        container.update({'input': tensor})
        
        # Should be able to get it directly from sub-container
        result = container.get_item('input')
        assert torch.equal(result, tensor)
        
        # Should also be able to get it via 'original' key
        result_original = container.get_item('input.original')
        assert torch.equal(result_original, tensor)
    
    def test_update_with_tuple_of_tensors(self):
        """Test updating with tuple of tensors wraps it as 'original'."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        container = PerBatchContainer()
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5, 6])
        container.update({'input': (tensor1, tensor2)})
        
        # Should be able to get it directly from sub-container
        result = container.get_item('input')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)
        
        # Should also be able to get it via 'original' key
        result_original = container.get_item('input.original')
        assert isinstance(result_original, tuple)
        assert torch.equal(result_original[0], tensor1)
    
    def test_update_with_list_of_tensors(self):
        """Test updating with list of tensors wraps it as 'original'."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        container = PerBatchContainer()
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5, 6])
        container.update({'input': [tensor1, tensor2]})
        
        # Should be able to get it directly from sub-container
        result = container.get_item('input')
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], tensor1)
        assert torch.equal(result[1], tensor2)
    
    def test_update_with_numpy_array(self):
        """Test updating with numpy array wraps it as 'original'."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not available")
        
        container = PerBatchContainer()
        array = np.array([1, 2, 3])
        container.update({'input': array})
        
        # Should be able to get it directly from sub-container
        result = container.get_item('input')
        assert np.array_equal(result, array)
        
        # Should also be able to get it via 'original' key
        result_original = container.get_item('input.original')
        assert np.array_equal(result_original, array)
    
    def test_update_with_mixed_tensor_and_dict(self):
        """Test updating with both tensors and regular dict data."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
        
        container = PerBatchContainer()
        tensor = torch.tensor([1, 2, 3])
        container.update({
            'input': tensor,
            'target': {'labels': [0, 1, 2]},
            'additional': {'metadata': 'test'}
        })
        
        # Tensor should be unwrapped
        assert torch.equal(container.get_item('input'), tensor)
        
        # Regular dict should work normally
        assert container.get_item('target.labels') == [0, 1, 2]
        assert container.get_item('additional.metadata') == 'test'
    
    def test_get_item_unwraps_single_original_key(self):
        """Test that get_item unwraps dict with only 'original' key."""
        container = PerBatchContainer()
        container.set_item('input.original', [1, 2, 3])
        
        # Getting the sub-container should unwrap the 'original' key
        result = container.get_item('input')
        assert result == [1, 2, 3]
        
        # But if there are other keys, it should return the dict
        container.set_item('input.other', [4, 5, 6])
        result_dict = container.get_item('input')
        assert isinstance(result_dict, dict)
        assert 'original' in result_dict
        assert 'other' in result_dict
    
    def test_update_with_primitive_value_wraps_as_original(self):
        """Test that primitive values are wrapped as 'original'."""
        container = PerBatchContainer()
        container.update({'input': 42})
        
        # Should be able to get it directly
        result = container.get_item('input')
        assert result == 42
        
        # Should also be accessible via 'original'
        result_original = container.get_item('input.original')
        assert result_original == 42
