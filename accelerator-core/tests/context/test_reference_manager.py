"""Unit tests for reference management system."""

import gc
import weakref
import pytest
from accelerator.core.context.reference_manager import (
    ReferenceManager,
    DeadReferenceError,
)


class TestShouldUseWeakref:
    """Tests for ReferenceManager.should_use_weakref()."""
    
    def test_primitives_no_weakref(self):
        """Primitives should never use weak references."""
        assert ReferenceManager.should_use_weakref(42) is False
        assert ReferenceManager.should_use_weakref(3.14) is False
        assert ReferenceManager.should_use_weakref("string") is False
        assert ReferenceManager.should_use_weakref(True) is False
        assert ReferenceManager.should_use_weakref(None) is False
    
    def test_collections_no_weakref(self):
        """Built-in collections should not use weak references (they don't support them)."""
        assert ReferenceManager.should_use_weakref([1, 2, 3]) is False
        assert ReferenceManager.should_use_weakref({"a": 1, "b": 2}) is False
        assert ReferenceManager.should_use_weakref(tuple(range(10))) is False
        assert ReferenceManager.should_use_weakref({1, 2, 3}) is False
        assert ReferenceManager.should_use_weakref(frozenset([1, 2, 3])) is False
        
        # Even large collections don't support weak refs
        large_list = list(range(100))
        large_dict = {i: i for i in range(100)}
        
        assert ReferenceManager.should_use_weakref(large_list) is False
        assert ReferenceManager.should_use_weakref(large_dict) is False
    
    def test_custom_objects_use_weakref(self):
        """Custom objects should use weak references by default."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        assert ReferenceManager.should_use_weakref(obj) is True
    
    def test_pytorch_tensors_use_weakref(self):
        """PyTorch tensors should use weak references."""
        try:
            import torch
            
            tensor = torch.randn(10, 10)
            assert ReferenceManager.should_use_weakref(tensor) is True
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_pytorch_modules_use_weakref(self):
        """PyTorch modules should use weak references."""
        try:
            import torch
            import torch.nn as nn
            
            model = nn.Linear(10, 5)
            assert ReferenceManager.should_use_weakref(model) is True
        except ImportError:
            pytest.skip("PyTorch not available")


class TestCreateReference:
    """Tests for ReferenceManager.create_reference()."""
    
    def test_automatic_primitive_strong_ref(self):
        """Primitives should automatically get strong references."""
        value = 42
        ref, is_weakref = ReferenceManager.create_reference(value)
        
        assert is_weakref is False
        assert ref == value
    
    def test_automatic_object_weak_ref(self):
        """Objects should automatically get weak references."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj)
        
        assert is_weakref is True
        assert isinstance(ref, weakref.ref)
        assert ref() is obj
    
    def test_forced_weak_ref(self):
        """Force weak reference creation."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj, use_weakref=True)
        
        assert is_weakref is True
        assert isinstance(ref, weakref.ref)
    
    def test_forced_strong_ref(self):
        """Force strong reference creation."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj, use_weakref=False)
        
        assert is_weakref is False
        assert ref is obj
    
    def test_weakref_unsupported_fallback(self):
        """Objects that don't support weak refs should fall back to strong refs."""
        # Integers don't support weak references
        value = 42
        ref, is_weakref = ReferenceManager.create_reference(value, use_weakref=True)
        
        assert is_weakref is False
        assert ref == value
    
    def test_pytorch_tensor_weak_ref(self):
        """PyTorch tensors should get weak references."""
        try:
            import torch
            
            tensor = torch.randn(10, 10)
            ref, is_weakref = ReferenceManager.create_reference(tensor)
            
            assert is_weakref is True
            assert isinstance(ref, weakref.ref)
            assert ref() is tensor
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_collection_fallback_to_strong_ref(self):
        """Collections should fall back to strong references (they don't support weak refs)."""
        large_list = list(range(100))
        ref, is_weakref = ReferenceManager.create_reference(large_list)
        
        # Should fall back to strong ref since lists don't support weak refs
        assert is_weakref is False
        assert ref is large_list


class TestDereference:
    """Tests for ReferenceManager.dereference()."""
    
    def test_dereference_strong_ref(self):
        """Dereferencing a strong reference should return the value."""
        value = 42
        result = ReferenceManager.dereference(value, is_weakref=False)
        
        assert result == value
    
    def test_dereference_weak_ref_alive(self):
        """Dereferencing a live weak reference should return the object."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref = weakref.ref(obj)
        result = ReferenceManager.dereference(ref, is_weakref=True)
        
        assert result is obj
    
    def test_dereference_weak_ref_dead(self):
        """Dereferencing a dead weak reference should raise DeadReferenceError."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref = weakref.ref(obj)
        
        # Delete the object and force garbage collection
        del obj
        gc.collect()
        
        with pytest.raises(DeadReferenceError) as exc_info:
            ReferenceManager.dereference(ref, is_weakref=True)
        
        assert "garbage collected" in str(exc_info.value)
    
    def test_dereference_object_reference(self):
        """Dereferencing should work with custom objects."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj = CustomObject(42)
        ref = weakref.ref(obj)
        result = ReferenceManager.dereference(ref, is_weakref=True)
        
        assert result is obj
        assert result.value == 42


class TestIntegration:
    """Integration tests for complete reference lifecycle."""
    
    def test_create_and_dereference_cycle(self):
        """Test complete create -> dereference cycle."""
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj = CustomObject(42)
        
        # Create reference
        ref, is_weakref = ReferenceManager.create_reference(obj)
        
        # Dereference
        result = ReferenceManager.dereference(ref, is_weakref)
        
        assert result is obj
        assert result.value == 42
    
    def test_memory_cleanup_with_weakref(self):
        """Weak references should allow garbage collection."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj, use_weakref=True)
        
        assert is_weakref is True
        assert ref() is not None
        
        # Delete object and force garbage collection
        del obj
        gc.collect()
        
        # Weak reference should now be dead
        assert ref() is None
    
    def test_memory_retention_with_strong_ref(self):
        """Strong references should prevent garbage collection."""
        class CustomObject:
            pass
        
        obj = CustomObject()
        ref, is_weakref = ReferenceManager.create_reference(obj, use_weakref=False)
        
        assert is_weakref is False
        
        # Delete original reference
        obj_id = id(obj)
        del obj
        gc.collect()
        
        # Strong reference should still hold the object
        assert id(ref) == obj_id
    
    def test_mixed_reference_types(self):
        """Test handling multiple reference types together."""
        class CustomObject:
            pass
        
        # Create various references
        obj = CustomObject()
        primitive = 42
        
        obj_ref, obj_is_weak = ReferenceManager.create_reference(obj)
        prim_ref, prim_is_weak = ReferenceManager.create_reference(primitive)
        
        # Verify types
        assert obj_is_weak is True
        assert prim_is_weak is False
        
        # Dereference both
        obj_result = ReferenceManager.dereference(obj_ref, obj_is_weak)
        prim_result = ReferenceManager.dereference(prim_ref, prim_is_weak)
        
        assert obj_result is obj
        assert prim_result == primitive
    
    def test_pytorch_tensor_lifecycle(self):
        """Test complete lifecycle with PyTorch tensors."""
        try:
            import torch
            
            tensor = torch.randn(10, 10)
            original_data = tensor.clone()
            
            # Create reference
            ref, is_weakref = ReferenceManager.create_reference(tensor)
            assert is_weakref is True
            
            # Dereference and verify
            result = ReferenceManager.dereference(ref, is_weakref)
            assert torch.equal(result, original_data)
            
            # Keep a reference to verify weak ref is working
            assert ref() is tensor
            
            # Note: PyTorch tensors may have internal references that prevent
            # immediate garbage collection. The important thing is that the
            # weak reference mechanism works correctly when the tensor is
            # eventually collected.
        except ImportError:
            pytest.skip("PyTorch not available")
