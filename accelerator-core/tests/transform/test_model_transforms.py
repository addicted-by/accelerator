"""Tests for model transformation utilities."""

import pytest
import torch
import torch.nn as nn

from accelerator.core.transform.model import (
    set_eval_mode,
    set_train_mode,
    fuse_batch_norm,
    unfreeze_parameters,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x


class TestSetEvalMode:
    """Tests for set_eval_mode transform."""
    
    def test_sets_model_to_eval(self):
        """Test that set_eval_mode sets model to evaluation mode."""
        model = SimpleModel()
        model.train()
        assert model.training
        
        set_eval_mode(model)
        assert not model.training
        
    def test_affects_batch_norm(self):
        """Test that eval mode affects batch norm behavior."""
        model = SimpleModel()
        model.train()
        
        set_eval_mode(model)
        assert not model.bn.training


class TestSetTrainMode:
    """Tests for set_train_mode transform."""
    
    def test_sets_model_to_train(self):
        """Test that set_train_mode sets model to training mode."""
        model = SimpleModel()
        model.eval()
        assert not model.training
        
        set_train_mode(model)
        assert model.training
        
    def test_affects_batch_norm(self):
        """Test that train mode affects batch norm behavior."""
        model = SimpleModel()
        model.eval()
        
        set_train_mode(model)
        assert model.bn.training


class TestUnfreezeParameters:
    """Tests for unfreeze_parameters transform."""
    
    def test_unfreezes_all_parameters(self):
        """Test that unfreeze_parameters enables gradients for all params."""
        model = SimpleModel()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Verify they're frozen
        assert all(not p.requires_grad for p in model.parameters())
        
        # Unfreeze
        unfreeze_parameters(model)
        
        # Verify they're unfrozen
        assert all(p.requires_grad for p in model.parameters())
    
    def test_works_on_already_unfrozen(self):
        """Test that unfreeze_parameters works on already unfrozen model."""
        model = SimpleModel()
        
        # Should already be unfrozen by default
        assert all(p.requires_grad for p in model.parameters())
        
        # Should not raise error
        unfreeze_parameters(model)
        assert all(p.requires_grad for p in model.parameters())


class TestFuseBatchNorm:
    """Tests for fuse_batch_norm transform."""
    
    def test_fuses_conv_bn_pair(self):
        """Test that fuse_batch_norm fuses conv and bn layers."""
        model = SimpleModel()
        model.eval()
        
        # Check initial state
        assert isinstance(model.bn, nn.BatchNorm2d)
        
        # Fuse
        fuse_batch_norm(model)
        
        # BatchNorm should be replaced with Identity
        assert isinstance(model.bn, nn.Identity)
    
    def test_requires_eval_mode(self):
        """Test that fusion requires model to be in eval mode."""
        model = SimpleModel()
        model.train()
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            fuse_batch_norm(model)
    
    def test_preserves_output(self):
        """Test that fusion preserves model output."""
        model = SimpleModel()
        model.eval()
        
        # Get output before fusion
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            output_before = model(x)
        
        # Fuse
        fuse_batch_norm(model)
        
        # Get output after fusion
        with torch.no_grad():
            output_after = model(x)
        
        # Outputs should be close (allowing for numerical precision)
        assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-6)
    
    def test_conv_bias_created_if_none(self):
        """Test that conv bias is created if it doesn't exist."""
        model = SimpleModel()
        model.eval()
        
        # Remove conv bias
        model.conv.bias = None
        
        # Fuse
        fuse_batch_norm(model)
        
        # Conv should now have bias
        assert model.conv.bias is not None


class TestTransformComposition:
    """Tests for composing multiple transforms."""
    
    def test_can_chain_transforms(self):
        """Test that transforms can be chained together."""
        model = SimpleModel()
        model.train()
        
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply transforms in sequence
        set_eval_mode(model)
        fuse_batch_norm(model)
        unfreeze_parameters(model)
        
        # Verify all transforms were applied
        assert not model.training
        assert isinstance(model.bn, nn.Identity)
        assert all(p.requires_grad for p in model.parameters())
    
    def test_transform_list_application(self):
        """Test applying transforms from a list."""
        model = SimpleModel()
        model.train()
        
        transforms = [set_eval_mode, fuse_batch_norm]
        
        for transform in transforms:
            transform(model)
        
        assert not model.training
        assert isinstance(model.bn, nn.Identity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
