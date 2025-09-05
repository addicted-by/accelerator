"""Utility functions for CIFAR-10 ResNet-18 pipeline example.

This module provides utility functions for converting torchvision CIFAR-10 format
to accelerator framework format, along with common data transformations and
configuration helpers.
"""

import torch
from typing import List, Dict, Any, Tuple
import torchvision.transforms as transforms


def cifar10_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """Convert torchvision CIFAR-10 format to accelerator format.
    
    Converts standard torchvision dataset output format (image, label) to the
    accelerator framework's expected format: (inputs, targets, additional).
    
    Args:
        batch: List of (image, label) tuples from torchvision CIFAR-10 dataset
        
    Returns:
        Tuple containing:
        - inputs: List of input tensors [batch_images]
        - targets: Dict with target information {"class": class_labels}
        - additional: Dict with metadata {"batch_size": batch_size}
    """
    images, labels = zip(*batch)
    
    # Stack images into a single tensor
    batch_images = torch.stack(images)
    
    # Convert labels to tensor
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    # Format according to accelerator framework expectations
    inputs = [batch_images]  # List of input tensors
    targets = {"class": batch_labels}  # Dict with target information
    additional = {"batch_size": len(batch)}  # Dict with metadata
    
    return inputs, targets, additional


def get_cifar10_transforms(train: bool = True) -> transforms.Compose:
    """Get standard CIFAR-10 data transformations.
    
    Args:
        train: If True, returns training transforms with augmentation.
               If False, returns validation transforms without augmentation.
               
    Returns:
        Composed transforms for CIFAR-10 dataset
    """
    if train:
        # Training transforms with data augmentation
        transform_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means (commonly used for CIFAR-10)
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ]
    else:
        # Validation transforms without augmentation
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def get_cifar10_dataloader_config(batch_size: int = 128, num_workers: int = 4) -> Dict[str, Any]:
    """Get common dataloader configuration for CIFAR-10.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary with dataloader configuration
    """
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "collate_fn": {
            "_target_": "accelerator.examples.cifar10_resnet18.utils.cifar10_collate_fn"
        }
    }


def get_cifar10_dataset_config(root: str = "./data", download: bool = True) -> Dict[str, Any]:
    """Get common dataset configuration for CIFAR-10.
    
    Args:
        root: Root directory for dataset storage
        download: Whether to download dataset if not present
        
    Returns:
        Dictionary with dataset configuration for both train and validation
    """
    return {
        "train": {
            "_target_": "torchvision.datasets.CIFAR10",
            "root": root,
            "train": True,
            "download": download,
            "transform": {
                "_target_": "accelerator.examples.cifar10_resnet18.utils.get_cifar10_transforms",
                "train": True
            }
        },
        "val": {
            "_target_": "torchvision.datasets.CIFAR10",
            "root": root,
            "train": False,
            "download": download,
            "transform": {
                "_target_": "accelerator.examples.cifar10_resnet18.utils.get_cifar10_transforms",
                "train": False
            }
        }
    }


def get_resnet18_config(num_classes: int = 10, pretrained: bool = False) -> Dict[str, Any]:
    """Get ResNet-18 model configuration for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10)
        pretrained: Whether to use pretrained weights
        
    Returns:
        Dictionary with model configuration
    """
    return {
        "model_core": {
            "_target_": "torchvision.models.resnet18",
            "num_classes": num_classes,
            "pretrained": pretrained
        }
    }


def get_training_components_config() -> Dict[str, Any]:
    """Get common training components configuration for CIFAR-10 training.
    
    Returns:
        Dictionary with optimizer, loss, and scheduler configurations
    """
    return {
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": 0.001,
            "weight_decay": 1e-4
        },
        "loss": {
            "_target_": "torch.nn.CrossEntropyLoss"
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 30,
            "gamma": 0.1
        }
    }


def validate_cifar10_batch(inputs: List[torch.Tensor], targets: Dict[str, torch.Tensor], additional: Dict[str, Any]) -> bool:
    """Validate that a batch conforms to expected CIFAR-10 format.
    
    Args:
        inputs: List of input tensors
        targets: Dict with target information
        additional: Dict with metadata
        
    Returns:
        True if batch format is valid, False otherwise
    """
    try:
        # Check inputs format
        if not isinstance(inputs, list) or len(inputs) != 1:
            return False
            
        batch_images = inputs[0]
        if not isinstance(batch_images, torch.Tensor):
            return False
            
        # Check image dimensions: [batch_size, 3, 32, 32] for CIFAR-10
        if len(batch_images.shape) != 4 or batch_images.shape[1:] != (3, 32, 32):
            return False
            
        # Check targets format
        if not isinstance(targets, dict) or "class" not in targets:
            return False
            
        batch_labels = targets["class"]
        if not isinstance(batch_labels, torch.Tensor):
            return False
            
        # Check label dimensions and range
        if len(batch_labels.shape) != 1 or batch_labels.shape[0] != batch_images.shape[0]:
            return False
            
        if torch.any(batch_labels < 0) or torch.any(batch_labels >= 10):
            return False
            
        # Check additional metadata
        if not isinstance(additional, dict) or "batch_size" not in additional:
            return False
            
        if additional["batch_size"] != batch_images.shape[0]:
            return False
            
        return True
        
    except Exception:
        return False