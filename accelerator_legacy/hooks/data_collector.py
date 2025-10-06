"""
DataCollector class for managing hook data storage and retrieval.
"""

import gc
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch

from .types import HookData, HooksConfig


class DataNotAvailableError(Exception):
    """Raised when requested hook data doesn't exist."""


class MemoryLimitExceededError(Exception):
    """Raised when data storage exceeds configured limits."""


class DataCollector:
    """
    Manages storage and retrieval of hook data with memory management.

    Supports different data modes:
    - "replace": New data replaces existing data for the same module
    - "accumulate": New data is added to existing data (for tensors)
    - "list": New data is appended to a list of historical data
    """

    def __init__(self, config: HooksConfig):
        """
        Initialize the DataCollector.

        Args:
            config: Configuration for data collection behavior
        """
        self.config = config
        self._activations: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}
        self._gradients: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}
        self._activation_metadata: Dict[str, Union[HookData, List[HookData]]] = {}
        self._gradient_metadata: Dict[str, Union[HookData, List[HookData]]] = {}

    def store_activation(self, module_name: str, data: torch.Tensor) -> None:
        """
        Store activation data for a module.

        Args:
            module_name: Name of the module that produced the activation
            data: The activation tensor to store

        Raises:
            MemoryLimitExceededError: If storing would exceed memory limits
        """
        # Move data to configured device
        processed_data = self._process_tensor_placement(data)

        # Check memory limit after processing but before storing
        self._check_memory_limit_with_new_data(processed_data)

        # Create metadata
        hook_data = HookData(
            module_name=module_name,
            data_type="activation",
            tensor_data=processed_data,
            shape=tuple(processed_data.shape),
            device=str(processed_data.device),
            dtype=processed_data.dtype,
            timestamp=datetime.now()
        )

        # Store based on data mode
        if self.config.data_mode == "replace":
            self._activations[module_name] = processed_data
            self._activation_metadata[module_name] = hook_data
        elif self.config.data_mode == "accumulate":
            if module_name in self._activations:
                self._activations[module_name] = self._activations[module_name] + processed_data
                # Update metadata with accumulated tensor
                self._activation_metadata[module_name].tensor_data = self._activations[module_name]
                self._activation_metadata[module_name].shape = tuple(self._activations[module_name].shape)
                self._activation_metadata[module_name].timestamp = datetime.now()
            else:
                self._activations[module_name] = processed_data
                self._activation_metadata[module_name] = hook_data
        elif self.config.data_mode == "list":
            if module_name not in self._activations:
                self._activations[module_name] = []
                self._activation_metadata[module_name] = []
            self._activations[module_name].append(processed_data)
            self._activation_metadata[module_name].append(hook_data)

    def store_gradient(self, module_name: str, data: torch.Tensor) -> None:
        """
        Store gradient data for a module.

        Args:
            module_name: Name of the module that produced the gradient
            data: The gradient tensor to store

        Raises:
            MemoryLimitExceededError: If storing would exceed memory limits
        """
        # Move data to configured device
        processed_data = self._process_tensor_placement(data)

        # Check memory limit after processing but before storing
        self._check_memory_limit_with_new_data(processed_data)

        # Create metadata
        hook_data = HookData(
            module_name=module_name,
            data_type="gradient",
            tensor_data=processed_data,
            shape=tuple(processed_data.shape),
            device=str(processed_data.device),
            dtype=processed_data.dtype,
            timestamp=datetime.now()
        )

        # Store based on data mode
        if self.config.data_mode == "replace":
            self._gradients[module_name] = processed_data
            self._gradient_metadata[module_name] = hook_data
        elif self.config.data_mode == "accumulate":
            if module_name in self._gradients:
                self._gradients[module_name] = self._gradients[module_name] + processed_data
                # Update metadata with accumulated tensor
                self._gradient_metadata[module_name].tensor_data = self._gradients[module_name]
                self._gradient_metadata[module_name].shape = tuple(self._gradients[module_name].shape)
                self._gradient_metadata[module_name].timestamp = datetime.now()
            else:
                self._gradients[module_name] = processed_data
                self._gradient_metadata[module_name] = hook_data
        elif self.config.data_mode == "list":
            if module_name not in self._gradients:
                self._gradients[module_name] = []
                self._gradient_metadata[module_name] = []
            self._gradients[module_name].append(processed_data)
            self._gradient_metadata[module_name].append(hook_data)

    def get_activation(self, module_name: str) -> torch.Tensor:
        """
        Retrieve activation data for a module.

        Args:
            module_name: Name of the module to get activation data for

        Returns:
            The activation tensor for the module

        Raises:
            DataNotAvailableError: If no activation data exists for the module
        """
        if module_name not in self._activations:
            raise DataNotAvailableError(
                f"No activation data available for module '{module_name}'. "
                f"Available modules: {list(self._activations.keys())}"
            )

        data = self._activations[module_name]
        if self.config.data_mode == "list":
            if not data:
                raise DataNotAvailableError(
                    f"No activation data available for module '{module_name}' (empty list)"
                )
            # Return the most recent activation
            return data[-1]
        return data

    def get_gradient(self, module_name: str) -> torch.Tensor:
        """
        Retrieve gradient data for a module.

        Args:
            module_name: Name of the module to get gradient data for

        Returns:
            The gradient tensor for the module

        Raises:
            DataNotAvailableError: If no gradient data exists for the module
        """
        if module_name not in self._gradients:
            raise DataNotAvailableError(
                f"No gradient data available for module '{module_name}'. "
                f"Available modules: {list(self._gradients.keys())}"
            )

        data = self._gradients[module_name]
        if self.config.data_mode == "list":
            if not data:
                raise DataNotAvailableError(
                    f"No gradient data available for module '{module_name}' (empty list)"
                )
            # Return the most recent gradient
            return data[-1]
        return data

    def get_activation_history(self, module_name: str) -> List[torch.Tensor]:
        """
        Retrieve all activation data for a module (only works in list mode).

        Args:
            module_name: Name of the module to get activation history for

        Returns:
            List of activation tensors for the module

        Raises:
            DataNotAvailableError: If no activation data exists or not in list mode
        """
        if self.config.data_mode != "list":
            raise DataNotAvailableError(
                f"Activation history only available in 'list' data mode, "
                f"current mode is '{self.config.data_mode}'"
            )

        if module_name not in self._activations:
            raise DataNotAvailableError(
                f"No activation data available for module '{module_name}'"
            )

        return self._activations[module_name].copy()

    def get_gradient_history(self, module_name: str) -> List[torch.Tensor]:
        """
        Retrieve all gradient data for a module (only works in list mode).

        Args:
            module_name: Name of the module to get gradient history for

        Returns:
            List of gradient tensors for the module

        Raises:
            DataNotAvailableError: If no gradient data exists or not in list mode
        """
        if self.config.data_mode != "list":
            raise DataNotAvailableError(
                f"Gradient history only available in 'list' data mode, "
                f"current mode is '{self.config.data_mode}'"
            )

        if module_name not in self._gradients:
            raise DataNotAvailableError(
                f"No gradient data available for module '{module_name}'"
            )

        return self._gradients[module_name].copy()

    def clear_all_data(self) -> None:
        """Clear all stored activation and gradient data."""
        self._activations.clear()
        self._gradients.clear()
        self._activation_metadata.clear()
        self._gradient_metadata.clear()

        # Force garbage collection to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def clear_activations(self, module_name: Optional[str] = None) -> None:
        """
        Clear activation data for a specific module or all modules.

        Args:
            module_name: Name of module to clear data for. If None, clears all.
        """
        if module_name is None:
            self._activations.clear()
            self._activation_metadata.clear()
        else:
            self._activations.pop(module_name, None)
            self._activation_metadata.pop(module_name, None)

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def clear_gradients(self, module_name: Optional[str] = None) -> None:
        """
        Clear gradient data for a specific module or all modules.

        Args:
            module_name: Name of module to clear data for. If None, clears all.
        """
        if module_name is None:
            self._gradients.clear()
            self._gradient_metadata.clear()
        else:
            self._gradients.pop(module_name, None)
            self._gradient_metadata.pop(module_name, None)

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all stored data.

        Returns:
            Dictionary containing data summary information
        """
        summary = {
            "data_mode": self.config.data_mode,
            "total_modules_with_activations": len(self._activations),
            "total_modules_with_gradients": len(self._gradients),
            "activation_modules": list(self._activations.keys()),
            "gradient_modules": list(self._gradients.keys()),
            "memory_usage_mb": self._estimate_memory_usage(),
        }

        # Add detailed info for each module
        activation_details = {}
        for module_name, data in self._activations.items():
            if self.config.data_mode == "list":
                activation_details[module_name] = {
                    "count": len(data),
                    "shapes": [tuple(t.shape) for t in data],
                    "devices": [str(t.device) for t in data],
                    "dtypes": [str(t.dtype) for t in data]
                }
            else:
                activation_details[module_name] = {
                    "shape": tuple(data.shape),
                    "device": str(data.device),
                    "dtype": str(data.dtype)
                }
        summary["activation_details"] = activation_details

        gradient_details = {}
        for module_name, data in self._gradients.items():
            if self.config.data_mode == "list":
                gradient_details[module_name] = {
                    "count": len(data),
                    "shapes": [tuple(t.shape) for t in data],
                    "devices": [str(t.device) for t in data],
                    "dtypes": [str(t.dtype) for t in data]
                }
            else:
                gradient_details[module_name] = {
                    "shape": tuple(data.shape),
                    "device": str(data.device),
                    "dtype": str(data.dtype)
                }
        summary["gradient_details"] = gradient_details

        return summary

    def has_activation_data(self, module_name: str) -> bool:
        """
        Check if activation data exists for a module.

        Args:
            module_name: Name of the module to check

        Returns:
            True if activation data exists for the module
        """
        return module_name in self._activations

    def has_gradient_data(self, module_name: str) -> bool:
        """
        Check if gradient data exists for a module.

        Args:
            module_name: Name of the module to check

        Returns:
            True if gradient data exists for the module
        """
        return module_name in self._gradients

    def get_module_names_with_activations(self) -> List[str]:
        """
        Get list of module names that have activation data.

        Returns:
            List of module names with stored activation data
        """
        return list(self._activations.keys())

    def get_module_names_with_gradients(self) -> List[str]:
        """
        Get list of module names that have gradient data.

        Returns:
            List of module names with stored gradient data
        """
        return list(self._gradients.keys())

    def _process_tensor_placement(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor placement according to configuration.

        Args:
            tensor: Input tensor to process

        Returns:
            Tensor moved to the configured device
        """
        if self.config.device_placement == "cpu":
            return tensor.detach().cpu()
        if self.config.device_placement == "cuda":
            if torch.cuda.is_available():
                return tensor.detach().cuda()
            # Fall back to CPU if CUDA not available
            return tensor.detach().cpu()
        # "same" - keep on original device
        return tensor.detach()

    def _check_memory_limit_with_new_data(self, new_tensor: torch.Tensor) -> None:
        """
        Check if storing new data would exceed memory limits.

        Args:
            new_tensor: The tensor that would be added

        Raises:
            MemoryLimitExceededError: If memory limit would be exceeded
        """
        if self.config.max_memory_mb is None:
            return

        current_usage = self._estimate_memory_usage()
        new_data_mb = (new_tensor.numel() * new_tensor.element_size()) / (1024 * 1024)
        total_usage = current_usage + new_data_mb

        if total_usage > self.config.max_memory_mb:
            raise MemoryLimitExceededError(
                f"Memory usage ({total_usage:.1f} MB) would exceed limit "
                f"({self.config.max_memory_mb} MB)"
            )

    def _estimate_memory_usage(self) -> float:
        """
        Estimate current memory usage of stored tensors in MB.

        Returns:
            Estimated memory usage in megabytes
        """
        total_bytes = 0

        # Calculate activation memory
        for data in self._activations.values():
            if self.config.data_mode == "list":
                for tensor in data:
                    total_bytes += tensor.numel() * tensor.element_size()
            else:
                total_bytes += data.numel() * data.element_size()

        # Calculate gradient memory
        for data in self._gradients.values():
            if self.config.data_mode == "list":
                for tensor in data:
                    total_bytes += tensor.numel() * tensor.element_size()
            else:
                total_bytes += data.numel() * data.element_size()

        return total_bytes / (1024 * 1024)  # Convert to MB

    def __len__(self) -> int:
        """
        Get total number of modules with stored data.

        Returns:
            Number of unique modules with either activation or gradient data
        """
        all_modules = set(self._activations.keys()) | set(self._gradients.keys())
        return len(all_modules)

    def __contains__(self, module_name: str) -> bool:
        """
        Check if module has any stored data.

        Args:
            module_name: Name of the module to check

        Returns:
            True if module has activation or gradient data
        """
        return (module_name in self._activations or
                module_name in self._gradients)

    def __repr__(self) -> str:
        """String representation of the DataCollector."""
        return (f"DataCollector(data_mode='{self.config.data_mode}', "
                f"modules={len(self)}, "
                f"memory_usage={self._estimate_memory_usage():.1f}MB)")