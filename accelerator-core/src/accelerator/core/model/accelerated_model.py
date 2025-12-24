"""
Accelerated model wrapper for PyTorch models.

This module provides the AcceleratedModel class which wraps PyTorch models
with additional acceleration capabilities, gamma correction, pixel shuffling,
and custom processing functions. The wrapper maintains full compatibility with
PyTorch's nn.Module interface while adding advanced features for model
optimization and specialized processing pipelines.

Key Features:
    - Gamma correction for color space transformations
    - Pixel shuffle operations for upsampling
    - Custom collate and separate functions for data processing
    - Acceleration operation support (pruning, quantization, etc.)
    - Regularization term management
    - Comprehensive model introspection and debugging

The module is designed to be used as a drop-in replacement for standard
PyTorch models while providing additional functionality for advanced
computer vision and machine learning workflows.
"""

import importlib
from dataclasses import dataclass, field
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from accelerator.core.acceleration import AccelerationOperationBase, acceleration_registry
from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.default_config import _DefaultConfig
from accelerator.utilities.hydra_utils import instantiate
from accelerator.utilities.logging import get_logger
from accelerator.utilities.typings import ConfigType

from .gamma import Gamma, GammaDefaultConfig

if TYPE_CHECKING:
    from accelerator.core.context import Context


log = get_logger(__name__)


@dataclass
class ModelDefaultConfig(_DefaultConfig):
    """
    Configuration structure for AcceleratedModel initialization.

    This dataclass defines the default configuration parameters used to
    initialize and configure an AcceleratedModel instance. It extends
    the base _DefaultConfig to provide model-specific settings.

    Attributes:
        gamma (Optional[ConfigType]): Configuration for gamma correction
            operations. Defaults to GammaDefaultConfig. Set to None to
            disable gamma correction entirely.
        d2s (int): Pixel shuffle upscale factor for depth-to-space operations.
            Set to 0 to disable pixel shuffle. Values > 0 enable upsampling
            by the specified factor.
        collate_fn (Optional[Callable]): Custom function for collating input
            data before model processing. If None, uses default collation.
        separate_fn (Optional[Callable]): Custom function for separating
            model outputs into structured format. If None, uses default
            separation logic.

    Example:
        >>> config = ModelDefaultConfig(
        ...     gamma={"use_gamma": True, "gamma_value": 2.2},
        ...     d2s=2,
        ...     collate_fn=None,
        ...     separate_fn=None
        ... )
    """

    gamma: Optional[ConfigType] = field(default_factory=GammaDefaultConfig)
    d2s: int = 0

    collate_fn: Optional[Callable] = None
    separate_fn: Optional[Callable] = None


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Beta")
class AcceleratedModel(torch.nn.Module):
    """
    Advanced PyTorch model wrapper with acceleration and processing capabilities.

    AcceleratedModel extends torch.nn.Module to provide a comprehensive wrapper
    around standard PyTorch models with additional features for computer vision
    and machine learning workflows. It maintains full compatibility with PyTorch's
    module interface while adding specialized functionality.

    Key Capabilities:
        - **Gamma Correction**: Automatic color space transformations with
          configurable gamma values for input and output processing
        - **Pixel Shuffle Operations**: Built-in depth-to-space upsampling
          for super-resolution and image enhancement tasks
        - **Custom Processing Pipeline**: Configurable collate and separate
          functions for flexible data preprocessing and postprocessing
        - **Acceleration Support**: Integration with various model acceleration
          techniques including pruning, quantization, and optimization
        - **Regularization Management**: Built-in regularization term tracking
          and accumulation for advanced training scenarios
        - **Model Introspection**: Comprehensive debugging and analysis tools
          with detailed model representation

    The class is designed to be a drop-in replacement for standard PyTorch
    models while providing additional functionality. It handles complex
    processing pipelines transparently and maintains gradient flow throughout
    all operations.

    Architecture:
        The model follows a pipeline architecture:
        Input → Collate → Gamma Forward → Core Model → Separate →
        Pixel Shuffle → Gamma Inverse → Output

    Thread Safety:
        This class is not thread-safe. Use separate instances for concurrent
        processing or implement appropriate synchronization.

    Example:
        >>> import torch
        >>> from accelerator.runtime.model import AcceleratedModel
        >>>
        >>> # Wrap existing model
        >>> base_model = torch.nn.Linear(10, 5)
        >>> config = {"gamma": {"use_gamma": True}, "d2s": 2}
        >>> accelerated = AcceleratedModel(base_model, config)
        >>>
        >>> # Use like any PyTorch model
        >>> input_tensor = torch.randn(1, 10)
        >>> output = accelerated(input_tensor)

    Attributes:
        model_config (ModelDefaultConfig): Configuration object containing
            all model settings and parameters
        model_core (torch.nn.Module): The wrapped PyTorch model that performs
            the core computation
        gamma (Optional[Gamma]): Gamma correction component for color space
            transformations, None if disabled
        d2s (Optional[torch.nn.PixelShuffle]): Pixel shuffle component for
            upsampling operations, None if disabled
        collate_fn (Callable): Function used to collate input data before
            processing
        separate_fn (Callable): Function used to separate and structure
            model outputs
        regularization_term (Optional[torch.Tensor]): Accumulated regularization
            value, None if no regularization is active
    """

    _regularization_term: torch.Tensor = torch.tensor([0.0])

    def __init__(self, model: torch.nn.Module, cfg: Optional[dict[str, Any]] = None):
        """Initialize an accelerated model wrapper.

        Args:
            model: The core PyTorch model to wrap.
            cfg: Configuration containing model settings.
        """
        super().__init__()

        # Core configuration and model
        self.model_config: ModelDefaultConfig = ModelDefaultConfig.create(cfg)
        self.model_core: torch.nn.Module = model

        # Component attributes (initialized to None, set in _init_components)
        self.gamma: Optional[Gamma] = None
        self.d2s: Optional[torch.nn.PixelShuffle] = None
        self.collate_fn: Callable = None
        self.separate_fn: Callable = None

        # Acceleration tracking
        self._acceleration_operations: list[AccelerationOperationBase] = []

        self._init_components()

    def _init_components(self):
        """Initialize all required components for model:

        Components:
            - gamma: See info in `accelerator.runtime.model.gamma` package
            - d2s:
            - collate_fn:
            - separate_fn:
        """
        self._init_gamma()
        self._init_pixel_shuffle()
        self._init_processing_functions()
        # self._register_reg_term()

    def _init_gamma(self):
        """Initialize gamma correction component."""
        self.gamma = Gamma(self.model_config["gamma"]) if self.model_config["gamma"]["use_gamma"] else None

    def _init_pixel_shuffle(self):
        """Initialize pixel shuffle component for upsampling."""
        self.d2s = torch.nn.PixelShuffle(self.model_config["d2s"]) if self.model_config["d2s"] > 0 else None

    def _init_processing_functions(self):
        """Initialize collate and separate functions for data processing."""
        # Collate function
        collate_fn_name = self.model_config["collate_fn"]
        if collate_fn_name is None:
            log.warning("Model collate function is set to `None`. Using default")
            self.collate_fn = self._default_collate_fn
        else:
            self.collate_fn = self._get_collate_fn(collate_fn_name)

        # Separate function
        separate_fn_name = self.model_config["separate_fn"]
        if separate_fn_name is None:
            log.warning("Model outputs separate function is set to `None`. Using default")
            self.separate_fn = lambda x: {"net_result": x}
        elif separate_fn_name == "empty":
            log.info("Using pre-default `separate_fn`")
            self.separate_fn = lambda x: x
        else:
            self.separate_fn = self._get_collate_fn(separate_fn_name)

    @staticmethod
    def _default_collate_fn(*args, **kwargs):
        return args, kwargs

    def _get_collate_fn(self, collate_fn_name: str) -> Callable:
        """Retrieve the collate function from a fully qualified name.

        Args:
            collate_fn_name: String like 'module.submodule.function'.

        Returns:
            Callable collate function.

        Raises:
            NotImplementedError: If the function cannot be imported or found.
        """
        try:
            package, name = collate_fn_name.rsplit(".", 1)
            module = importlib.import_module(package)
            collate_fn = getattr(module, name, None)
            if collate_fn is None:
                raise AttributeError(f"Function '{name}' not found in '{package}'")
            return collate_fn
        except (ImportError, AttributeError) as e:
            raise NotImplementedError(
                dedent(
                    f"""
                    Failed to load collate function '{collate_fn_name}': {str(e)}.
                    Ensure it’s implemented and accessible, or adjust 'model.collate_fn' in config.
                    """
                )
            ) from e

    def add_regularization(self, value: torch.Tensor):
        """Add a regularization value to the accumulated regularization term."""
        self._regularization_term += value  # pylint: disable=no-member

    def reset_regularization_term(self, value=0.0):
        """Reset the regularization term to a specified value."""
        self._regularization_term.data = torch.full_like(  # pylint: disable=no-member
            self._regularization_term, value  # pylint: disable=no-member
        )

    @property
    def regularization_term(self) -> Optional[torch.Tensor]:
        if torch.allclose(
            self._regularization_term,  # pylint: disable=no-member
            torch.zeros_like(self._regularization_term),  # pylint: disable=no-member
        ):
            return None

        return self._regularization_term  # pylint: disable=no-member

    @staticmethod
    def instantiate_model(model_config: ConfigType):
        if "model_core" not in model_config:
            raise ValueError("`model_config` must contain `model_core` section")

        model = instantiate(model_config["model_core"])
        return AcceleratedModel(model, model_config)

    @property
    def acceleration__meta_data__(self):
        return self._acceleration_operations

    def accelerate(self, context: "Context", acceleration_config: Optional[ConfigType] = None):
        if acceleration_config is None:
            return

        self._apply_acceleration(context, acceleration_config)
        self._calibrate(context)

    def update_acceleration_meta_data(self, acceleration_operation: AccelerationOperationBase):
        self._acceleration_operations.append(acceleration_operation)

    def _apply_acceleration(self, context: "Context", acceleration_config: Optional[ConfigType] = None) -> None:
        """Apply acceleration techniques specified in the config."""
        if acceleration_config is None:
            return

        accel_order = acceleration_config.get("order", [])
        for acceleration in accel_order:
            log.info(f"Applying acceleration {acceleration}")
            accel_cfg = acceleration_config.get(acceleration, {})
            accel_type = accel_cfg.get("operation_type")
            accel_name = accel_cfg.get("operation")

            accel_operation = acceleration_registry.get_acceleration(accel_type, accel_name)(
                acceleration_config=accel_cfg
            )
            if accel_operation in self._acceleration_operations:
                log.warning("You have already applied it. Skip...")
            else:
                accel_operation.apply(context)
                self._acceleration_operations.append(accel_operation)

    def _calibrate(self, context: "Context"):
        for acceleration in self._acceleration_operations:
            acceleration.calibrate(context)

    def _reapply_accelerations(self, acceleration_metadata: dict[str, Any]) -> "AcceleratedModel":
        """Reapply acceleration techniques from saved metadata.

        Args:
            acceleration_metadata: Dictionary of acceleration names and their arguments.

        Returns:
            Self for method chaining.

        Raises:
            KeyError: If an acceleration operation is not registered.
        """
        for accel_type, accel_state_dict in acceleration_metadata.items():
            accel_cfg = accel_state_dict["config"]
            accel_name = accel_cfg["operation"]

            accel_operation: AccelerationOperationBase = acceleration_registry.get_acceleration(
                accel_type, accel_name
            )()
            accel_operation.load_state(accel_state_dict)
            if accel_operation in self._acceleration_operations:
                log.warning("You have already applied it. Skip...")
            else:
                accel_operation.reapply(self)

    def run_model(self, *args, **kwargs):
        req_grad = kwargs.pop("req_grad", True)

        with torch.set_grad_enabled(req_grad):
            output = self.forward(*args, **kwargs)

        return output

    def forward(self, *args, **kwargs):
        self.reset_regularization_term()

        inputs, additional = self.collate_fn(*args, **kwargs)

        print(inputs[0].shape)

        if self.gamma:
            x, extra, _ = self.gamma._gamma_forward(*inputs, ref_frame_index=0)  # pylint: disable=protected-access
            inputs = (x, extra)

        output = self.model_core(*inputs, **additional)

        outputs = self.separate_fn(output)

        if self.d2s:
            outputs["net_result"] = self.d2s(outputs["net_result"])

        if self.gamma:
            outputs["net_result"] = self.gamma._gamma_inverse(outputs["net_result"])  # pylint: disable=protected-access

        outputs["additional"] = additional
        return outputs

    def _get_parameter_info(self) -> list[str]:
        """Calculate parameter statistics and return formatted lines.

        Returns:
            List of strings containing parameter information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        mode = "training" if self.training else "evaluation"

        return [
            f"  Parameters: {total_params:,} total ({trainable_params:,} trainable, {frozen_params:,} frozen)",
            f"  Mode: {mode}",
        ]

    def _get_component_info(self) -> list[str]:
        """Generate component status information and return formatted lines.

        Returns:
            List of strings containing component configuration and status.
        """
        lines = ["  Configuration:"]
        lines.append(f"    Gamma: {'enabled' if self.gamma else 'disabled'}")
        lines.append(f"    Pixel Shuffle (d2s): {'enabled' if self.d2s else 'disabled'}")

        collate_fn_name = getattr(self.collate_fn, "__name__", "custom_function")
        separate_fn_name = getattr(self.separate_fn, "__name__", "lambda")
        lines.append(f"    Collate Function: {collate_fn_name}")
        lines.append(f"    Separate Function: {separate_fn_name}")

        # Active components section
        components = []
        if self.gamma:
            gamma_info = f"Gamma(enabled={self.model_config['gamma'].get('use_gamma', False)})"
            components.append(gamma_info)

        if self.d2s:
            d2s_info = f"PixelShuffle(upscale_factor={self.model_config.get('d2s', 0)})"
            components.append(d2s_info)

        if components:
            lines.append("  Active Components:")
            for component in components:
                lines.append(f"    - {component}")

        return lines

    def _get_acceleration_info(self) -> list[str]:
        """Format acceleration operation details and return formatted lines.

        Returns:
            List of strings containing acceleration operation information.
        """
        if self._acceleration_operations:
            lines = ["  Applied Accelerations:"]
            for i, accel_op in enumerate(self._acceleration_operations, 1):
                accel_type = type(accel_op).__name__
                op_name = getattr(accel_op, "operation_name", "unknown")
                lines.append(f"    {i}. {accel_type} ({op_name})")
        else:
            lines = ["  Applied Accelerations: none"]

        return lines

    def __repr__(self) -> str:
        """Return a detailed string representation extending torch.nn.Module's repr.

        Shows the original PyTorch module structure + AcceleratedModel-specific information.
        """
        original_repr = super().__repr__()
        lines = [original_repr, "", "AcceleratedModel Extensions:"]

        # Add parameter information
        lines.extend(self._get_parameter_info())

        # Add component information
        lines.extend(self._get_component_info())

        # Add acceleration information
        lines.extend(self._get_acceleration_info())

        return "\n".join(lines)
