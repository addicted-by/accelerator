from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from omegaconf import OmegaConf

from .checkpoint_handler import CheckpointHandler
from .meta_handler import MetadataHandler
from .operation.registry import OperationType, registry

if TYPE_CHECKING:
    from accelerator.core.context import Context

from accelerator.core.model.accelerated_model import AcceleratedModel
from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.default_config import _DefaultConfig, dataclass
from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.logging import get_logger
from accelerator.utilities.model_utils import unwrap_model
from accelerator.utilities.string import render_filename
from accelerator.utilities.typings import ConfigType, ModelProtocol, PathType

logger = get_logger(__name__)


@dataclass
class CheckpointManagerDefaultConfig(_DefaultConfig):
    monitor: str = "loss"
    mode: str = "min"
    filename_template: str = "epoch_{epoch:05d}.pth"  #'checkpoint_epoch{epoch:05d}_{monitor}_{metric:.4f}.pth'

    state_dict_key: str = "model_state"
    optimizer_state_dict_key: str = "optimizer_state"
    load_optimizer_state: bool = False
    load_strategy: str = "accelerated"
    save_strategy: str = "accelerated"
    strict_validation: bool = True
    strict_loading: bool = True

    pre_load_ops: Optional[list[str]] = None
    ckpt_transforms: Optional[list[str]] = None
    post_load_ops: Optional[list[str]] = None

    ignore_model_keys: Optional[list[str]] = None
    ignore_acceleration_keys: Optional[list[str]] = None

    weights_only: bool = False  # checkpoint should contain only tensors and pickable structures, not objects


@dataclass
class CheckpointManagerRawDefaultConfig(CheckpointManagerDefaultConfig):
    load_strategy: str = "raw"
    strict_validation: bool = False


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""

    pass


class CheckpointLoadError(CheckpointError):
    """Raised when loading a checkpoint fails."""

    pass


class CheckpointSaveError(CheckpointError):
    """Raised when saving a checkpoint fails."""

    pass


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint validation fails."""

    pass


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Beta")
class CheckpointManager:
    """Manages saving and loading of model checkpoints with acceleration support."""

    def __init__(self, cfg: Optional[ConfigType] = None):
        """Initialize the checkpoint manager.

        Args:
            cfg: Configuration object with 'paths', 'checkpoint', and 'acceleration' sections.
        """
        self.setup_checkpoint_config(cfg)

        self.checkpoint_handler = CheckpointHandler(self.checkpoint_cfg)
        self.metadata_handler = MetadataHandler(self.checkpoint_cfg)

    def setup_checkpoint_config(
        self, updates: ConfigType, default_config: _DefaultConfig = CheckpointManagerDefaultConfig
    ):
        self.checkpoint_cfg = default_config.create(updates)

    @distributed_state.on_main_process
    def save(
        self,
        model_: Union[torch.nn.Module, AcceleratedModel],
        save_dir: PathType,
        metrics: Optional[dict[str, float]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[ConfigType] = None,
        epoch: int = 0,
        **kwargs,
    ) -> str:
        """
        Save a model checkpoint with acceleration metadata and metrics.

        This method:
        1. Determines if the model is accelerated or raw
        2. Extracts appropriate state dictionaries
        3. Collects acceleration metadata if applicable
        4. Saves everything to disk
        5. Updates the "best" checkpoint if this is better than previous checkpoints

        Returns:
            Path to the saved checkpoint file
        """
        model = unwrap_model(model_)
        try:
            is_accelerated = isinstance(model, AcceleratedModel)
            save_checkpoint_dict = self.checkpoint_cfg.copy()
            save_strategy = self.checkpoint_cfg["save_strategy"]
            logger.info(
                f"Saving checkpoint for {'accelerated' if is_accelerated else 'raw'} model with {save_strategy} strategy"
            )

            if save_strategy == "raw":
                state_dict = model.model_core.state_dict() if is_accelerated else model.state_dict()
                save_checkpoint_dict["load_strategy"] = "raw"
            else:  # model_core.
                if is_accelerated:
                    state_dict = model.state_dict()
                    save_checkpoint_dict["load_strategy"] = "accelerated"
                else:
                    raise CheckpointValidationError(
                        dedent(
                            f"""
                        Invalid `save_strategy`={save_strategy} for model type {type(model)}.
                        Use 'raw' for non-accelerated models or ensure model is an AcceleratedModel.
                        """
                        )
                    )

            checkpoint = {
                "epoch": epoch,
                "metrics": metrics,
                "model_state": state_dict,
                "acceleration_metadata": self.metadata_handler.get_metadata(model),
                "checkpoint_config": save_checkpoint_dict,
                "config": OmegaConf.to_container(config) if config else {},
                **kwargs,
            }

            if optimizer is not None:
                checkpoint["optimizer_state"] = optimizer.state_dict()

            save_dir = Path(save_dir)
            save_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
            save_path = self._get_save_path(save_dir, **checkpoint)
            self.checkpoint_handler.save_checkpoint(checkpoint, save_path)

            # self.checkpoint_handler.update_history(save_path, metrics)
            if metrics and metrics.get(self.checkpoint_cfg["monitor"], None):
                self.checkpoint_handler.save_best(checkpoint, metrics, save_path, save_dir)

            return str(save_path)

        except OSError as e:
            error_msg = f"Failed to save checkpoint due to I/O error: {str(e)}"
            logger.error(error_msg)
            raise CheckpointSaveError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error while saving checkpoint: {str(e)}"
            logger.error(error_msg)
            raise CheckpointSaveError(error_msg) from e

    def save_checkpoint_context(self, context: "Context") -> str:
        return self.save(
            model=context.model,
            optimizer=context.optimizer,
            config=context.config,
            **context.training_manager.get_training_state_snapshot(),
        )

    @distributed_state.on_main_process
    @staticmethod
    def save_checkpoint(
        model: Union[torch.nn.Module, AcceleratedModel],
        save_dir: PathType,
        metrics: Optional[dict[str, float]] = None,
        cfg: Optional[ConfigType] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        **kwargs,
    ) -> str:
        """Static method to save a checkpoint without instantiating CheckpointManager."""
        cm = CheckpointManager(cfg)
        return cm.save(model, save_dir, optimizer=optimizer, metrics=metrics, epoch=epoch, **kwargs)

    @staticmethod
    def load_checkpoint(
        path: Union[str, Path],
        device: str = "cpu",
        model: Optional[ModelProtocol] = None,
        cfg_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Static method to load checkpoints.
        TODO: realize cfg.model_cfg_override
        TODO: realize cfg.acceleration_cfg_override
        """
        cm = CheckpointManager()
        checkpoint_path = cm._resolve_path(path)
        logger.info(f"Loading {checkpoint_path}...")

        checkpoint = cm.checkpoint_handler.load_checkpoint(checkpoint_path, device)

        checkpoint_cfg = checkpoint.get("checkpoint_config", {})

        if cfg_override:
            checkpoint_cfg.update(cfg_override)

        if not checkpoint.get("acceleration_metadata", None):
            # checkpoint is raw
            checkpoint_cfg = CheckpointManagerRawDefaultConfig.create(checkpoint_cfg)

        if model is not None:
            if not isinstance(model, AcceleratedModel):
                # if model is not wrapped
                checkpoint_cfg = CheckpointManagerRawDefaultConfig.create(checkpoint_cfg)
                model_config = checkpoint.get("acceleration_metadata", {}).get("model_config", {})
                # if model_cfg_override:
                model = AcceleratedModel(model, model_config)

        else:
            # if model is not passed then it should be instantiated
            if "acceleration_metadata" not in checkpoint:
                raise ValueError(
                    dedent(
                        """`load_checkpoint` requires 'acceleration_metadata' if you do not pass the model.
                    Pass the model or use general functions for loading."""
                    )
                )

            model_config = checkpoint["acceleration_metadata"]["model_config"]
            if "model_core" not in model_config:
                raise ValueError(
                    dedent(
                        """Checkpoint config or override must contain 'model.model_core'
                    for Hydra instantiation of AcceleratedModel."""
                    )
                )

            model = AcceleratedModel.instantiate_model(model_config)

        cm.setup_checkpoint_config(checkpoint_cfg)
        result = cm.load(checkpoint, model, device=device)
        return result

    def load(
        self,
        checkpoint: Any,
        model: Union[torch.nn.Module, AcceleratedModel],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint with automatic acceleration handling and validation."""
        result = checkpoint
        state_dict = checkpoint.get(self.checkpoint_cfg["state_dict_key"])
        if state_dict is None:
            raise CheckpointLoadError("Something went wrong, please check `state_dict_key`")
        optimizer_state_dict = checkpoint.get(self.checkpoint_cfg["optimizer_state_dict_key"], None)
        metadata = checkpoint.get("acceleration_metadata", None)

        if metadata:
            model_config = metadata["model_config"]
            acceleration_metadata = metadata["acceleration__meta_data__"]
        else:
            model_config = None
            acceleration_metadata = None

        if self.checkpoint_cfg["strict_validation"]:
            self.metadata_handler.validate_configs(model, model_config, acceleration_metadata)

        # 1. PRE LOAD OPS
        logger.info("\tApplying pre-load operations")
        self._apply_operations(model, OperationType.PRE_LOAD_OPS.value, self.checkpoint_cfg["pre_load_ops"])

        # 2. Reapply acceleration following the saved metadata
        if acceleration_metadata:
            logger.info("\tReapplying the accelerations following the metadata")
            model._reapply_accelerations(acceleration_metadata)

        # 3. CKPT TRANSFORMS
        logger.info("\tApplying ckpt transforms operations")
        self._apply_operations(
            model,
            OperationType.CKPT_TRANSFORMS.value,
            self.checkpoint_cfg["ckpt_transforms"],
            ckpt_model_dict=state_dict,
        )

        # 4. Loading state dict
        logger.info("\tLoading model state")
        load_strategy = self.checkpoint_cfg["load_strategy"]

        if load_strategy == "raw":
            self._load_raw(model, state_dict)
        else:
            self._load_accel(model, state_dict)

        logger.info("\tApplying post-load operations")
        self._apply_operations(model, OperationType.POST_LOAD_OPS.value, self.checkpoint_cfg["post_load_ops"])

        if optimizer_state_dict and self.checkpoint_cfg["load_optimizer_state"]:
            if optimizer:
                optimizer.load_state_dict(optimizer_state_dict)
            else:
                result["optimizer_state"] = optimizer_state_dict

        result.update(
            {
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "model": model,
            }
        )
        return result

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_dir():
            best_path = path / "best.pth"
            if best_path.exists() and self.checkpoint_cfg.get("load_best", False):
                return best_path
            checkpoints = sorted(path.rglob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {path}")
            return checkpoints[0]
        return path

    def _get_save_path(self, save_dir: Path, **kwargs) -> Path:
        metrics = kwargs.get("metrics", {})
        fill_values = {
            "monitor": self.checkpoint_cfg["monitor"],
            "metric": metrics.get(self.checkpoint_cfg["monitor"]) if metrics else 0.0,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        fill_values.update(kwargs)

        template = self.checkpoint_cfg["filename_template"]
        filename = render_filename(template, fill_values)
        return save_dir / filename

    def _load_accel(self, model: AcceleratedModel, state_dict: dict):
        err = model.load_state_dict(state_dict, strict=self.checkpoint_cfg["strict_loading"])
        logger.info(err)

    def _load_raw(self, model: Union[torch.nn.Module, AcceleratedModel], state_dict: dict):
        if isinstance(model, AcceleratedModel):
            err = model.model_core.load_state_dict(state_dict, strict=self.checkpoint_cfg["strict_loading"])
        else:
            err = model.load_state_dict(state_dict, strict=self.checkpoint_cfg["strict_loading"])
        logger.info(err)

    def _apply_operations(self, model: Any, op_type: str, ops_list: Optional[list] = None, **kwargs):
        """Apply operations of a given type from the registry.

        Args:
            model: The model to apply operations to.
            op_type: Operation type (e.g., 'pre_load_ops').
            ops_list: List of operation names or dicts with args.
            **kwargs: Additional arguments for operations (e.g., state_dict).

        Raises:
            ValueError: If any operation is not registered.
        """
        if ops_list is None:
            return

        missed = []
        for op in ops_list:
            op_name = op if isinstance(op, str) else list(op.keys())[0]
            op_args = {} if isinstance(op, str) else op[op_name]
            try:
                operation = registry.get_operation(op_type, op_name)
                operation(model, **op_args, **kwargs)
            except KeyError:
                missed.append(op_name)
        if missed:
            raise ValueError(
                dedent(
                    f"""Operations {missed} not found in '{op_type}'.
                    Register them with 'registry.register_operation("{op_type}")'.
                    Available operations: {registry.list_operations(op_type)[op_type]}"""
                )
            )
