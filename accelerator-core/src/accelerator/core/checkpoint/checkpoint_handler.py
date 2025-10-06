from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime
import shutil

import torch
from omegaconf import DictConfig

from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.logging import get_logger


logger = get_logger(__name__)


class CheckpointHandler:
    """Handles low-level checkpoint operations and retention policies."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.checkpoint_history: Dict[str, Dict[str, Any]] = {}
        self.best_metric = None
        
        mode = self.config['mode']
        if mode == 'min':
            self.best_metric = float('inf')
        elif mode == 'max':
            self.best_metric = -float('inf')

    @distributed_state.on_main_process
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], save_path: Path) -> None:
        """Save checkpoint data to disk.

        Args:
            checkpoint_data: Dictionary containing checkpoint data
            save_path: Path where to save the checkpoint
        """
        try:
            torch.save(checkpoint_data, save_path)
            logger.info(f"Checkpoint saved to {save_path}")
        except (OSError, IOError) as e:
            error_msg = f"Failed to save checkpoint due to I/O error: {str(e)}"
            logger.error(error_msg)
            raise

    def load_checkpoint(self, path: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
        """Load checkpoint data from disk.

        Args:
            path: Path to the checkpoint file
            device: Device to load the checkpoint to

        Returns:
            Dictionary containing checkpoint data
        """
        try:
            weights_only = self.config.get('weights_only', True)
            return torch.load(path, map_location=device, weights_only=weights_only)
        except Exception as e:
            error_msg = f"Failed to load checkpoint {path}: {str(e)}"
            logger.error(error_msg)
            raise

    def update_history(self, save_path: Path, metrics: Dict[str, Any]) -> None:
        """Update checkpoint history with metrics.

        Args:
            save_path: Path where checkpoint was saved
            metrics: Dictionary of metrics for the checkpoint
        """
        metrics['timestamp'] = datetime.now().timestamp()
        self.checkpoint_history[str(save_path)] = metrics

    @distributed_state.on_main_process
    def save_best(self, checkpoint: Dict, metrics: Dict, current_path: Path, save_dir: Path) -> None:
        """Save best checkpoint if current metrics are better.

        Args:
            checkpoint: Checkpoint data
            metrics: Current metrics
            current_path: Path to current checkpoint
            save_dir: Directory to save checkpoints
        """
        monitor = self.config.get("monitor")
        
        if monitor is None:
            logger.info("No monitor metric configured, skipping best checkpoint update")
            return
            
        current_value = metrics.get(monitor)
        if current_value is None:
            logger.warning(f"Monitor metric '{monitor}' not found in metrics")
            return

        mode = self.config.get("mode", "min")
        is_better = (
            (mode == "min" and current_value < self.best_metric) or
            (mode == "max" and current_value > self.best_metric)
        )

        if is_better:
            self.best_metric = current_value
            best_path = save_dir / "best.pth"
            shutil.copyfile(current_path, best_path)
            logger.info(f"Updated best checkpoint at {best_path} with {monitor}={current_value}")