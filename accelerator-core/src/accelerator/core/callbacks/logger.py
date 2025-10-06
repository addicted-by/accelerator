import csv
import mlflow # type: ignore
from pathlib import Path
from typing import Dict, Any

from .base import BaseLoggerCallback
from torch.utils.tensorboard import SummaryWriter


from accelerator.utilities.distributed_state.state import distributed_state
from accelerator.utilities.logging import get_logger

logger = get_logger(__name__)


class TensorBoardLogger(BaseLoggerCallback):
    def __init__(self, 
                 log_dir: str = './tb_logs', 
                 log_interval: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.writer = None
        
    @distributed_state.on_main_process
    def on_train_begin(self, context):
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        if hasattr(context, 'config') and context.config:
            self.log_hyperparams(dict(context.config))
            
    @distributed_state.on_main_process
    def on_train_batch_end(self, context):
        if self.writer:
            step = context.training_manager.state.global_step
            if step % self.log_interval == 0:
                metrics = context.training_manager.state.training_metrics
                if metrics:
                    self.log_metrics(metrics, step)
    
    @distributed_state.on_main_process
    def on_train_epoch_end(self, context):
        if self.writer:
            epoch = context.training_manager.state.global_epoch
            
            train_metrics = context.training_manager.state.training_metrics
            if train_metrics:
                timing = {k: v for k, v in train_metrics.items() if k in ("eta", "elapsed")}
                other_metrics = {k: v for k, v in train_metrics.items() if k not in timing}
                if other_metrics:
                    prefixed_metrics = {f"train/{k}": v for k, v in other_metrics.items()}
                    self.log_metrics(prefixed_metrics, epoch)
                if timing:
                    self.log_metrics(timing, epoch)
            
            val_metrics = getattr(context.training_manager.state, 'validation_metrics', {})
            if val_metrics:
                prefixed_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                self.log_metrics(prefixed_metrics, epoch)
            
            learning_rates = getattr(context.training_manager.state, 'learning_rates', {})
            if learning_rates:
                prefixed_metrics = {f"lr/{k}": v for k, v in learning_rates.items()}
                self.log_metrics(prefixed_metrics, epoch)
                
    def log_scalar(self, name: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(name, value, step)
                
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.writer:
            for k, v in metrics.items():
                if k in ("eta", "elapsed"):
                    # Log timing metrics as text to preserve formatting
                    self.writer.add_text(k, str(v), step)
                elif isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)
                
    def log_hyperparams(self, hparams: Dict[str, Any]):
        # FIX!!!
        if self.writer:
            clean_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_hparams[k] = v
                else:
                    clean_hparams[k] = str(v)
            self.writer.add_hparams(clean_hparams, {})
            
    def log_image(self, name: str, image, step: int):
        if self.writer:
            self.writer.add_image(name, image, step)
            
    @distributed_state.on_main_process
    def on_train_end(self, context):
        if self.writer:
            self.writer.close()
            self.writer = None


class MLflowLogger(BaseLoggerCallback):
    def __init__(self, 
                 tracking_uri: str = './mlruns',
                 experiment_name: str = 'default',
                 log_interval: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        self.run = None
        
    @distributed_state.on_main_process
    def on_train_begin(self, context):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()
        
        if hasattr(context, 'config') and context.config:
            params = {k: str(v) for k, v in dict(context.config).items()}
            mlflow.log_params(params)
    
    @distributed_state.on_main_process
    def on_train_batch_end(self, context):
        if self.run:
            step = context.training_manager.state.global_step
            if step % self.log_interval == 0:
                metrics = context.training_manager.state.training_metrics
                if metrics:
                    self.log_metrics(metrics, step)
    
    @distributed_state.on_main_process
    def on_train_epoch_end(self, context):
        if self.run:
            epoch = context.training_manager.state.current_epoch
            
            train_metrics = context.training_manager.state.training_metrics
            if train_metrics:
                timing = {k: v for k, v in train_metrics.items() if k in ("eta", "elapsed")}
                other_metrics = {k: v for k, v in train_metrics.items() if k not in timing}
                if other_metrics:
                    prefixed_metrics = {f"train_{k}": v for k, v in other_metrics.items()}
                    self.log_metrics(prefixed_metrics, epoch)
                if timing:
                    self.log_metrics(timing, epoch)
            
            val_metrics = getattr(context.training_manager.state, 'validation_metrics', {})
            if val_metrics:
                timing = {k: v for k, v in val_metrics.items() if k in ("eta", "elapsed")}
                other_metrics = {k: v for k, v in val_metrics.items() if k not in timing}
                if other_metrics:
                    prefixed_metrics = {f"val_{k}": v for k, v in other_metrics.items()}
                    self.log_metrics(prefixed_metrics, epoch)
                if timing:
                    self.log_metrics(timing, epoch)
            
    def log_scalar(self, name: str, value: float, step: int):
        if self.run:
            mlflow.log_metric(name, value, step=step)
            
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self.run:
            clean_metrics = {}
            for k, v in metrics.items():
                if k in ("eta", "elapsed"):
                    mlflow.log_text(str(v), f"{k}_step_{step}.txt")
                elif isinstance(v, (int, float)):
                    clean_metrics[k] = v
            if clean_metrics:
                mlflow.log_metrics(clean_metrics, step=step)
                
    def log_hyperparams(self, hparams: Dict[str, Any]):
        if self.run:
            params = {k: str(v) for k, v in hparams.items()}
            mlflow.log_params(params)
            
    def log_image(self, name: str, image, step: int):
        if self.run:
            mlflow.log_image(image, f"{name}_step_{step}.png")
            
    @distributed_state.on_main_process
    def on_train_end(self, context):
        if self.run:
            mlflow.end_run()
            self.run = None


class CSVLogger(BaseLoggerCallback):
    def __init__(self, 
                 log_dir: str = './csv_logs', 
                 log_interval: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.file_path = None
        self.fieldnames = None
        
    @distributed_state.on_main_process
    def on_train_begin(self, context):
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = log_dir / "metrics.csv"
        
        self.fieldnames = ["step", "epoch", "phase"]
        
        with open(self.file_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    @distributed_state.on_main_process
    def on_train_batch_end(self, context):
        step = context.training_manager.state.global_step
        if step % self.log_interval == 0:
            epoch = context.training_manager.state.current_epoch
            metrics = context.training_manager.state.training_metrics
            if metrics:
                self._write_metrics(metrics, step, epoch, "train_batch")
    
    @distributed_state.on_main_process        
    def on_train_epoch_end(self, context):
        epoch = context.training_manager.state.current_epoch
        step = context.training_manager.state.global_step
        
        train_metrics = context.training_manager.state.training_metrics
        if train_metrics:
            self._write_metrics(train_metrics, step, epoch, "train_epoch")
            
        val_metrics = getattr(context.training_manager.state, 'validation_metrics', {})
        if val_metrics:
            self._write_metrics(val_metrics, step, epoch, "val_epoch")
    
    def _write_metrics(self, metrics: Dict[str, Any], step: int, epoch: int, phase: str):
        if not self.file_path:
            return
            
        row_data = {
            "step": step,
            "epoch": epoch,
            "phase": phase
        }
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)) or k in ("eta", "elapsed"):
                row_data[k] = v
            if k not in self.fieldnames:
                self.fieldnames.append(k)

        with open(self.file_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)
                
    def log_scalar(self, name: str, value: float, step: int):
        pass
                
    def log_metrics(self, metrics: Dict[str, float], step: int):
        pass
                
    def log_hyperparams(self, hparams: Dict[str, Any]):
        if self.file_path:
            config_path = self.file_path.parent / "config.csv"
            with open(config_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["parameter", "value"])
                writer.writeheader()
                for k, v in hparams.items():
                    writer.writerow({"parameter": k, "value": str(v)})
                    
    def log_image(self, name: str, image, step: int):
        pass
