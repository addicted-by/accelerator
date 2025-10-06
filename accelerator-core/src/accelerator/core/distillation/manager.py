from typing import Dict, Optional, Any
from omegaconf import DictConfig


from accelerator.core.model import AcceleratedModel

from accelerator.utilities.logging import get_logger
from accelerator.utilities.api_desc import APIDesc

log = get_logger(__name__)


@APIDesc.developer(dev_info='Ryabykin Alexey r00926208')
@APIDesc.status(status_level='Unstable')
class DistillationManager:
    """Manages teacher models for knowledge distillation.
    
    This class handles the initialization and management of teacher models used for
    knowledge distillation. It supports loading multiple teacher models with their
    respective configurations and checkpoints. All teacher models are automatically
    set to evaluation mode with gradients disabled.
    """
    
    def __init__(self, teachers: Dict[str, AcceleratedModel]):
        """Initialize the DistillationManager with pre-loaded teacher models.
        
        Args:
            teachers: Dictionary mapping teacher names to AcceleratedModel instances.
                     All models will be automatically frozen (gradients disabled).
        """
        self.teachers: Dict[str, AcceleratedModel] = {}
        
        # Add teachers and freeze them
        for name, model in teachers.items():
            self.teachers[name] = model
            self._freeze_model(model)
            log.info(f"Added and froze teacher model: {name}")
    
    @staticmethod
    def from_config(cfg: DictConfig) -> 'DistillationManager':
        """Initialize the DistillationManager from configuration.
        
        Args:
            cfg: Configuration containing teacher model specifications and settings.
                Expected structure:
                {
                    'enabled': bool,  # Whether distillation is enabled
                    'teachers': {
                        'teacher_name': {
                            'model': DictConfig,  # Model configuration
                            'checkpoint': {
                                'path': str,      # Path to checkpoint
                                'config': DictConfig  # Checkpoint loading config
                            }
                        }
                    }
                }
                
        Returns:
            Initialized DistillationManager instance
        """
        teachers = {}
        
        if not cfg.get('enabled', False):
            log.info("Distillation is disabled")
            return DistillationManager(teachers)
        
        if not cfg.get('teachers'):
            log.warning("No teacher models configured")
            return DistillationManager(teachers)
        
        for name, teacher_cfg in cfg.teachers.items():
            try:
                model = DistillationManager._create_teacher_model(name, teacher_cfg)
                if model:
                    teachers[name] = model
                    log.info(f"Successfully initialized teacher model: {name}")
            except Exception as e:
                log.error(f"Failed to initialize teacher model '{name}': {str(e)}")
                continue
        
        return DistillationManager(teachers)
    
    @staticmethod
    def _create_teacher_model(name: str, teacher_cfg: DictConfig) -> Optional[AcceleratedModel]:
        """Create a single teacher model from configuration.
        
        Args:
            name: Name of the teacher model
            teacher_cfg: Configuration for this specific teacher
            
        Returns:
            AcceleratedModel instance or None if creation failed
        """
        model_cfg = teacher_cfg.get('model')
        if not model_cfg:
            log.warning(f"No model configuration found for teacher '{name}'")
            return None
        
        try:
            model = AcceleratedModel.instantiate_model(model_cfg)
        except Exception as e:
            log.error(f"Failed to instantiate model for teacher '{name}': {str(e)}")
            return None
        
        checkpoint_cfg = teacher_cfg.get('checkpoint')
        if checkpoint_cfg:
            try:
                DistillationManager._load_teacher_checkpoint(model, checkpoint_cfg)
                log.info(f"Checkpoint loaded for teacher '{name}'")
            except Exception as e:
                log.error(f"Failed to load checkpoint for teacher '{name}': {str(e)}")
                return None
        
        return model
    
    @staticmethod
    def _load_teacher_checkpoint(model: AcceleratedModel, checkpoint_cfg: DictConfig) -> None:
        """Load checkpoint for a teacher model with custom configuration.
        
        Args:
            model: Teacher model to load checkpoint into
            checkpoint_cfg: Checkpoint configuration containing path and loading settings
        """
        from accelerator.core.checkpoint import CheckpointManager
        
        checkpoint_path = checkpoint_cfg.get('path')
        if not checkpoint_path:
            log.warning("No checkpoint path specified")
            return
        
        manager_cfg = checkpoint_cfg.get('config', DictConfig({}))
        checkpoint_manager = CheckpointManager(manager_cfg)
        
        checkpoint_manager.load(checkpoint_path, model)
        log.info(f"Loaded checkpoint from: {checkpoint_path}")
    
    def _freeze_model(self, model: AcceleratedModel) -> None:
        """Freeze all parameters in a teacher model.
        
        Args:
            model: The teacher model to freeze
        """
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        for module in model.modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    
    def get_teacher(self, name: str) -> Optional[AcceleratedModel]:
        """Get a teacher model by name.
        
        Args:
            name: Name of the teacher model to retrieve
            
        Returns:
            The requested teacher model or None if not found
        """
        return self.teachers.get(name)
    
    @property
    def all_teachers(self) -> Dict[str, AcceleratedModel]:
        """Get all teacher models.
        
        Returns:
            Dictionary of all teacher models
        """
        return self.teachers
    
    def has_teacher(self, name: str) -> bool:
        """Check if a teacher model exists.
        
        Args:
            name: Name of the teacher model to check
            
        Returns:
            True if the teacher exists, False otherwise
        """
        return name in self.teachers
    
    def list_teachers(self) -> list:
        """List all available teacher names.
        
        Returns:
            List of teacher model names
        """
        return list(self.teachers.keys())
    
    def forward(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Run forward pass on all teacher models with gradients disabled.
        
        Args:
            *args: Positional arguments to pass to each teacher model
            **kwargs: Keyword arguments to pass to each teacher model
            
        Returns:
            Dictionary mapping teacher names to their output dictionaries
        """
        results = {}
        
        for name, teacher in self.teachers.items():
            try:
                output = teacher.run_model(*args, req_grad=False, **kwargs)
                results[name] = output
            except Exception as e:
                log.error(f"Forward pass failed for teacher '{name}': {str(e)}")
                results[name] = None
        
        return results
    
    def __call__(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Make the manager callable, forwarding to forward method.
        
        Args:
            *args: Positional arguments to pass to teacher models
            **kwargs: Keyword arguments to pass to teacher models
            
        Returns:
            Dictionary mapping teacher names to their output dictionaries
        """
        return self.forward(*args, **kwargs)
    
    def __len__(self) -> int:
        """Return the number of teacher models.
        
        Returns:
            Number of teacher models
        """
        return len(self.teachers)
    
    def __contains__(self, name: str) -> bool:
        """Check if a teacher model exists using 'in' operator.
        
        Args:
            name: Name of the teacher model to check
            
        Returns:
            True if the teacher exists, False otherwise
        """
        return name in self.teachers
    
    def __repr__(self) -> str:
        """Return string representation of the manager.
        
        Returns:
            String representation showing teacher models and their status
        """
        if not self.teachers:
            return "DistillationManager(no teachers)"
        
        teacher_info = []
        for name, model in self.teachers.items():
            model_type = type(model.model_core).__name__
            param_count = sum(p.numel() for p in model.parameters())
            frozen_status = "frozen" if not any(p.requires_grad for p in model.parameters()) else "not frozen"
            
            teacher_info.append(f"{name}: {model_type} ({param_count:,} params, {frozen_status})")
        
        return f"DistillationManager({len(self.teachers)} teachers):\n  " + "\n  ".join(teacher_info)