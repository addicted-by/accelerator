from typing import Optional
import torch
import torch.nn.utils.parametrize as parametrize
from .base_pruner import BasePruner
from .. import tracer
from . import feature_merging

class MergingPruner(BasePruner):
    """Implementation of feature merging pruning strategy.
    
    This pruner reduces model size by merging similar features using various
    importance metrics and parametrization techniques.
    """

    def __init__(self, **kwargs):
        """Initialize MergingPruner.

        Args:
            **kwargs: Configuration options including:
                - imp_fn: Name of importance function to use
                - initialization_cfg: Configuration for weight initialization
                - pruning_dims: Dimensions to prune
                - ignored_dims: Dimensions to ignore during pruning
                - use_scales: Whether to use scaling factors
                - single_parametrization: Whether to share parametrization
                - pretrain_importances: Whether to pretrain importance scores
        """
        super().__init__(**kwargs)

        self.imp_fn_name = self.pruner_cfg.get("imp_fn")
        if self.imp_fn_name:
            if self.imp_fn_name not in feature_merging.imp_fns:
                raise ValueError(f"Importance function {self.imp_fn_name} not implemented")
            
            self.imp_fn = feature_merging.imp_fns[self.imp_fn_name]()
            
            # Enable gradient computation for certain importance functions
            if isinstance(self.imp_fn, (feature_merging.importance.TaylorImportance,
                                      feature_merging.importance.HessianImportance)):
                if self.verbose:
                    print(f"{self.imp_fn.__class__.__name__} requires gradients")
                self.req_grad = True
        else:
            self.imp_fn = None

        self.initialization_cfg = self.pruner_cfg.get("initialization_cfg")

    def prune(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, **kwargs) -> torch.nn.Module:
        """Execute the feature merging pruning process.

        Args:
            model: Model to be pruned
            dataloader: DataLoader for evaluation
            **kwargs: Additional arguments

        Returns:
            Pruned model with merged features
        """
        # Initialize if not already done
        if not hasattr(self, 'input_example'):
            self.device = next(iter(model.parameters())).device
            self.input_example, *_ = self._BasePruner__get_sample_input(dataloader, self.device)

        # Build feature groups
        groups = tracer.GroupTracer(
            model,
            self.pruner_cfg["pruning_dims"],
            self.pruner_cfg["ignored_dims"]
        ).build_groups(*self.input_example)
        
        # Initialize parametrized modules tracking
        setattr(model, 'parametrized_modules', set())

        # Process each group
        for group in groups:
            if self.verbose:
                print(group)

            initial_size = group.size
            new_size = self._get_target_size(group, initial_size)

            self._record_pruning(group, initial_size, new_size)

            if initial_size == new_size:
                if self.verbose:
                    print(f"\t[GROUP INFO] Group will preserve size {new_size}")
                continue

            if self.verbose:
                print(f"\t[GROUP INFO] Group will be pruned to {new_size}")

            # Calculate importance scores and indices
            important_indices = self._get_important_indices(group, new_size)
            
            # Apply parametrizations
            self._apply_parametrizations(model, group, initial_size, new_size, important_indices)

        # Configure gradients
        self._configure_gradients(model)

        return model

    def _get_target_size(self, group, initial_size: int) -> int:
        """Determine target size for the group."""
        if self.pruner_cfg.get("pruning_dict"):
            return next(
                (size for name, size in self.pruner_cfg["pruning_dict"].items()
                 if name in group.params[0]["name"]),
                initial_size
            )
        return int((1 - self.pruner_cfg['pruning_ratio']) * initial_size)

    def _record_pruning(self, group, initial_size: int, new_size: int) -> None:
        """Record pruning operation details."""
        self.pruning_record.append((
            str(group).replace("\n", ""),
            initial_size,
            new_size
        ))

    def _get_important_indices(self, group, new_size: int) -> Optional[torch.Tensor]:
        """Calculate importance scores and get important indices."""
        if not self.imp_fn:
            return None

        imp = self.imp_fn(group)
        important_indices = imp.argsort(descending=True)
        important_indices = important_indices[:new_size]
        return important_indices.sort().values

    def _apply_parametrizations(self, model, group, initial_size: int, new_size: int,
                              important_indices: Optional[torch.Tensor]) -> None:
        """Apply parametrizations to the group."""
        single_parametrization = None

        for param in group.params:
            module_name, wb = param['name'].rsplit('.', 1)
            submodule = model.get_submodule(module_name)
            device = submodule.weight.device

            if wb == 'weight':
                parametrization = self._create_feature_merging(
                    initial_size, new_size, param['dim'],
                    important_indices, param['name'],
                    single_parametrization, device
                )
                if self.pruner_cfg.get('single_parametrization', False):
                    single_parametrization = parametrization.mlp.fc
                    if isinstance(single_parametrization, torch.nn.Sequential):
                        single_parametrization = single_parametrization[0]
            elif wb == 'bias':
                parametrization = feature_merging.BiasParametrization(new_size, important_indices)
            else:
                raise ValueError(f"Unknown parameter type: {wb}")

            parametrize.register_parametrization(submodule, wb, parametrization, unsafe=True)
            model.parametrized_modules.add(module_name)

    def _create_feature_merging(self, initial_size: int, new_size: int, dim: int,
                              important_indices: Optional[torch.Tensor], name: str,
                              single_parametrization: Optional[torch.nn.Module],
                              device: torch.device) -> feature_merging.FeatureMerging:
        """Create feature merging parametrization."""
        return feature_merging.FeatureMerging(
            size=initial_size,
            new_size=new_size,
            dim=dim,
            important_indices=important_indices,
            name=name,
            use_scales=(
                self.pruner_cfg['use_scales']
                and single_parametrization is not None
            ),
            linear=single_parametrization,
            initialization_cfg=self.initialization_cfg,
            pretrain_stage=self.pruner_cfg['pretrain_importances']
        ).to(device)

    def _configure_gradients(self, model: torch.nn.Module) -> None:
        """Configure gradient computation for the model."""
        model.zero_grad()

        # Freeze most parameters
        for params in model.parameters():
            params.requires_grad = False

        # Enable gradients for specific components
        for module_name in model.parametrized_modules:
            module = model.get_submodule(module_name)
            if hasattr(module, 'bias') and parametrize.is_parametrized(module, 'bias'):
                parametrize.remove_parametrizations(module, 'bias', True)
                module.bias.requires_grad = True

        # Enable gradients for MLP parameters
        for module in model.modules():
            if isinstance(module, feature_merging.FeatureMergingMLP):
                for params in module.parameters():
                    params.requires_grad = True