from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

try:
    import torch_integral
except:  # noqa: E722
    print('Torch integral is not installed')

import torch
import os
import sys
import yaml
from ..utils.typing import PathType, ModuleType, InOutType
from ..utils import (
    to_device, 
    save_report, 
    fuse_batchnorm,
    remove_parametrizations,
    reapply_parametrizations
)

from ..utils.op_counter import count_ops_and_params
from . import feature_merging
from .. import tracer
import torch.nn.utils.parametrize as parametrize


class BasePruner:
    def __init__(
        self,
        experiment_path: Optional[PathType]=None,
        log_file: Optional[TextIO]=None,
        loss_fn: Optional[Callable]=None,
        postprocess_output: Optional[Callable]=None,
        pruner_cfg: Dict=None,
        **kwargs,
    ):
        """
        Initializes the BasePruner with the given configuration.

        ### Args:
            experiment_path (Optional[PathType]): Path to the directory where pruning results
                will be saved. If `None` then the results will be saved in the current folder.
            log_file (Optional[TextIO]): The text output stream to write logging information.
            loss_fn (Optional[Callable]): The loss function used to calculate the loss.
                If `None` then the L1 loss will be used.

        ### Kwargs:
            pruning configs that must contain:
                pruner_cfg corresponds to the chosen pruner type inherited by this class.
        """
        self.experiment_path: PathType = (
            Path(experiment_path) if experiment_path else os.getcwd()
        )
        self.log_file: Optional[TextIO] = log_file if log_file else sys.stdout
        self.pruner_cfg: Dict = pruner_cfg.copy()
        self.num_batches = kwargs.get("num_batches", 50)
        self.verbose = kwargs.get("verbose", True)
        self.to_save_report = kwargs.get("to_save_report", True)
        self.loss_fn: Optional[Callable] = loss_fn
        self.req_grad = False
        self.stats = defaultdict()
        self.pruning_record = []
        if self.pruner_cfg.get("pruning_dict"):
            self.pruner_cfg["pruning_ratio"] = 0.
        
        self.postprocess_output = postprocess_output
        print(self)

    @abstractmethod
    def prune(self):
        """Abstract method to implement pruning logic."""
        pass

    def _calculate_loss(
        self,
        model,
        dataloader,
        device,
        req_grad=False,
        num_batches=-1
    ):
        print(f"Loss calculation with grad: {req_grad}")
        loss = 0.0
        if num_batches == -1:
            num_batches = len(dataloader)
            
        for idx, sample in enumerate(dataloader):
            if idx == num_batches:
                break

            to_device(sample, device)
            inputs, ground_truth, *_ = sample
            loss_value = self._get_loss_value(model, inputs, ground_truth, req_grad)

            loss += loss_value.detach().cpu().item() / num_batches
            print(f"Batch {idx}: {loss_value.item()}")

        return loss

    def _get_loss_value(
        self,
        model: ModuleType,
        inputs: InOutType,
        ground_truth: InOutType,
        req_grad: bool,
        loss_fn: Optional[Callable] = None,
        return_net_result: bool = False,
    ):
        with torch.set_grad_enabled(req_grad):
            net_result = model(*inputs)

        if self.postprocess_output:
            net_result = self.postprocess_output(net_result)

        if isinstance(net_result, torch.Tensor):
            net_result = (net_result,)

        if isinstance(ground_truth, torch.Tensor):
            ground_truth = (ground_truth,)

        if self.loss_fn:
            loss = self.loss_fn(*net_result, *ground_truth)
        else:
            # predefined loss calculation: L1 loss
            loss = 0.0
            for net, gt in zip(net_result, ground_truth):
                loss += torch.mean(torch.abs(net - gt))

        if req_grad:
            if (
                hasattr(self, 'imp_fn') 
                and 
                isinstance(self.imp_fn, feature_merging.importance.HessianImportance)
            ):
                print("Accumulating g^2 for Hessian")
                model.zero_grad() # clear gradients
                loss.backward(retain_graph=True) # simgle-sample gradient
                self.imp_fn.accumulate_grad(model) # accumulate g^2
            else:
                print("Accumulating gradients")
                loss.backward()

        if return_net_result:
            return loss, net_result
        return loss

    @classmethod
    def __get_sample_input(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Retrieves a sample input from the dataloader
        and transfers it to the specified device.

        ### Args:
            `dataloader` (torch.utils.data.DataLoader): The dataloader to sample from.
            `device` (torch.device): The device to transfer the sample to.

        ### Returns:
            Tuple[torch.Tensor, ...]: The sampled input data and target.
        """
        sample = next(iter(dataloader))

        to_device(sample, device)
        return sample

    def _calculate_stats(self, model, dataloader, stage, req_grad=False):
        with torch.no_grad():
            (
                self.stats[f"flops_{stage}"],
                self.stats[f"total_params_{stage}"],
            ) = count_ops_and_params(model=model, example_inputs=self.input_example)

        self.stats[f"loss_{stage}"] = self._calculate_loss(
            model=model,
            dataloader=dataloader,
            req_grad=req_grad,
            num_batches=self.num_batches,
            device=self.device,
        )

        self.stats[f"flops_{stage}"] = self.stats[f"flops_{stage}"] / 1e9

        if stage == "after":
            self.stats["ratio"] = (
                self.stats["total_params_before"] / self.stats["total_params_after"]
            )
            self.stats["compression"] = (
                1 - self.stats["total_params_after"] / self.stats["total_params_before"]
            )

    def __call__(
        self,
        model: ModuleType,
        dataloader: torch.utils.data.DataLoader,
        **kwargs: Any,
    ) -> ModuleType:
        """
        Executes pruning on the provided model
        using the given dataloader with options
        for verbosity and saveing a report

        ### Args:
            `model` (ModuleType): The model to be pruned.
            `dataloader` (torch.utils.data.DataLoader): The dataloader to use for assessing the model.
            `verbose` (bool): If True, prints detailed information about the pruning process.
            `to_save_report` (bool): If True, saves a detailed report of the pruning process.


        ### Keyword Args (Any):
            `dataset` (torch.utils.data.Dataset): dataset for further finetuning.
                Since the logic with partial training is realized.
            `custom_collate_fn` (Callable): function how to collate the batch
                in format: `inputs` (Tuple), `gts` (Tuple), `additional` (Any).

        ### Returns:
            ModuleType: The pruned model.
        """
        
        # loss calculation before, parameters, flops, logging
        model.train(False)
        fuse_batchnorm(model)
        if self.req_grad:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    print(f"Be careful. Unfreezing {name}!")
                    p.requires_grad = True

        self.device = next(iter(model.parameters())).device
        self.input_example, gt, *_ = self.__get_sample_input(dataloader, self.device)
        self._calculate_stats(model, dataloader, 'before', self.req_grad)

        model = self.prune(model, dataloader=dataloader, **kwargs)
        if isinstance(model, torch_integral.IntegralModel):
            inn_model = model
            model = inn_model.get_unparametrized_model()
            model.integral = True
        
        parametrized = remove_parametrizations(model)
        self._calculate_stats(model, dataloader, 'after', False)
        reapply_parametrizations(model, parametrized, True)
        if hasattr(model, "integral"):
            model = inn_model
        if self.verbose:
            self.print_stats()

        if len(self.pruning_record) == 0:
            print("There is no pruning!")
        elif self.to_save_report:
            save_report(self.pruning_record, self.experiment_path, self.stats["ratio"])

        model.train(True)
        return model

    def print_stats(self):
        print(
            "\n".join(
                [
                    f"Loss before: {self.stats['loss_before']:.8f}",
                    f"Loss after: {self.stats['loss_after']:.8f}",
                    f"Total number of params before: {(self.stats['total_params_before'] / 1e+6):.4f}M",
                    f"Total number of params after: {(self.stats['total_params_after'] / 1e+6):.4f}M",
                    f"Flops: {self.stats['flops_before']:.5f} GFlops -> {self.stats['flops_after']:.5f} GFlops",
                    f"Ratio: {self.stats['ratio']:.4f}x less params",
                    f"Compression: {self.stats['compression']}"
                ]
            )
        )


    @property
    def get_last_stats(self):
        """
        Returns the most recent training statistics
        """
        return self.stats

    def __repr__(self):
        """
        Generates a YAML-based string representation
        of the object for debugging and logging
        """
        attrs = {
            k: str(v)
            for k, v in vars(self).items()
            if not callable(v) and not k.startswith("_")
        }
        desc = "\n".join(
            [
                self.__class__.__name__,
                "Attributes:",
                yaml.dump(attrs),
            ]
        )
        return desc


class MergingPruner(BasePruner):
    def __init__(self, **kwargs):
        super(MergingPruner, self).__init__(**kwargs)

        self.imp_fn_name: str = self.pruner_cfg.get("imp_fn", None)
        if self.imp_fn_name:
            assert (
                self.imp_fn_name in feature_merging.imp_fns.keys()
            ), f"{self.imp_fn_name} does not implemented!"
            self.imp_fn: Callable = feature_merging.imp_fns[self.imp_fn_name]
            if self.imp_fn in [
                feature_merging.importance.TaylorImportance,
                feature_merging.importance.HessianImportance,
            ]:
                print(f"{self.imp_fn.__name__} requires grad!")
                self.req_grad = True
            
            self.imp_fn = self.imp_fn()
        else:
            self.imp_fn = None

        self.initialization_cfg = self.pruner_cfg.get("initialization_cfg", None)


    def prune(
        self,
        model: ModuleType,
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> ModuleType:
        if hasattr(self, 'input_example'):
            ...
        else:
            self.device = next(iter(model.parameters())).device
            self.input_example, *_ = self._BasePruner__get_sample_input(
                dataloader, self.device
            )
        
        groups = tracer.GroupTracer(
            model, 
            self.pruner_cfg["pruning_dims"], 
            self.pruner_cfg["ignored_dims"]
        ).build_groups(*self.input_example)
        setattr(model, 'parametrized_modules', set())
        # model.parametrized_modules = set()

        # print(len(groups))
        for group in groups:
            print(group)
            initial_size = group.size
            if self.pruner_cfg.get("pruning_dict", None):
                new_size = next(
                    (
                        size
                        for name, size in self.pruner_cfg["pruning_dict"].items()
                        if name in group.params[0]["name"]
                    ), initial_size
                )

                print(initial_size, new_size, sep=" => ")

            else:
                new_size = int((1 - self.pruner_cfg['pruning_ratio']) * initial_size)

            
            self.pruning_record.append(
                (
                    str(group).replace("\n", ""),
                    initial_size,
                    new_size,
                )  # ! TODO: update pruning record similarly to the `Pruner`
            )
            if initial_size == new_size:
                print(
                    f"\t[GROUP INFO] Group above will preserve size {new_size}"
                )
            else:
                print(
                    f"\t[GROUP INFO] Group above will be pruned to {new_size}"
                )
            
            if self.imp_fn:
                imp = self.imp_fn(group)
                important_indices = imp.argsort(descending=True)
                important_indices = important_indices[:new_size]
                important_indices = important_indices.sort().values
            else:
                important_indices = None

            single_parametrization = None
            for param in group.params:
                module_name, wb = param['name'].rsplit('.', 1)
                submodule = model.get_submodule(module_name)
                device = submodule.weight.device
            
                if wb == 'weight':
                    parametrization = feature_merging.FeatureMerging(
                        size=initial_size, 
                        new_size=new_size, 
                        dim=param['dim'], 
                        important_indices=important_indices, 
                        name=param['name'],
                        use_scales=(
                            self.pruner_cfg['use_scales']
                            and
                            single_parametrization is not None
                        ),
                        linear=single_parametrization,
                        initialization_cfg=self.initialization_cfg,
                        pretrain_stage=self.pruner_cfg['pretrain_importances']
                    ).to(device)
                    if self.pruner_cfg.get('single_parametrization', False):
                        print("REUSING SINGLE PARAMETRIZATION")
                        single_parametrization = parametrization.mlp.fc
                        if  isinstance(single_parametrization, torch.nn.Sequential):
                            single_parametrization = single_parametrization[0]
                elif wb == 'bias':
                    parametrization = feature_merging.BiasParametrization(
                        new_size,
                        important_indices
                    )
                else:
                    raise NotImplementedError("UNKNOWN KIND!")
                
                    
                parametrize.register_parametrization(
                    submodule,
                    wb,
                    parametrization,
                    unsafe=True
                )
                
                model.parametrized_modules.add(module_name)
          
        model.zero_grad()
        for name, params in model.named_parameters():
            # if 'weight' in name:
            params.requires_grad = False
        
        for module_name in model.parametrized_modules:
            module = model.get_submodule(module_name)
            if hasattr(module, 'bias'):
                if parametrize.is_parametrized(module, 'bias'):
                    parametrize.remove_parametrizations(module, 'bias', True)

                module.bias.requires_grad = True
            
        for name, module in model.named_modules():
            if isinstance(module, feature_merging.FeatureMergingMLP):    
                for name, params in module.named_parameters():
                    params.requires_grad = True

        return model


class IntegralPruner(BasePruner):
    def __init__(self, **kwargs):
        super(IntegralPruner, self).__init__(**kwargs)

        self.continuous_dims = self.pruner_cfg.get("continuous_dims")
        self.discrete_dims = self.pruner_cfg.get("discrete_dims", None)
        self.parametrization_cfg = self.pruner_cfg.get("parametrization_cfg", {
            "scale": 1,
            "use_gridsample": True,
            "quadrature": "TrapezoidalQuadrature"
        })
        self.wrapper_cfg = self.pruner_cfg.get("wrapper_cfg", {
            "init_from_discrete": True,
            "fuse_bn": True,
            "optimize_iters": 0,
            "start_lr": 1e-2,
            "verbose": True
        })
        self.permutation_cfg = self.pruner_cfg.get("permutation_cfg", {
            "class": torch_integral.permutation.NOptPermutation,
            "iters": 100,
        })



    def prune(
        self,
        model: ModuleType,
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> ModuleType:
        if hasattr(self, 'input_example'):
            ...
        else:
            self.device = next(iter(model.parameters())).device
            self.input_example, *_ = self._BasePruner__get_sample_input(
                dataloader, self.device
            )
        if self.continuous_dims is None:
            self.continuous_dims = torch_integral.standard_continuous_dims(model)
        
        
        wrapper = torch_integral.IntegralWrapper(
            **self.wrapper_cfg,
            parametrization_config=self.parametrization_cfg,
            permutation_config=self.permutation_cfg
        )

        to_device(self.input_example, "cpu")
        model = wrapper(
            model.cpu(), 
            self.input_example,
            self.continuous_dims,
            self.discrete_dims
        ).to(self.device)
        to_device(self.input_example, self.device)
        for group in model.groups:
            print(group)
            initial_size = group.size
            if self.pruner_cfg.get("pruning_dict", None):
                new_size = next(
                    (
                        size
                        for name, size in self.pruner_cfg["pruning_dict"].items()
                        if name in group.params[0]["name"]
                    ), initial_size
                )
                if new_size == "ratio":
                    print(f"USING PRUNING RATIO: {self.pruner_cfg['pruning_dict']['ratio']}")
                    new_size = int((1 - self.pruner_cfg["pruning_dict"]['ratio']) * initial_size)

                print(initial_size, new_size, sep=" => ")

            else:
                print(f"Uniformly resizing with ratio: {self.pruner_cfg['pruning_ratio']}")
                new_size = int((1 - self.pruner_cfg['pruning_ratio']) * initial_size)

            
            self.pruning_record.append(
                (
                    str(group).replace("\n", ""),
                    initial_size,
                    new_size,
                )  # ! TODO: update pruning record similarly to the `Pruner`
            )
            if initial_size == new_size:
                print(
                    f"\t[GROUP INFO] Group above will preserve size {new_size}"
                )
            else:
                print(
                    f"\t[GROUP INFO] Group above will be pruned to {new_size}"
                )
            
            group.reset_grid(torch_integral.TrainableGrid1D(new_size))

        model.grid_tuning(False, True, False)
        return model
