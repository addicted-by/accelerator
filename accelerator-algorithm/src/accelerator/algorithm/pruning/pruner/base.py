import os
import sys
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import torch
import yaml

from ..utils import fuse_batchnorm, reapply_parametrizations, remove_parametrizations, save_report, to_device
from ..utils.op_counter import count_ops_and_params
from ..utils.typing import InOutType, ModuleType, PathType


class BasePruner:
    """Base class for implementing pruning strategies.

    This class provides the foundation for implementing various pruning approaches.
    It handles common functionality such as loss calculation, statistics tracking,
    and reporting.
    """

    def __init__(
        self,
        experiment_path: Optional[PathType] = None,
        log_file: Optional[TextIO] = None,
        loss_fn: Optional[Callable] = None,
        postprocess_output: Optional[Callable] = None,
        pruner_cfg: dict = None,
        **kwargs,
    ):
        """Initialize the BasePruner.

        Args:
            experiment_path: Directory path for saving pruning results.
            log_file: Text stream for logging information.
            loss_fn: Custom loss function (defaults to L1 loss if None).
            postprocess_output: Function to process model outputs before loss calculation.
            pruner_cfg: Configuration dictionary for the pruner.
            **kwargs: Additional configuration options.
        """
        self.experiment_path = Path(experiment_path) if experiment_path else Path(os.getcwd())
        self.log_file = log_file if log_file else sys.stdout
        self.pruner_cfg = pruner_cfg.copy() if pruner_cfg else {}
        self.num_batches = kwargs.get("num_batches", 50)
        self.verbose = kwargs.get("verbose", True)
        self.to_save_report = kwargs.get("to_save_report", True)
        self.loss_fn = loss_fn
        self.req_grad = False
        self.stats = defaultdict()
        self.pruning_record = []

        # Handle pruning dictionary configuration
        if self.pruner_cfg.get("pruning_dict"):
            self.pruner_cfg["pruning_ratio"] = 0.0

        self.postprocess_output = postprocess_output
        if self.verbose:
            print(self)

    @abstractmethod
    def prune(self, model: ModuleType, dataloader: torch.utils.data.DataLoader, **kwargs) -> ModuleType:
        """Implement pruning logic in derived classes.

        Args:
            model: The model to be pruned.
            dataloader: DataLoader for model evaluation.
            **kwargs: Additional arguments for specific pruning implementations.

        Returns:
            The pruned model.
        """
        pass

    def _calculate_loss(
        self,
        model: ModuleType,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        req_grad: bool = False,
        num_batches: int = -1,
    ) -> float:
        """Calculate average loss over the dataset.

        Args:
            model: Model to evaluate.
            dataloader: DataLoader containing evaluation data.
            device: Device to run calculations on.
            req_grad: Whether to compute gradients.
            num_batches: Number of batches to process (-1 for all).

        Returns:
            Average loss value.
        """
        if self.verbose:
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
            if self.verbose:
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
    ) -> torch.Tensor:
        """Calculate loss for a single batch.

        Args:
            model: Model to evaluate.
            inputs: Input data.
            ground_truth: Target data.
            req_grad: Whether to compute gradients.
            loss_fn: Optional custom loss function.
            return_net_result: Whether to return model outputs.

        Returns:
            Loss value (and optionally model outputs).
        """
        with torch.set_grad_enabled(req_grad):
            net_result = model(*inputs)

        if self.postprocess_output:
            net_result = self.postprocess_output(net_result)

        if isinstance(net_result, torch.Tensor):
            net_result = (net_result,)

        if isinstance(ground_truth, torch.Tensor):
            ground_truth = (ground_truth,)

        # Calculate loss using custom or default (L1) loss function
        if self.loss_fn:
            loss = self.loss_fn(*net_result, *ground_truth)
        else:
            loss = sum(torch.mean(torch.abs(net - gt)) for net, gt in zip(net_result, ground_truth))

        if req_grad:
            loss.backward()

        return (loss, net_result) if return_net_result else loss

    def _calculate_stats(
        self, model: ModuleType, dataloader: torch.utils.data.DataLoader, stage: str, req_grad: bool = False
    ) -> None:
        """Calculate and store model statistics.

        Args:
            model: Model to analyze.
            dataloader: DataLoader for evaluation.
            stage: Stage identifier ('before' or 'after').
            req_grad: Whether to compute gradients.
        """
        with torch.no_grad():
            flops, total_params = count_ops_and_params(model=model, example_inputs=self.input_example)
            self.stats[f"flops_{stage}"] = flops / 1e9
            self.stats[f"total_params_{stage}"] = total_params

        self.stats[f"loss_{stage}"] = self._calculate_loss(
            model=model,
            dataloader=dataloader,
            req_grad=req_grad,
            num_batches=self.num_batches,
            device=self.device,
        )

        if stage == "after":
            self.stats["ratio"] = self.stats["total_params_before"] / self.stats["total_params_after"]
            self.stats["compression"] = 1 - (self.stats["total_params_after"] / self.stats["total_params_before"])

    def __call__(
        self,
        model: ModuleType,
        dataloader: torch.utils.data.DataLoader,
        **kwargs: Any,
    ) -> ModuleType:
        """Execute pruning process on the model.

        Args:
            model: Model to be pruned.
            dataloader: DataLoader for evaluation.
            **kwargs: Additional arguments.

        Returns:
            Pruned model.
        """
        model.train(False)
        fuse_batchnorm(model)

        if self.req_grad:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    if self.verbose:
                        print(f"Unfreezing {name}")
                    p.requires_grad = True

        self.device = next(iter(model.parameters())).device
        sample = next(iter(dataloader))
        to_device(sample, self.device)
        self.input_example, *_ = sample

        # Calculate initial statistics
        self._calculate_stats(model, dataloader, "before", self.req_grad)

        # Perform pruning
        model = self.prune(model, dataloader=dataloader, **kwargs)

        # Calculate final statistics
        parametrized = remove_parametrizations(model)
        self._calculate_stats(model, dataloader, "after", False)
        reapply_parametrizations(model, parametrized, True)

        if self.verbose:
            self.print_stats()

        if len(self.pruning_record) > 0 and self.to_save_report:
            save_report(self.pruning_record, self.experiment_path, self.stats["ratio"])
        elif len(self.pruning_record) == 0 and self.verbose:
            print("No pruning was performed!")

        model.train(True)
        return model

    def print_stats(self) -> None:
        """Print pruning statistics."""
        stats_lines = [
            f"Loss before: {self.stats['loss_before']:.8f}",
            f"Loss after: {self.stats['loss_after']:.8f}",
            f"Total parameters before: {(self.stats['total_params_before'] / 1e6):.4f}M",
            f"Total parameters after: {(self.stats['total_params_after'] / 1e6):.4f}M",
            f"FLOPs: {self.stats['flops_before']:.5f}G -> {self.stats['flops_after']:.5f}G",
            f"Compression ratio: {self.stats['ratio']:.4f}x",
            f"Compression rate: {self.stats['compression']:.4f}",
        ]
        print("\n".join(stats_lines))

    @property
    def get_last_stats(self) -> dict:
        """Get the most recent pruning statistics.

        Returns:
            Dictionary containing pruning statistics.
        """
        return self.stats

    def __repr__(self) -> str:
        """Generate string representation of the pruner.

        Returns:
            YAML-formatted string of pruner attributes.
        """
        attrs = {k: str(v) for k, v in vars(self).items() if not callable(v) and not k.startswith("_")}
        return "\n".join(
            [
                self.__class__.__name__,
                "Attributes:",
                yaml.dump(attrs),
            ]
        )
