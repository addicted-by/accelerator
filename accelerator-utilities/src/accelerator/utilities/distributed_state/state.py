import os
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.distributed as dist

from accelerator.utilities.api_desc import APIDesc
from accelerator.utilities.typings import _DEVICE

if TYPE_CHECKING:
    from accelerator.core.engine import DistributedBackend


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="Stable")
@APIDesc.see_also(
    [
        "https://pytorch-lightning.readthedocs.io/en/1.7.7/api/pytorch_lightning.utilities.rank_zero.html",
        "https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/rank_zero.html#rank_zero_only",
    ]
)
class DistributedState:
    """
    Singleton class that provides distributed training utilities and state management.

    This class acts as a global access point for distributed operations, providing
    decorators and context managers for distributed workflows

    When no distributed engine is available, falls back to environment variable
    detection for rank information (RANK, LOCAL_RANK, SLURM_PROCID, JSM_NAMESPACE_RANK).
    """

    _instance: Optional["DistributedState"] = None
    _lock = Lock()

    def __new__(cls) -> "DistributedState":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            self._engine: Optional[DistributedBackend] = None
            self._initialized = True
            self._cpu_execution = False
            self._device_set = None

    def _get_rank_from_env(self) -> int:
        # torchrun/accelerate launch/mp launch/if mp spawn set os env
        rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
        for key in rank_keys:
            rank = os.environ.get(key)
            if rank is not None:
                return int(rank)

        # if mp spawn not set env
        if dist.is_initialized():
            return dist.get_rank()
        return 0

    def _get_world_size_from_env(self) -> int:
        world_size_keys = ("WORLD_SIZE", "SLURM_NTASKS", "JSM_NAMESPACE_SIZE")
        for key in world_size_keys:
            world_size = os.environ.get(key)
            if world_size is not None:
                return int(world_size)
        return 1

    def set_device(self, device: _DEVICE):
        self.reset()
        self._device_set = torch.device(device)

    @property
    def cpu_execution(self) -> bool:
        return self._cpu_execution

    @cpu_execution.setter
    def cpu_execution(self, flag: bool) -> None:
        self._cpu_execution = bool(flag)
        if self._cpu_execution:
            self.set_device("cpu")

    @APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
    @APIDesc.status(status_level="Experimental")
    @property
    def device(self) -> torch.DeviceObjType:
        if self._device_set:
            return self._device_set

        if self._engine is not None and hasattr(self._engine, "device"):
            return self._engine.device

        return torch.device(f"cuda:{self.rank}")

    def set_engine(self, engine: Optional["DistributedBackend"]) -> None:
        self._engine = engine

    def reset(self) -> None:
        self._engine = None

    @property
    def is_distributed(self) -> bool:
        return self._engine is not None or self.world_size > 1

    @property
    def rank(self) -> int:
        if self._engine:
            return self._engine.rank()
        return self._get_rank_from_env()

    @property
    def world_size(self) -> int:
        if self._engine:
            return self._engine.world_size()
        return self._get_world_size_from_env()

    @property
    def is_main_process(self) -> bool:
        if self._engine:
            return self._engine.is_main_process()
        return self.rank == 0

    def barrier(self) -> None:
        if self._engine:
            self._engine.barrier()

        elif dist.is_initialized():
            torch.distributed.barrier()

    def on_main_process(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator that only runs the decorated function on the main process.

        Args:
            func: The function to decorate

        Returns:
            The original function if on main process, else a no-op function
        """
        if self.is_main_process:
            return func
        return lambda *args, **kwargs: None

    def on_rank(self, rank: int):
        """
        Decorator that only runs the decorated function on the specified rank.

        Args:
            rank: The rank on which to run the function

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if self.rank == rank:
                return func
            return lambda *args, **kwargs: None

        return decorator

    def on_last_process(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator that only runs the decorated function on the last process.

        Args:
            func: The function to decorate

        Returns:
            The original function if on last process, else a no-op function
        """
        if self.rank == self.world_size - 1:
            return func
        return lambda *args, **kwargs: None

    @contextmanager
    def main_process_first(self):
        """
        Context manager where the main process goes first, others wait.

        Example:
            with distributed_state.main_process_first():
                # Main process executes first
                download_dataset()
            # All processes continue together
        """
        if not self.is_main_process:
            self.barrier()

        yield

        if self.is_main_process:
            self.barrier()

    @contextmanager
    def rank_first(self, rank: int):
        """
        Context manager where the specified rank goes first, others wait.

        Args:
            rank: The rank that should go first
        """
        if self._engine:
            if self.rank != rank:
                self.barrier()

            yield

            if self.rank == rank:
                self.barrier()
        else:
            # Without engine, can't synchronize, so just execute
            yield

    def all_reduce(self, tensor, op: str = "mean"):
        """
        All-reduce operation on a tensor.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('mean', 'sum')

        Returns:
            Reduced tensor (unchanged if no engine available)
        """
        if self._engine:
            return self._engine.all_reduce(tensor, op)
        return tensor

    def gather(self, tensor):
        """
        Gather tensor from all processes.

        Args:
            tensor: Tensor to gather

        Returns:
            Gathered tensors from all processes (single tensor list if no engine)
        """
        if self._engine:
            return self._engine.gather(tensor)
        return [tensor]

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        """
        Broadcast object from source rank to all processes.

        Args:
            obj: Object to broadcast
            src: Source rank

        Returns:
            Broadcasted object (unchanged if no engine available)
        """
        if self._engine:
            return self._engine.broadcast(obj, src)
        return obj

    def __repr__(self) -> str:
        engine_name = type(self._engine).__name__ if self._engine else "env_fallback"
        return (
            f"DistributedState(\n"
            f"  engine={engine_name},\n"
            f"  rank={self.rank},\n"
            f"  world_size={self.world_size},\n"
            f"  is_main_process={self.is_main_process}\n"
            f")"
        )


distributed_state = DistributedState()
