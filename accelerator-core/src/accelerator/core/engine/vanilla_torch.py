import os
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerator.utilities.default_config import _DefaultConfig, dataclass
from accelerator.utilities.distributed_state import distributed_state
from accelerator.utilities.logging import get_logger
from accelerator.utilities.typings import ConfigType

from .base import DistributedBackend

logger = get_logger(__name__)


@dataclass
class DDPEngineDefaults(_DefaultConfig):
    backend: str = "nccl"
    auto_spawn: bool = False
    init_method: str = "env://"
    find_unused_parameters: bool = False
    num_gpus: int = torch.cuda.device_count()
    master_addr: str = "localhost"
    master_port: str = "12355"


class DDPEngine(DistributedBackend):
    def __init__(self, config: Optional[ConfigType] = None):
        super().__init__(config, default_config=DDPEngineDefaults)
        self._world_size = None
        self._rank = None
        self._local_rank = None

        self.backend = self.config.get("backend")
        self.auto_spawn = self.config.get("auto_spawn")
        self.num_gpus = self.config.get("num_gpus")
        self.master_addr = self.config.get("master_addr")
        self.master_port = self.config.get("master_port")
        self.init_method = self.config.get("init_method")

    @property
    def device(self) -> torch.DeviceObjType:
        return torch.device(f"cuda:{self._rank}") if self._rank is not None else torch.device("cpu")

    def rank(self) -> int:
        return self._rank if self._rank is not None else 0

    @property
    def world_size(self) -> int:
        return self._world_size if self._world_size is not None else 1

    def setup(self) -> None:
        if not self.auto_spawn:
            self._spawn_processes()
        else:
            self._init_process_group()

    def _spawn_processes(self) -> None:
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port

        mp.spawn(self._distributed_worker, args=(self.num_gpus,), nprocs=self.num_gpus, join=True)

    def _distributed_worker(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        self._local_rank = rank
        self._init_process_group()

    def _init_process_group(self) -> None:
        if self._rank is None:
            self._rank = int(os.environ.get("RANK", 0))
            self._world_size = int(os.environ.get("WORLD_SIZE", 1))
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend=self.backend, rank=self.rank(), world_size=self.world_size, init_method=self.init_method
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)

        logger.info(f"Initialized DDP: rank={self.rank()}, local_rank={self._local_rank}, world_size={self.world_size}")
        distributed_state.set_engine(self)

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        print("Model:", self._local_rank)
        if torch.cuda.is_available():
            model = model.cuda(self._local_rank)

        find_unused_parameters = self.config.get("find_unused_parameters", False)

        return DDP(
            model,
            device_ids=[self._local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=find_unused_parameters,
        )

    def prepare_dataloader(self, loader: DataLoader) -> DataLoader:
        if hasattr(loader, "dataset"):
            sampler = DistributedSampler(
                loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=getattr(loader, "shuffle", False)
            )

            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                sampler=sampler,
                num_workers=loader.num_workers,
                pin_memory=loader.pin_memory,
                drop_last=loader.drop_last,
                collate_fn=loader.collate_fn,
            )
        return loader

    def prepare_optimizer(self, optimizer) -> Any:
        return optimizer

    def is_main_process(self) -> bool:
        return self.rank() == 0

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor, op="mean") -> torch.Tensor:
        if not dist.is_initialized():
            return tensor

        if op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size()
        elif op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

        return tensor

    def gather(self, tensor) -> Any:
        if not dist.is_initialized():
            return [tensor]

        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size())]
        dist.all_gather(gathered_tensors, tensor)
        return gathered_tensors

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        if not dist.is_initialized():
            return obj

        if self.rank() == src:
            obj_list = [obj]
        else:
            obj_list = [None]

        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def spawn(self, fn: Callable, *args) -> None:
        if self.auto_spawn:
            mp.spawn(fn, args=args, nprocs=self.num_gpus, join=True)
        else:
            fn(*args)

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Cleaned up DDP process group")

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
