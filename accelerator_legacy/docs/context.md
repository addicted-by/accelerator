# Context

The **Context** object is the central coordination point of the Accelerator
framework.  It stores configuration, components and training state so that all
parts of an experiment can communicate through a single interface.

## Responsibilities

The context orchestrates several key pieces:

* **Configuration** – accepts an OmegaConf `DictConfig` and exposes it to every
  component.
* **Component management** – wraps a :class:`~accelerator.runtime.context.component.ComponentManager`
  that registers and retrieves objects such as the model, data module, optimizer
  and callbacks.
* **Training state** – owns a
  :class:`~accelerator.runtime.context.training.TrainingManager` which tracks
  epochs, global step and other stateful information.
* **Distributed setup** – can attach a distributed backend and prepare models,
  optimizers and dataloaders for multi‑process training.
* **Initialization and cleanup** – prints configuration, materializes
  components, optionally loads checkpoints and tears everything down at the end
  of a run.

## Basic Usage

```python
from omegaconf import OmegaConf
from accelerator.runtime.context.context import Context

cfg = OmegaConf.load("config.yaml")

ctx = Context(cfg)
ctx.initialize()           # build components and optionally load checkpoints
ctx.setup_engine()         # enable distributed backend if configured
ctx.make_distributed()     # wrap model/optimizer/dataloaders for distributed

model = ctx.model          # access registered components
optimizer = ctx.optimizer
train_loader = ctx.data.train_loader

# ... training loop ...

ctx.cleanup()              # free distributed resources and reset state
```

## Component Accessors

Components are retrieved through properties.  Accessing `ctx.model` for
instance will instantiate the model if it has not been created already and
register it with the component manager.  Similar helpers exist for the data
module, optimizer, scheduler and callback manager.

## Distributed Training

If a distributed backend is configured, `setup_engine` establishes the backend
and `make_distributed` prepares the model, dataloaders and optimizer for
distributed execution.  The context keeps track of whether the current process
is part of a distributed run via the `is_distributed` flag.

## Checkpointing and Resuming

Calling `initialize` with a checkpoint path will load model weights and training
state via the context's `CheckpointManager`.  This allows experiments to resume
from previous runs with minimal boilerplate.

## Cleanup

After training, `cleanup` should be invoked to free distributed resources and
reset internal managers so that the context can be re-used in subsequent runs.

---

For an architectural overview of how the context fits into the wider framework
see [`architecture.md`](architecture.md).

