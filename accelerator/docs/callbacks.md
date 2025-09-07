# Callbacks

Callbacks let you inject custom behavior at key points of the training and
evaluation lifecycle. Each callback implements hook methods such as
`on_train_begin`, `on_train_epoch_end`, `on_backward_begin`, or
`on_optimizer_step_end`. A :class:`~accelerator.runtime.callbacks.manager.CallbackManager`
stored on the :class:`~accelerator.runtime.context.context.Context` owns the
active callbacks and triggers their hooks.

## Default behaviour

Unlike frameworks such as PyTorch Lightning, Accelerator ships with a small set
of callbacks that are always active. They handle bookkeeping and acceleration
concerns so that you can start training without specifying any callbacks at
all. The default set includes:

- `StepEpochTrackerCallback` – keeps global step and epoch counters in sync.
- `TimeTrackingCallback` – measures elapsed and remaining time.
- A progress bar callback – renders either a `tqdm` or `rich` progress bar
  when enabled.

These callbacks are added automatically when the context is initialized. Custom
callbacks can be appended through configuration, but omitting them entirely is a
perfectly valid setup.

## Invoking callbacks

Training loops (or any custom code) should wrap logical phases with the
manager's :meth:`phase` context manager. Entering the context fires a
`<phase>_begin` hook on every callback, and leaving it fires
`<phase>_end`.

```python
callbacks = ctx.callbacks
with callbacks.phase("train", ctx):
    for epoch in range(num_epochs):
        with callbacks.phase("train_epoch", ctx):
            for batch in train_loader:
                with callbacks.phase("train_batch", ctx):
                    # forward pass, loss computation ...
                    with callbacks.phase("backward", ctx):
                        loss.backward()
                    with callbacks.phase("optimizer_step", ctx):
                        optimizer.step()
```

This explicit wrapping keeps the training loop transparent while still enabling
callbacks to modify behaviour. You can also manually trigger a single event
via `ctx.callbacks.trigger("train_begin", ctx)` if needed.

## Adding custom callbacks

To extend the runtime, subclass
`accelerator.runtime.callbacks.base.BaseCallback` and override the hooks you
care about:

```python
from accelerator.runtime.callbacks.base import BaseCallback

class PrintEpochEnd(BaseCallback):
    def on_train_epoch_end(self, context):
        print(f"Epoch {context.training.epoch} finished")
```

Register the callback in the configuration so that the context can instantiate
it:

```yaml
callbacks:
  active_callbacks: [print_epoch]
  print_epoch:
    _target_: path.to.PrintEpochEnd
    priority: 80 # lower numbers run first
```

Callbacks are ordered by their `priority` attribute. If a callback is marked
as `critical` and raises an exception, training will stop immediately.

## Differences from PyTorch Lightning

PyTorch Lightning expects users to either subclass `LightningModule` hooks or
pass explicit callback instances to the `Trainer`. In Accelerator, most common
operations are implemented as always-on callbacks and invoked through context
managers. You only define extra callbacks when you need additional side
effects, keeping the core loop simple and under your control.

## Progress Bars

`accelerator` can provide a progress bar callback so that training steps offer
immediate feedback. No progress bar is shown by default, but different progress
implementations can be enabled through the global configuration.

### Selecting a Progress Bar

The choice of progress bar is controlled via the `progress_bar` field in the
configuration (see `configs/main.yaml`). Supported values are:

- `"tqdm"` – enable the tqdm-based progress bar.
- `"rich"` – use the rich-library based progress bar.
- `null` – disable the progress bar entirely (default).

For example, to enable the rich progress bar:

```yaml
progress_bar: rich
```

To disable any progress display:

```yaml
progress_bar: null
```

User-provided callbacks can still be added through the standard `callbacks`
configuration and will be appended to the always-on set.

### Timing Metrics

When time tracking is enabled, the training metrics include `eta` (estimated
time remaining) and `elapsed` (total time spent). These fields are
automatically logged by all logger callbacks and shown on the active progress
bar whenever they are available.
