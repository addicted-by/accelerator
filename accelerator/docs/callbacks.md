# Progress Bars

`accelerator` can provide a progress bar callback so that training steps offer
immediate feedback. No progress bar is shown by default, but different progress
implementations can be enabled through the global configuration.

## Selecting a Progress Bar

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
