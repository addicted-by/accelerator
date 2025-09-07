# Getting Started

This guide walks you through installing Accelerator and running your first experiment.

## Installation

Clone the repository and install the package in editable mode with development tools:

```bash
pip install -e .[dev]
```

After installation, enable the pre-commit hooks to keep formatting consistent:

```bash
pre-commit install
```

## Run an Example Experiment

The repository includes a CIFAR-10 training example. Launch it with:

```bash
python -m accelerator.cli.experiment experiment=cifar10_resnet18
```

Hydra composes the configuration from the `configs/` directory, then the training loop logs outputs to `outputs/`.

## Override Configuration Values

Any configuration option can be overridden from the command line. For example, to change the learning rate:

```bash
python -m accelerator.cli.experiment experiment=cifar10_resnet18 optimizer.lr=0.01
```

This flexibility lets you iterate quickly without editing YAML files.

## Next Steps

- Read the [architecture overview](architecture.md) to understand the project's structure.
- Dive into component docs such as `context.md` and `callbacks.md`.
- Explore the `examples/` directory for more end-to-end setups.
