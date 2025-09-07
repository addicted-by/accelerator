# Accelerator Framework Architecture

This document provides a high-level and deep-dive overview of the Accelerator framework's architecture. The goal is to explain how the major pieces fit together so that new contributors and users can navigate the code base effectively.

## Package Layout

```
accelerator/
├── accelerator/        # Python package with runtime and domain code
├── configs/            # Hydra configuration files
├── examples/           # Example experiments
├── run_pipe.py         # Multi-step pipeline runner
└── tests/              # Unit tests
```

Most runtime logic lives under the top-level `accelerator` package. The repository is designed to be modular and extensible, with additional functionality grouped by domain.

## Runtime Core

At the heart of the framework is the **runtime** subsystem. The primary concepts are:

- **Context** – Central object that stores configuration, components, device information, and distributed state. It is created at the start of an experiment and passed through all stages of training and evaluation.
- **Components** – Pluggable objects such as models, optimizers, schedulers, losses, and callbacks. Components are registered via Python entry points and instantiated from configuration.
- **TrainLoop** – Drives epoch and batch iteration. It moves data to devices, computes losses, steps optimizers, and invokes callbacks around key events.

These pieces allow experiments to be constructed from decoupled components while sharing a common execution flow.

## Configuration System

The framework uses [Hydra](https://hydra.cc) for configuration management. Configuration files live under `configs/` and are composed at runtime. Key ideas include:

- **Defaults List** – Each config declares a `defaults` section describing which sub-configs to include.
- **Override Capability** – Values can be overridden from the command line or higher-level configs.
- **Index Files** – The `configs/` directory contains index YAML files that group related configurations for easier discovery.

Config-driven composition makes experiments reproducible and encourages separation between code and settings.

## Command Line Interface

A Fire-based CLI exposes high-level commands for interacting with the framework:

- `accelerator.cli.experiment` – entry point for running training or analysis tasks.
- `accelerator.cli.add` – scaffolds new components like models or losses.
- `accelerator.cli.configure` – utilities for initializing or cleaning configuration directories.

The CLI reduces boilerplate and helps users generate consistent project structures.

## Pipelines and MLflow Integration

Long-running experiments can be orchestrated through `run_pipe.py`. This script reads a pipeline configuration, executes each step in sequence, and tracks results with [MLflow](https://mlflow.org). The repository also includes an `MLproject` file to define entry points for `mlflow run`.

## Domain Extensions

Subpackages under `accelerator/domain/` provide domain-specific functionality (e.g., computer vision or NLP). Each domain can register custom components and callbacks while leveraging the core runtime.

## Development Tools

The project uses `pre-commit` with tools like Black, isort, Flake8, and Prettier to enforce code style. Tests are written with `pytest` and live under the `tests/` directory.

## Further Reading

- [`architecture_diagram.md`](architecture_diagram.md) – a mermaid diagram illustrating the relationships between major modules.
- `callbacks.md`, `context.md`, `cli.md`, etc. – planned deep-dives into specific subsystems.

This document should serve as a starting point for understanding the Accelerator codebase. Contributions to expand the remaining documentation pages are welcome.
