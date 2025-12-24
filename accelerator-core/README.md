# Accelerator Core

Core runtime components and framework infrastructure for the Accelerator ML framework.

## Contents

- **runtime/**: Core runtime components (engine, pipeline, context, callbacks, etc.)
- **hooks/**: PyTorch hooks for statistics collection
- **typings/**: Core type definitions and base abstractions

## Installation

```bash
pip install accelerator-core
```

## Usage

```python
from accelerator.core.runtime.engine import Engine
from accelerator.core.runtime.pipeline import Pipeline
from accelerator.core.hooks import HookRegistry
```

## Dependencies

- accelerator-utilities
- torch>=2.5.1
- hydra-core>=1.3.2
