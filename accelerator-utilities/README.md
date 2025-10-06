# Accelerator Utilities

Framework-agnostic utility functions and common helpers for the accelerator framework.

## Overview

This package provides the foundational utilities used across all accelerator subpackages. It includes:

- Distributed computing utilities
- Hydra configuration helpers
- Model utilities and helpers
- Logging utilities
- Device management utilities
- Common data structures and algorithms

## Installation

```bash
pip install accelerator-utilities
```

## Usage

```python
from accelerator.utilities import get_logger, distributed_state, instantiate
from accelerator.utilities.model_utils import count_parameters
from accelerator.utilities.hydra_utils import compose_config
```

## Dependencies

This is the foundation package with no dependencies on other accelerator subpackages.