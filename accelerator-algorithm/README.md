# Accelerator Algorithm

Model acceleration algorithms and optimization techniques for the accelerator framework.

## Overview

This package provides acceleration algorithms including:
- Pruning techniques for model compression
- Optimization strategies for model efficiency
- Registry system for acceleration methods

## Installation

```bash
pip install accelerator-algorithm
```

## Usage

```python
from accelerator.algorithm.pruning import MagnitudePruner
from accelerator.algorithm.optimization import ModelOptimizer
from accelerator.algorithm.registry import AccelerationRegistry
```

## Dependencies

- accelerator-core: Core runtime components
- accelerator-utilities: Common utilities
- torch: PyTorch framework