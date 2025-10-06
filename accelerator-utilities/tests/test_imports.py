"""Test that all utilities can be imported correctly."""

import pytest


def test_main_imports():
    """Test that main utilities can be imported."""
    from accelerator.utilities import (
        get_logger,
        set_log_file,
        distributed_state,
        instantiate,
        APIDesc,
        get_experiment_tags,
        move_data_to_device,
    )
    
    # Basic smoke test - just ensure imports work
    assert callable(get_logger)
    assert callable(set_log_file)
    assert callable(instantiate)
    assert callable(get_experiment_tags)
    assert callable(move_data_to_device)


def test_submodule_imports():
    """Test that submodules can be imported."""
    from accelerator.utilities.model_utils import names, unwrap
    from accelerator.utilities.hydra_utils import utils, callbacks
    from accelerator.utilities.distributed_state import state
    from accelerator.utilities.rich_utils import config_tree
    
    # Basic smoke test - just ensure imports work
    assert hasattr(names, '__file__')
    assert hasattr(utils, '__file__')
    assert hasattr(state, '__file__')
    assert hasattr(config_tree, '__file__')


if __name__ == "__main__":
    test_main_imports()
    test_submodule_imports()
    print("All import tests passed!")