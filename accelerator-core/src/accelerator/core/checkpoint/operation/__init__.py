import importlib
from typing import Any

__all__ = [
    "ckpt_transforms",
    "load_ops",
    "pre_load_ops",
    "post_load_ops",
]

# Cache for lazy-loaded modules to avoid repeated imports
_module_cache = {}


def __getattr__(name: str) -> Any:
    """Lazy loading implementation for operation submodules.

    This function is called when an attribute is not found in the module's namespace.
    It handles lazy loading of ckpt_transforms, load_ops, pre_load_ops, and post_load_ops
    to break circular import dependencies.

    Args:
        name: The name of the attribute being accessed

    Returns:
        The requested module or attribute

    Raises:
        AttributeError: If the requested attribute is not available

    """
    if name in _module_cache:
        return _module_cache[name]

    try:
        if name == "ckpt_transforms":
            # Import the ckpt_transform module (note: directory name is ckpt_transform, not ckpt_transforms)
            module = importlib.import_module(".ckpt_transform", package=__name__)
            _module_cache[name] = module
            return module

        elif name == "load_ops":
            # Import the load_ops module
            module = importlib.import_module(".load_ops", package=__name__)
            _module_cache[name] = module
            return module

        elif name == "pre_load_ops":
            # Import pre_load_ops from the load_ops module
            load_ops_module = __getattr__("load_ops")  # Use lazy loading for load_ops too
            pre_load_ops = load_ops_module.pre_load_ops
            _module_cache[name] = pre_load_ops
            return pre_load_ops

        elif name == "post_load_ops":
            # Import post_load_ops from the load_ops module
            load_ops_module = __getattr__("load_ops")  # Use lazy loading for load_ops too
            post_load_ops = load_ops_module.post_load_ops
            _module_cache[name] = post_load_ops
            return post_load_ops

    except ImportError as e:
        raise AttributeError(
            f"Failed to import '{name}' from {__name__}. "
            f"Import error: {e}. "
            f"Available attributes: {', '.join(__all__)}"
        ) from e
    except Exception as e:
        raise AttributeError(
            f"Error accessing '{name}' from {__name__}: {e}. " f"Available attributes: {', '.join(__all__)}"
        ) from e

    # If we get here, the attribute is not one of our lazy-loaded modules
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
