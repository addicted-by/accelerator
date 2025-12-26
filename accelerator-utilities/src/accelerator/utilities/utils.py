import importlib

# from itertools import islice


def is_package_installed(package_name: str) -> bool:
    """Checks if a given package is installed.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.

    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


# def _chunked_indices(n_items: int, chunk: int):
#     it = iter(range(n_items))
#     while (sl := list(islice(it, chunk))):
# yield sl
