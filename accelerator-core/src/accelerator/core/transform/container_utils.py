"""Utilities for accessing and modifying context containers using path-based notation.

This module provides utilities for extracting and setting values in the Context's
lifecycle containers using hierarchical path notation (e.g., 'per_batch.input.rgb').
"""

from typing import Any, Optional, Union

from accelerator.utilities.logging import get_logger

logger = get_logger(__name__)


class ContainerGetItem:
    """Utility for extracting values from context using hierarchical paths.

    This class supports two modes of operation:
    1. Positional arguments (args): List of paths to extract as positional arguments
    2. Keyword arguments (kwargs): Dict mapping parameter names to paths

    Both modes can be used simultaneously for maximum flexibility.

    Examples:
        >>> # Extract positional arguments
        >>> getter = ContainerGetItem(args=['per_batch.input.rgb', 'per_batch.target.label'])
        >>> args, kwargs = getter(context)
        >>> # args = [rgb_tensor, label_tensor], kwargs = {}

        >>> # Extract keyword arguments
        >>> getter = ContainerGetItem(kwargs={'image': 'per_batch.input.rgb', 'label': 'per_batch.target.label'})
        >>> args, kwargs = getter(context)
        >>> # args = [], kwargs = {'image': rgb_tensor, 'label': label_tensor}

        >>> # Mixed mode
        >>> getter = ContainerGetItem(
        ...     args=['per_batch.input.rgb'],
        ...     kwargs={'pca_mat': 'persistent.additional.pca_mat'}
        ... )
        >>> args, kwargs = getter(context)
        >>> # args = [rgb_tensor], kwargs = {'pca_mat': pca_matrix}

    """

    def __init__(self, args: Optional[list[str]] = None, kwargs: Optional[dict[str, str]] = None):
        """Initialize the ContainerGetItem utility.

        Args:
            args: List of paths to extract as positional arguments.
                  Each path should be in format 'lifecycle_scope.sub_container.key'.
            kwargs: Dict mapping parameter names to paths. Keys are the parameter
                   names to use, values are the paths to extract from.

        Note:
            Both args and kwargs can be empty for passthrough transforms.

        """
        self.args_paths = args or []
        self.kwargs_paths = kwargs or {}

    def __call__(self, context) -> tuple[list[Any], dict[str, Any]]:
        """Extract values from context using configured paths.

        Args:
            context: The Context instance to extract values from.

        Returns:
            A tuple of (args, kwargs) where:
            - args is a list of values extracted using args_paths
            - kwargs is a dict of values extracted using kwargs_paths

        Raises:
            ValueError: If a path cannot be resolved in the context.
            PathResolutionError: If a path format is invalid.

        Examples:
            >>> getter = ContainerGetItem(args=['per_batch.input.rgb'])
            >>> args, kwargs = getter(context)
            >>> rgb_tensor = args[0]

        """
        args = []
        kwargs = {}

        # Extract positional arguments
        for path in self.args_paths:
            try:
                value = context.get_item(path)
                args.append(value)
            except Exception as e:
                logger.error(f"Failed to extract value from path '{path}': {e}")
                raise ValueError(f"Failed to extract value from path '{path}': {e}") from e

        # Extract keyword arguments
        for param_name, path in self.kwargs_paths.items():
            try:
                value = context.get_item(path)
                kwargs[param_name] = value
            except Exception as e:
                logger.error(f"Failed to extract value from path '{path}' for parameter '{param_name}': {e}")
                raise ValueError(f"Failed to extract value from path '{path}' for parameter '{param_name}': {e}") from e

        return args, kwargs

    def __repr__(self) -> str:
        """Return string representation of ContainerGetItem."""
        return f"ContainerGetItem(args={self.args_paths}, kwargs={self.kwargs_paths})"


class ContainerSetItem:
    """Utility for setting values in context using hierarchical paths.

    This class supports setting single or multiple values in the context's
    lifecycle containers using path-based notation.

    Examples:
        >>> # Set single value
        >>> setter = ContainerSetItem('per_step.loss.total')
        >>> setter(context, 0.5)

        >>> # Set multiple values
        >>> setter = ContainerSetItem(['per_step.prediction.output', 'per_step.loss.total'])
        >>> setter(context, [output_tensor, 0.5])

    """

    def __init__(self, items: Union[str, list[str]]):
        """Initialize the ContainerSetItem utility.

        Args:
            items: Single path string or list of paths where values should be set.
                  Each path should be in format 'lifecycle_scope.sub_container.key'.

        Examples:
            >>> setter = ContainerSetItem('per_batch.prediction.output')
            >>> setter = ContainerSetItem(['per_step.loss.ce', 'per_step.loss.reg'])

        """
        self.items = [items] if isinstance(items, str) else items

        if not self.items:
            raise ValueError("At least one item path must be provided")

    def __call__(self, context, values: Union[Any, list[Any], tuple[Any, ...]], use_weakref: Optional[bool] = None):
        """Set values in context at configured paths.

        Args:
            context: The Context instance to set values in.
            values: Single value or list/tuple of values to set. If multiple paths
                   are configured, values must be a list/tuple with matching length.
            use_weakref: If True, force weak references; if False, force strong references;
                        if None, use automatic detection. Applied to all values.

        Returns:
            The context instance (for method chaining).

        Raises:
            ValueError: If the number of values doesn't match the number of paths.
            PathResolutionError: If a path format is invalid.

        Examples:
            >>> setter = ContainerSetItem('per_step.loss.total')
            >>> setter(context, 0.5)

            >>> setter = ContainerSetItem(['per_step.loss.ce', 'per_step.loss.reg'])
            >>> setter(context, [0.3, 0.2])

        """
        # Normalize values to list
        if not isinstance(values, (list, tuple)):
            values = [values]

        # Validate number of values matches number of paths
        if len(values) != len(self.items):
            raise ValueError(
                f"Number of values ({len(values)}) doesn't match "
                f"number of items ({len(self.items)}). "
                f"Expected {len(self.items)} values for paths: {self.items}"
            )

        # Set each value at its corresponding path
        for path, value in zip(self.items, values):
            try:
                context.set_item(path, value, use_weakref=use_weakref)
            except Exception as e:
                logger.error(f"Failed to set value at path '{path}': {e}")
                raise ValueError(f"Failed to set value at path '{path}': {e}") from e

        return context

    def __repr__(self) -> str:
        """Return string representation of ContainerSetItem."""
        return f"ContainerSetItem(items={self.items})"
