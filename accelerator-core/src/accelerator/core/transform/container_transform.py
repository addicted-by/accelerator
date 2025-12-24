"""Transform that updates context containers based on configuration.

This module provides the ContainerUpdateTransform class which orchestrates
the extraction of inputs from context, application of transforms, and setting
of outputs back to context using hierarchical path notation.
"""

import copy
from typing import Any, Optional, Union

from accelerator.utilities.logging import get_logger

from .container_utils import ContainerGetItem, ContainerSetItem

logger = get_logger(__name__)


class ContainerCondition:
    """Evaluates conditions on context values to determine if transform should execute.

    Supports various comparison operators and logical combinations (AND/OR).

    Supported operators:
    - equal: Value equals expected value
    - not_equal: Value does not equal expected value
    - greater_than: Value is greater than expected value
    - less_than: Value is less than expected value
    - greater_equal: Value is greater than or equal to expected value
    - less_equal: Value is less than or equal to expected value
    - in: Value is in expected collection
    - not_in: Value is not in expected collection

    Logical operators:
    - AND: All conditions must be true (default)
    - OR: At least one condition must be true

    Examples:
        >>> # Simple condition
        >>> condition = ContainerCondition({
        ...     'path': 'persistent.config.enabled',
        ...     'operator': 'equal',
        ...     'value': True
        ... })

        >>> # Multiple conditions with AND logic
        >>> condition = ContainerCondition({
        ...     'logic': 'AND',
        ...     'conditions': [
        ...         {'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True},
        ...         {'path': 'per_epoch.metrics.accuracy', 'operator': 'greater_than', 'value': 0.9}
        ...     ]
        ... })

        >>> # Multiple conditions with OR logic
        >>> condition = ContainerCondition({
        ...     'logic': 'OR',
        ...     'conditions': [
        ...         {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'train'},
        ...         {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'finetune'}
        ...     ]
        ... })
    """

    OPERATORS = {
        "equal": lambda a, b: a == b,
        "not_equal": lambda a, b: a != b,
        "greater_than": lambda a, b: a > b,
        "less_than": lambda a, b: a < b,
        "greater_equal": lambda a, b: a >= b,
        "less_equal": lambda a, b: a <= b,
        "in": lambda a, b: a in b,
        "not_in": lambda a, b: a not in b,
    }

    def __init__(self, condition_config: Optional[dict[str, Any]] = None):
        """Initialize condition evaluator.

        Args:
            condition_config: Configuration dict specifying conditions to evaluate.
                            Can be a single condition or multiple conditions with logic.
        """
        self.condition_config = condition_config or {}

        # Validate configuration
        if self.condition_config:
            if "conditions" in self.condition_config:
                # Multiple conditions
                if "logic" not in self.condition_config:
                    self.condition_config["logic"] = "AND"  # Default to AND
            elif "path" in self.condition_config:
                # Single condition - validate it has required fields
                if "operator" not in self.condition_config:
                    raise ValueError("Condition must specify 'operator'")
                if "value" not in self.condition_config:
                    raise ValueError("Condition must specify 'value'")

    def __call__(self, context) -> bool:
        """Evaluate condition on context.

        Args:
            context: The Context instance to evaluate condition on.

        Returns:
            True if condition is met, False otherwise.
        """
        if not self.condition_config:
            return True

        # Check if this is a multi-condition config
        if "conditions" in self.condition_config:
            return self._evaluate_multiple_conditions(context)
        else:
            return self._evaluate_single_condition(context, self.condition_config)

    def _evaluate_single_condition(self, context, condition: dict[str, Any]) -> bool:
        """Evaluate a single condition.

        Args:
            context: The Context instance to evaluate condition on.
            condition: Single condition configuration.

        Returns:
            True if condition is met, False otherwise.
        """
        path = condition.get("path")
        operator = condition.get("operator")
        expected_value = condition.get("value")

        if not path or not operator:
            logger.warning(f"Invalid condition config: {condition}")
            return True

        # Get actual value from context
        try:
            actual_value = context.get_item(path)
        except Exception as e:
            logger.warning(f"Failed to get value at path '{path}': {e}")
            return False

        # Get operator function
        if operator not in self.OPERATORS:
            logger.error(f"Unknown operator '{operator}'. Available: {list(self.OPERATORS.keys())}")
            return False

        operator_func = self.OPERATORS[operator]

        # Evaluate condition
        try:
            result = operator_func(actual_value, expected_value)
            logger.debug(f"Condition evaluation: {actual_value} {operator} {expected_value} = {result}")
            return result
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    def _evaluate_multiple_conditions(self, context) -> bool:
        """Evaluate multiple conditions with logical operators.

        Args:
            context: The Context instance to evaluate conditions on.

        Returns:
            True if combined condition is met, False otherwise.
        """
        conditions = self.condition_config.get("conditions", [])
        logic = self.condition_config.get("logic", "AND").upper()

        if not conditions:
            return True

        results = [self._evaluate_single_condition(context, cond) for cond in conditions]

        if logic == "AND":
            result = all(results)
        elif logic == "OR":
            result = any(results)
        else:
            logger.error(f"Unknown logic operator '{logic}'. Using AND.")
            result = all(results)

        logger.debug(f"Multiple conditions ({logic}): {results} = {result}")
        return result

    def __repr__(self) -> str:
        """Return string representation of ContainerCondition."""
        return f"ContainerCondition({self.condition_config})"


class ContainerPrinter:
    """Formats and prints context values for debugging.

    Supports configurable banners and formatting options.

    Configuration options:
    - paths: List of paths to print (if not specified, prints summary)
    - banner: Whether to print a banner (default: True)
    - banner_char: Character to use for banner (default: '=')
    - banner_width: Width of banner (default: 80)
    - max_items: Maximum number of items to print from collections (default: 10)
    - max_str_len: Maximum string length before truncation (default: 100)
    - indent: Indentation for nested structures (default: 2)

    Examples:
        >>> # Simple printer with banner
        >>> printer = ContainerPrinter('IN', True)

        >>> # Printer with custom configuration
        >>> printer = ContainerPrinter('OUT', {
        ...     'paths': ['per_step.loss.total', 'per_step.prediction.output'],
        ...     'banner': True,
        ...     'banner_char': '-',
        ...     'max_items': 5
        ... })
    """

    def __init__(self, label: str, print_config: Union[bool, dict[str, Any]]):
        """Initialize printer.

        Args:
            label: Label to use in output (e.g., 'IN' or 'OUT').
            print_config: Boolean or dict with printing configuration.
        """
        self.label = label

        # Parse configuration
        if isinstance(print_config, dict):
            self.print_config = print_config
        else:
            self.print_config = {}

        # Set defaults
        self.paths = self.print_config.get("paths", None)
        self.banner = self.print_config.get("banner", True)
        self.banner_char = self.print_config.get("banner_char", "=")
        self.banner_width = self.print_config.get("banner_width", 80)
        self.max_items = self.print_config.get("max_items", 10)
        self.max_str_len = self.print_config.get("max_str_len", 100)
        self.indent = self.print_config.get("indent", 2)

    def __call__(self, context) -> None:
        """Print context information.

        Args:
            context: The Context instance to print information from.
        """
        output_lines = []

        # Print banner
        if self.banner:
            banner_text = f" Transform {self.label} "
            padding = (self.banner_width - len(banner_text)) // 2
            banner = self.banner_char * padding + banner_text + self.banner_char * padding
            if len(banner) < self.banner_width:
                banner += self.banner_char * (self.banner_width - len(banner))
            output_lines.append(banner)

        # Print specified paths or summary
        if self.paths:
            for path in self.paths:
                try:
                    value = context.get_item(path)
                    formatted_value = self._format_value(value, indent_level=1)
                    output_lines.append(f"  {path}: {formatted_value}")
                except Exception as e:
                    output_lines.append(f"  {path}: <Error: {e}>")
        else:
            # Print summary of context
            output_lines.append("  Context Summary:")
            output_lines.append(f"    Type: {type(context).__name__}")

            # Try to print lifecycle containers if available
            if hasattr(context, "per_batch"):
                output_lines.append("    Lifecycle containers: per_batch, per_step, per_epoch, persistent")

        # Print closing banner
        if self.banner:
            output_lines.append(self.banner_char * self.banner_width)

        # Print all lines
        print("\n".join(output_lines))

    def _format_value(self, value: Any, indent_level: int = 0) -> str:
        """Format a value for printing.

        Args:
            value: The value to format.
            indent_level: Current indentation level.

        Returns:
            Formatted string representation of the value.
        """
        # indent = " " * (self.indent * indent_level)

        # Handle None
        if value is None:
            return "None"

        # Handle basic types
        if isinstance(value, (int, float, bool)):
            return str(value)

        # Handle strings
        if isinstance(value, str):
            if len(value) > self.max_str_len:
                return f"'{value[:self.max_str_len]}...' (truncated, length={len(value)})"
            return f"'{value}'"

        # Handle tensors (PyTorch)
        try:
            import torch

            if isinstance(value, torch.Tensor):
                shape_str = "x".join(map(str, value.shape))
                dtype_str = str(value.dtype).replace("torch.", "")
                device_str = str(value.device)
                return f"Tensor(shape={shape_str}, dtype={dtype_str}, device={device_str})"
        except ImportError:
            pass

        # Handle lists and tuples
        if isinstance(value, (list, tuple)):
            type_name = "list" if isinstance(value, list) else "tuple"
            if len(value) == 0:
                return f"{type_name}(empty)"
            elif len(value) <= self.max_items:
                items = [self._format_value(item, 0) for item in value]
                return f"{type_name}([{', '.join(items)}])"
            else:
                items = [self._format_value(item, 0) for item in value[: self.max_items]]
                return f"{type_name}([{', '.join(items)}, ... ({len(value)} items total)])"

        # Handle dictionaries
        if isinstance(value, dict):
            if len(value) == 0:
                return "dict(empty)"
            elif len(value) <= self.max_items:
                items = [f"{k}: {self._format_value(v, 0)}" for k, v in value.items()]
                return f"dict({{{', '.join(items)}}})"
            else:
                items = [f"{k}: {self._format_value(v, 0)}" for k, v in list(value.items())[: self.max_items]]
                return f"dict({{{', '.join(items)}, ... ({len(value)} items total)}})"

        # Handle other objects
        return f"{type(value).__name__}(...)"

    def __repr__(self) -> str:
        """Return string representation of ContainerPrinter."""
        return f"ContainerPrinter(label='{self.label}', config={self.print_config})"


class ContainerUpdateTransform:
    """Transform that updates containers based on configuration.

    This class orchestrates the complete transform pipeline:
    1. Extract inputs from context using paths
    2. Apply a registered transform to the inputs
    3. Set outputs back to context at specified paths
    4. Support conditional execution and debug printing

    Examples:
        >>> # Simple transform with single input/output
        >>> transform = ContainerUpdateTransform(
        ...     items='per_step.prediction.output',
        ...     trans_inputs=['per_batch.input.rgb'],
        ...     trans_opt={'type': 'Normalize', 'mean': 0.5, 'std': 0.5}
        ... )
        >>> context = transform(context)

        >>> # Transform with multiple inputs using kwargs
        >>> transform = ContainerUpdateTransform(
        ...     items='per_step.prediction.pca_output',
        ...     trans_inputs={
        ...         'kwargs': {
        ...             'input': 'per_batch.input.rgb',
        ...             'pca_mat': 'persistent.additional.pca_mat'
        ...         }
        ...     },
        ...     trans_opt={'type': 'PCATransform', 'n_components': 50}
        ... )

        >>> # Transform with condition
        >>> transform = ContainerUpdateTransform(
        ...     items='per_step.loss.total',
        ...     trans_inputs=['per_batch.prediction.output', 'per_batch.target.label'],
        ...     trans_opt={'type': 'CrossEntropyLoss'},
        ...     condition={'path': 'persistent.config.use_loss', 'operator': 'equal', 'value': True}
        ... )
    """

    def __init__(
        self,
        items: Optional[Union[str, list[str]]] = None,
        trans_opt: Optional[dict[str, Any]] = None,
        trans_inputs: Optional[Union[list[str], dict[str, Any]]] = None,
        condition: Optional[dict[str, Any]] = None,
        print_in: Union[bool, dict[str, Any]] = False,
        print_out: Union[bool, dict[str, Any]] = False,
        copy_context: bool = False,
    ):
        """Initialize ContainerUpdateTransform.

        Args:
            items: Output path(s) where transform results should be stored.
                  Can be a single path string or list of paths.
                  Example: 'per_step.prediction.output' or
                          ['per_step.loss.ce', 'per_step.loss.reg']

            trans_opt: Configuration dict for the transform to apply.
                      Must include 'type' key with transform class name.
                      Additional keys are passed as kwargs to transform constructor.
                      Example: {'type': 'PCATransform', 'n_components': 50}
                      If None, no transform is applied (passthrough mode).

            trans_inputs: Specification of input paths to extract from context.
                         Supports two formats:
                         1. List of paths (extracted as positional args):
                            ['per_batch.input.rgb', 'per_batch.target.label']
                         2. Dict with 'args' and/or 'kwargs' keys:
                            {
                                'args': ['per_batch.input.rgb'],
                                'kwargs': {'pca_mat': 'persistent.additional.pca_mat'}
                            }

            condition: Optional condition configuration for conditional execution.
                      If provided, transform only executes when condition is met.
                      Example: {'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True}

            print_in: If True or dict, print input values before transform.
                     Dict can specify formatting options.

            print_out: If True or dict, print output values after transform.
                      Dict can specify formatting options.

            copy_context: If True, deep copy context before modification.
                         Useful for debugging or when context should remain unchanged.

        Raises:
            ValueError: If trans_opt is provided but missing 'type' key.
            ValueError: If trans_inputs format is invalid.
        """
        # Store output paths
        self.items = items
        self.trans_opt_original = trans_opt.copy() if trans_opt else None

        # Parse and validate trans_inputs
        if trans_inputs is None:
            trans_inputs = []

        if isinstance(trans_inputs, dict):
            # Check if it's the structured format with 'args' and/or 'kwargs'
            if "args" in trans_inputs or "kwargs" in trans_inputs:
                self.input_getter = ContainerGetItem(args=trans_inputs.get("args"), kwargs=trans_inputs.get("kwargs"))
            else:
                # Treat as kwargs mapping (backward compatibility)
                self.input_getter = ContainerGetItem(kwargs=trans_inputs)
        elif isinstance(trans_inputs, list):
            # List of paths - extract as positional args
            self.input_getter = ContainerGetItem(args=trans_inputs)
        else:
            raise ValueError(f"trans_inputs must be a list or dict, got {type(trans_inputs)}")

        # Initialize transform from registry
        self.transform = None
        if trans_opt is not None:
            if "type" not in trans_opt:
                raise ValueError("trans_opt must include 'type' key specifying transform class")

            # Import registry here to avoid circular imports
            try:
                from accelerator.core.transform.registry import transforms_registry

                # Make a copy to avoid modifying the original config
                trans_opt_copy = trans_opt.copy()
                trans_type = trans_opt_copy.pop("type")

                # Get or create transform instance from registry
                self.transform = transforms_registry.get_or_create_instance(trans_type, trans_opt_copy)
                logger.debug(f"Initialized transform '{trans_type}' with config: {trans_opt_copy}")
            except ImportError as e:
                logger.warning(f"Could not import transforms_registry: {e}. Transform will be None.")
            except KeyError as e:
                # Transform not found in registry - re-raise for proper error handling
                logger.error(f"Transform not found in registry: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize transform from registry: {e}")
                raise

        # Initialize output setter
        self.output_setter = None
        if items is not None:
            self.output_setter = ContainerSetItem(items)

        # Initialize condition evaluator
        self.condition = ContainerCondition(condition) if condition else None

        # Initialize printers for debugging
        self.print_in = self._create_printer("IN", print_in) if print_in else None
        self.print_out = self._create_printer("OUT", print_out) if print_out else None

        # Store copy flag
        self.copy_context = copy_context

        logger.debug(
            f"Initialized ContainerUpdateTransform: "
            f"items={items}, trans_opt={trans_opt}, trans_inputs={trans_inputs}"
        )

    def _create_printer(self, label: str, print_config: Union[bool, dict[str, Any]]) -> ContainerPrinter:
        """Create a ContainerPrinter instance.

        Args:
            label: Label for the printer ('IN' or 'OUT').
            print_config: Boolean or dict with printer configuration.

        Returns:
            ContainerPrinter instance.
        """
        return ContainerPrinter(label, print_config)

    def __call__(self, context) -> Any:
        """Execute the transform on the context.

        This method orchestrates the complete transform pipeline:
        1. Print input values if print_in is enabled
        2. Check condition if specified
        3. Copy context if copy_context is True
        4. Extract inputs using ContainerGetItem
        5. Apply transform to inputs
        6. Set outputs using ContainerSetItem
        7. Print output values if print_out is enabled

        Args:
            context: The Context instance to operate on.

        Returns:
            The modified context (or copy if copy_context=True).

        Raises:
            RuntimeError: If transform execution fails.
        """
        # Step 1: Print input values if enabled
        if self.print_in:
            self.print_in(context)

        # Step 2: Check condition - skip execution if condition not met
        if self.condition and not self.condition(context):
            logger.debug("Condition not met, skipping transform execution")
            return context

        # Step 3: Copy context if requested
        if self.copy_context:
            context = copy.deepcopy(context)
            logger.debug("Created deep copy of context")

        # Step 4: Extract inputs from context
        try:
            args, kwargs = self.input_getter(context)
            logger.debug(f"Extracted {len(args)} args and {len(kwargs)} kwargs from context")
        except Exception as e:
            logger.error(f"Failed to extract inputs from context: {e}")
            raise RuntimeError(f"Failed to extract inputs from context: {e}") from e

        # Step 5: Apply transform to inputs
        if self.transform:
            try:
                result = self.transform(*args, **kwargs)
                logger.debug("Transform executed successfully")
            except Exception as e:
                logger.error(f"Transform execution failed: {e}")
                raise RuntimeError(f"Transform execution failed with config {self.trans_opt_original}: {e}") from e
        else:
            # Passthrough mode - no transform applied
            if len(args) == 1 and not kwargs:
                result = args[0]
            elif args and not kwargs:
                result = args
            elif kwargs and not args:
                result = kwargs
            else:
                result = (args, kwargs)
            logger.debug("No transform specified, using passthrough mode")

        # Step 6: Set outputs in context
        if self.output_setter:
            try:
                context = self.output_setter(context, result)
                logger.debug(f"Set outputs in context at paths: {self.items}")
            except Exception as e:
                logger.error(f"Failed to set outputs in context: {e}")
                raise RuntimeError(f"Failed to set outputs in context: {e}") from e

        # Step 7: Print output values if enabled
        if self.print_out:
            self.print_out(context)

        return context

    def __repr__(self) -> str:
        """Return string representation of ContainerUpdateTransform."""
        return (
            f"ContainerUpdateTransform("
            f"items={self.items}, "
            f"trans_opt={self.trans_opt_original}, "
            f"has_condition={self.condition is not None})"
        )
