import inspect
from typing import Any, Callable


def get_documentation(obj: Any) -> str:
    """Extract and format documentation string from an object.

    Args:
        obj: The object to get documentation from

    Returns:
        Cleaned and truncated documentation string

    """
    try:
        doc = inspect.getdoc(obj)
        if not doc:
            return ""

        lines = doc.strip().split("\n")
        first_line = lines[0].strip()

        if len(first_line) > 80:
            first_line = first_line[:77] + "..."

        return first_line
    except (AttributeError, TypeError):
        return ""


def format_annotation(annotation: Any) -> str:
    """Format type annotations in a readable way."""
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    elif hasattr(annotation, "_name"):  # For generic types like List, Dict
        return str(annotation)
    else:
        return str(annotation)


def get_method_info(cls: type) -> list[str]:
    """Get formatted information about public methods of a class.

    Args:
        cls: The class to inspect

    Returns:
        List of formatted method information strings

    """
    methods = []
    try:
        for attr_name in dir(cls):
            if (
                not attr_name.startswith("_")
                and callable(getattr(cls, attr_name, None))
                and attr_name not in dir(object)
            ):
                try:
                    method = getattr(cls, attr_name)
                    if inspect.ismethod(method) or inspect.isfunction(method):
                        sig = inspect.signature(method)
                        method_params = []
                        for param_name, param in sig.parameters.items():
                            if param_name in ("self", "cls"):
                                continue
                            param_str = param_name
                            if param.annotation != inspect.Parameter.empty:
                                param_str += f": {format_annotation(param.annotation)}"
                            method_params.append(param_str)

                        method_info = f"{attr_name}({', '.join(method_params)})"
                        method_doc = get_documentation(method)
                        if method_doc:
                            method_info += f" [{method_doc}]"
                        methods.append(method_info)
                except (AttributeError, TypeError, ValueError):
                    methods.append(f"{attr_name}(...)")
    except (AttributeError, TypeError):
        pass

    return methods


def get_function_signature(func: Callable) -> str:
    """Get formatted function signature with parameters and return type.

    Args:
        func: The function to inspect

    Returns:
        Formatted signature string

    """
    try:
        sig = inspect.signature(func)
        params = []
        for param_name, param in sig.parameters.items():
            param_str = param_name
            if param.annotation != inspect.Parameter.empty:
                param_str += f": {format_annotation(param.annotation)}"
            if param.default != inspect.Parameter.empty:
                param_str += f" = {repr(param.default)}"
            params.append(param_str)

        signature = f"({', '.join(params)})"

        if sig.return_annotation != inspect.Parameter.empty:
            signature += f" -> {format_annotation(sig.return_annotation)}"

        return signature
    except (ValueError, TypeError):
        return "(<signature unavailable>)"


def get_class_signature(cls: type) -> str:
    """Get formatted class constructor signature.

    Args:
        cls: The class to inspect

    Returns:
        Formatted constructor signature string

    """
    try:
        init_sig = inspect.signature(cls.__init__)
        params = []
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            param_str = param_name
            if param.annotation != inspect.Parameter.empty:
                param_str += f": {format_annotation(param.annotation)}"
            if param.default != inspect.Parameter.empty:
                param_str += f" = {repr(param.default)}"
            params.append(param_str)

        if params:
            return f"({', '.join(params)})"
        else:
            return "()"
    except (ValueError, TypeError):
        return "(<signature unavailable>)"


def format_class_info(name: str, cls: type) -> str:
    """Format comprehensive information for a class object.

    Args:
        name: Name of the class
        cls: The class object

    Returns:
        Formatted class information string

    """
    info_parts = [f"{name} (class)"]

    info_parts.append(get_class_signature(cls))

    doc = get_documentation(cls)
    if doc:
        info_parts.append(f"\n      Doc: {doc}")

    methods = get_method_info(cls)
    if methods:
        info_parts.append("\n      Methods:")
        for method in methods:
            info_parts.append(f"\n        - {method}")

    return "".join(info_parts)


def format_function_info(name: str, func: Callable) -> str:
    """Format comprehensive information for a function object.

    Args:
        name: Name of the function
        func: The function object

    Returns:
        Formatted function information string

    """
    info_parts = [f"{name} (function)"]

    info_parts.append(get_function_signature(func))

    doc = get_documentation(func)
    if doc:
        info_parts.append(f"\n      Doc: {doc}")

    return "".join(info_parts)


def get_object_info(name: str, obj: Callable) -> str:
    """Get detailed information about a registered object.

    Args:
        name: Name of the object
        obj: The callable object (class or function)

    Returns:
        Formatted string with object information

    """
    try:
        if inspect.isclass(obj):
            return format_class_info(name, obj)
        else:
            return format_function_info(name, obj)
    except Exception as e:
        return f"{name}: <error getting info: {e}>"
