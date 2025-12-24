import functools
import inspect
import warnings
from typing import Any, Callable, Optional, TypeVar, Union

F = TypeVar("F", bound=Union[Callable[..., Any], type[Any]])


def _append_to_docstring(obj: F, section_title: str, lines: list[str]) -> None:
    """Helper to format and append a section to an object's docstring."""
    if not lines:
        return

    if len(lines) > 1:
        content = "\n".join(f"    - {line}" for line in lines)
    else:
        content = f"    {lines[0]}"

    appendix = f"\n\n{section_title}:\n{content}"

    original_doc = inspect.getdoc(obj)
    if original_doc:
        obj.__doc__ = original_doc + appendix
    else:
        obj.__doc__ = appendix.lstrip()


class APIDesc:
    """
    Namespace for decorators that add descriptive metadata to API components (functions, classes).
    Metadata is typically appended to the object's docstring.
    """

    @staticmethod
    def developer(dev_info: Union[str, list[str], tuple[str, ...]]):
        if isinstance(dev_info, str):
            developers = [dev_info]
        elif isinstance(dev_info, (list, tuple)) and all(isinstance(d, str) for d in dev_info):
            developers = list(dev_info)
        else:
            raise TypeError("APIDesc.developer requires a string or a list/tuple of strings.")

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "Responsible Developer(s)", developers)
            return obj

        return decorator

    @staticmethod
    def status(status_level: str, details: Optional[str] = None, since: Optional[str] = None):
        """
        Decorator factory to add API status information (e.g., stable, experimental, internal).

        Args:
            status_level: The status (e.g., 'Stable', 'Experimental', 'Alpha', 'Beta', 'Internal Use Only').
            details: Optional additional details about the status.
            since: Optional version or date when this status became effective.
        """
        if not isinstance(status_level, str):
            raise TypeError("status_level must be a string.")

        lines = [status_level]
        if since:
            lines[0] += f" (Since: {since})"
        if details:
            lines.append(f"Details: {details}")

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "API Status", lines)
            return obj

        return decorator

    @staticmethod
    def since(version: str, description: Optional[str] = None):
        """
        Decorator factory to specify the version when the API was introduced.

        Args:
            version: The version string (e.g., 'v1.0.0', 'Release 2024-Q1').
            description: Optional description of what was introduced.
        """
        if not isinstance(version, str):
            raise TypeError("version must be a string.")

        lines = [f"Introduced in version: {version}"]
        if description:
            lines.append(description)

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "Availability", lines)
            return obj

        return decorator

    @staticmethod
    def deprecated(
        since: str,
        replacement: Optional[str] = None,
        removal_in: Optional[str] = None,
        message: str = "This API is deprecated and may be removed in a future version.",
    ):
        """
        Decorator factory to mark an API as deprecated. Also issues a DeprecationWarning.

        Args:
            since: Version or date when this API was deprecated.
            replacement: Suggestion for a replacement API (e.g., 'use new_function() instead').
            removal_in: Planned version or date for removal (e.g., 'v3.0.0', '2026-01-01').
            message: Custom deprecation message.
        """
        if not isinstance(since, str):
            raise TypeError("since must be a string.")

        lines = [f"Deprecated since: {since}"]
        if replacement:
            lines.append(f"Replacement: {replacement}")
        if removal_in:
            lines.append(f"Scheduled for removal in: {removal_in}")
        if message and message not in lines[0]:  # Avoid duplication if basic message used
            lines.append(message)

        full_warning_message = f"{message} (Deprecated since {since})"
        if replacement:
            full_warning_message += f" Consider using {replacement}."
        if removal_in:
            full_warning_message += f" It may be removed in {removal_in}."

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "Deprecation Notice", lines)

            if callable(obj):

                @functools.wraps(obj)
                def wrapper(*args, **kwargs):
                    warnings.warn(full_warning_message, DeprecationWarning, stacklevel=2)
                    return obj(*args, **kwargs)

                return wrapper
            else:
                # For classes, we can't easily wrap __init__ or all methods this way.
                # Docstring modification is the primary effect for classes.
                # A more complex implementation could wrap __init__.
                pass

            return obj

        return decorator

    @staticmethod
    def see_also(references: Union[str, list[str], tuple[str, ...]]):
        """
        Decorator factory to add links or references to related documentation, tickets, or APIs.

        Args:
            references: A string or list/tuple of strings representing URLs, ticket IDs,
                        or related function/class names.
        """
        if isinstance(references, str):
            refs = [references]
        elif isinstance(references, (list, tuple)) and all(isinstance(r, str) for r in references):
            refs = list(references)
        else:
            raise TypeError("APIDesc.see_also requires a string or a list/tuple of strings.")

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "See Also", refs)
            return obj

        return decorator

    @staticmethod
    def warning(message: str):
        """
        Decorator factory to add a usage warning or important note.

        Args:
            message: The warning message text.
        """
        if not isinstance(message, str):
            raise TypeError("message must be a string.")

        def decorator(obj: F) -> F:
            _append_to_docstring(obj, "Warning", [message])
            return obj

        return decorator
