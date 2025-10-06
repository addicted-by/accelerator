from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from accelerator.core.context import Context as _Context
else:
    class _Context:  # type: ignore[revived-private-name]
        ...

# Core type variables
CONTEXT = TypeVar('CONTEXT', bound=_Context)