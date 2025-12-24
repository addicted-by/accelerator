from typing import Any, Optional


class LossAPIException(Exception):
    """Base exception for all loss API errors."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class LossConfigurationError(LossAPIException):
    """Raised when loss configuration is invalid."""

    pass


class LossCalculationError(LossAPIException):
    """Raised during loss calculation."""

    pass


class TransformationError(LossAPIException):
    """Raised during tensor transformation."""

    pass


class ValidationError(LossAPIException):
    """Raised during input validation."""

    pass
