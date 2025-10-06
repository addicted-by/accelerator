"""Optional patch modules for external integrations."""

try:
    from . import mlflow  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    mlflow = None

__all__ = ['mlflow'] if mlflow is not None else []

