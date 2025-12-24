from typing import Optional


def get_experiment_tags(tags: Optional[dict[str, str]]):
    from mlflow.tracking.context import registry as context_registry

    return context_registry.resolve_tags(tags)
