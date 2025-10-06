from typing import Optional, Dict


def get_experiment_tags(tags: Optional[Dict[str, str]]):
    from mlflow.tracking.context import registry as context_registry
    return context_registry.resolve_tags(tags)