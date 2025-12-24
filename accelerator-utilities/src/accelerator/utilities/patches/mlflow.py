"""
Custom tags:
- "mlflow.run_id": Custom run identifier
- "mlflow.artifact_location": Custom artifact storage location
"""
import pathlib
import re
import sys
import uuid

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local

from accelerator.utilities.logging import get_logger
from accelerator.utilities.utils import is_package_installed

log = get_logger(__name__)


def _extract_custom_run_values(tags):
    if not tags:
        return None, None, tags or []

    custom_run_id = None
    custom_artifact_location = None
    filtered_tags = []

    for tag in tags:
        if tag.key == "mlflow.run_id":
            custom_run_id = tag.value
        elif tag.key == "mlflow.artifact_location":
            custom_artifact_location = tag.value
        else:
            filtered_tags.append(tag)

    return custom_run_id, custom_artifact_location, filtered_tags


def _validate_custom_run_id(run_id, session):
    if not run_id:
        return

    if not re.match(r"^[a-z0-9]+$", run_id):
        raise MlflowException(
            f"Invalid run_id format: '{run_id}'. Run ID must contain only digits and lowercase letters.",
            INVALID_PARAMETER_VALUE,
        )

    from mlflow.store.tracking.dbmodels.models import SqlRun

    existing_run = session.query(SqlRun).filter(SqlRun.run_uuid == run_id).first()
    if existing_run:
        raise MlflowException(f"Run with id={run_id} already exists", RESOURCE_ALREADY_EXISTS)


def _validate_custom_artifact_location(artifact_location):
    if not artifact_location:
        return

    try:
        resolved_location = resolve_uri_if_local(artifact_location)
        if not resolved_location:
            raise MlflowException(f"Invalid artifact_location: '{artifact_location}'", INVALID_PARAMETER_VALUE)
    except Exception as e:
        raise MlflowException(
            f"Invalid artifact_location: '{artifact_location}'. Error: {str(e)}", INVALID_PARAMETER_VALUE
        ) from e


def patched_create_run(self, experiment_id, user_id, start_time, tags, run_name):
    """
    Patched version of SqlAlchemyStore.create_run that supports custom run_id and artifact_location.
    """
    with self.ManagedSessionMaker() as session:
        experiment = self.get_experiment(experiment_id)
        self._check_experiment_is_active(experiment)

        custom_run_id, custom_artifact_location, filtered_tags = _extract_custom_run_values(tags)

        if custom_run_id:
            _validate_custom_run_id(custom_run_id, session)
        if custom_artifact_location:
            _validate_custom_artifact_location(custom_artifact_location)

        if custom_run_id:
            run_id = custom_run_id
        else:
            run_id = uuid.uuid4().hex

        if custom_artifact_location:
            artifact_location = resolve_uri_if_local(custom_artifact_location)
        else:
            artifact_location = append_to_uri_path(experiment.artifact_location, run_id, self.ARTIFACTS_FOLDER_NAME)

        tags = filtered_tags or []

        from mlflow.entities import RunStatus, RunTag, SourceType
        from mlflow.entities.lifecycle_stage import LifecycleStage
        from mlflow.store.tracking.dbmodels.models import SqlRun, SqlTag
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, _get_run_name_from_tags
        from mlflow.utils.name_utils import _generate_random_name

        run_name_tag = _get_run_name_from_tags(tags)
        if run_name and run_name_tag and (run_name != run_name_tag):
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                INVALID_PARAMETER_VALUE,
            )
        run_name = run_name or run_name_tag or _generate_random_name()
        if not run_name_tag:
            tags.append(RunTag(key=MLFLOW_RUN_NAME, value=run_name))

        run = SqlRun(
            name=run_name,
            artifact_uri=artifact_location,
            run_uuid=run_id,
            experiment_id=experiment_id,
            source_type=SourceType.to_string(SourceType.UNKNOWN),
            source_name=sys.argv[0],  # set as sys.argv[0]
            entry_point_name=pathlib.Path(sys.argv[0]).stem,  # set as sys.argv[0].stem?
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=start_time,
            end_time=None,
            deleted_time=None,
            source_version="",  # commit id?
            lifecycle_stage=LifecycleStage.ACTIVE,
        )

        run.tags = [SqlTag(key=tag.key, value=tag.value) for tag in tags]
        session.add(run)

        return run.to_mlflow_entity()


def apply_patch():
    """
    Apply the patch to SqlAlchemyStore.create_run method.
    """
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    SqlAlchemyStore._original_create_run = SqlAlchemyStore.create_run

    SqlAlchemyStore.create_run = patched_create_run

    log.debug("MLflow SqlAlchemyStore.create_run method has been patched successfully!")
    log.debug("Custom tags supported:")
    log.debug("  - 'mlflow.run_id': Specify custom run identifier")
    log.debug("  - 'mlflow.artifact_location': Specify custom artifact storage location")


def remove_patch():
    """
    Remove the patch and restore original create_run method.
    """
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    if hasattr(SqlAlchemyStore, "_original_create_run"):
        SqlAlchemyStore.create_run = SqlAlchemyStore._original_create_run
        delattr(SqlAlchemyStore, "_original_create_run")
        log.debug("MLflow SqlAlchemyStore patch has been removed successfully!")
    else:
        log.debug("No patch found to remove.")


if is_package_installed("mlflow"):
    try:
        apply_patch()
    except Exception as e:  # pragma: no cover - best effort logging
        log.debug(f"Skipping MLflow patch due to error: {e}")
