from typing import Any, Optional, Union

from omegaconf import DictConfig

from accelerator.core.acceleration import AccelerationOperationBase
from accelerator.core.model import AcceleratedModel
from accelerator.utilities.logging import get_logger

logger = get_logger(__name__)


METADATA_VERSION = "1.0"


def _filter_latest_ops(ops: list[AccelerationOperationBase]) -> list[AccelerationOperationBase]:
    """Collapse a list of AccelerationOperation so that for each distinct only the last one remains."""
    seen: list[str] = list()
    filtered: list[tuple[str, AccelerationOperationBase]] = []
    for op in reversed(ops):
        key = op.__class__.registry_type
        # print(key, seen)
        if key not in seen:
            seen.append(key)
            filtered.append((key, op.state_dict()))
    return dict(reversed(filtered))


class MetadataHandler:
    """Handles checkpoint metadata operations and validation."""

    def __init__(self, config: DictConfig):
        self.config = DictConfig(config)

    def get_metadata(self, model: Union[AcceleratedModel, Any]) -> dict:
        """Extract metadata from accelerated model.

        Args:
            model: The accelerated model to extract metadata from

        Returns:
            Dictionary containing model metadata

        """
        if not isinstance(model, AcceleratedModel):
            return {}

        return {
            "version": METADATA_VERSION,
            "model_config": model.model_config,
            "acceleration__meta_data__": _filter_latest_ops(model.acceleration__meta_data__),
        }

    def validate_configs(
        self, model: AcceleratedModel, saved_model_config: Optional[dict], acceleration_metadata: Optional[dict]
    ) -> None:
        """Validate model and acceleration configurations.

        Args:
            model: The model to validate
            saved_model_config: Saved model configuration
            acceleration_metadata: Saved acceleration metadata

        Raises:
            RuntimeError: If configurations don't match

        """
        ignore_model = self.config.get("ignore_model_keys", [])
        ignore_accel = self.config.get("ignore_acceleration_keys", [])
        if saved_model_config:
            diff = self._deep_diff(saved_model_config, model.model_config, ignore=ignore_model)
            if diff:
                raise RuntimeError(f"Model config mismatch:\n{self._format_diff(diff)}")
            else:
                logger.info("[PASSED] Model configs are identical. Good job!")
        else:
            logger.warning("No model config found in checkpoint")

        if acceleration_metadata:
            logger.info("[PLACEHOLDER] FOR ACCELERATION CONFIG VALIDATION. SHOULD BE DISCUSSED")
            print(f"ignore {ignore_accel}")
            # for accel_name, accel_meta in acceleration_metadata.items():
            #     current_cfg = OmegaConf.select(self.config.acceleration, accel_name)
            #     diff = self._deep_diff(accel_meta["cfg"], current_cfg, ignore=ignore_accel)
            #     if diff:
            #         raise RuntimeError(
            #             f"Acceleration config mismatch for {accel_name}:\n{self._format_diff(diff)}"
            #         )
            #     else:
            #         logger.info('[PASSED] Acceleration configs are identical. Good job!')
        else:
            logger.warning("No acceleration metadata found in checkpoint")

    def _format_diff(self, diff: dict) -> str:
        """Format configuration differences for better readability.

        Args:
            diff: Dictionary of differences from _deep_diff

        Returns:
            Formatted string representation of differences

        """
        if not diff:
            return "No differences"

        lines = ["Configuration differences:"]
        for path, values in diff.items():
            if isinstance(values, dict) and "saved" in values and "current" in values:
                lines.append(f"  {path}:")
                lines.append(f"    - saved:   {values['saved']}")
                lines.append(f"    + current: {values['current']}")
            else:
                lines.append(f"  {path}: {values}")

        return "\n".join(lines)

    def _deep_diff(self, d1: dict, d2: dict, ignore: tuple = (), path: str = "") -> dict:
        """Compare two dictionaries deeply and return differences.

        Args:
            d1: First dictionary
            d2: Second dictionary
            ignore: List of paths to ignore
            path: Current path in the dictionary

        Returns:
            Dictionary containing differences

        """
        diff = {}
        for key in set(d1.keys()) | set(d2.keys()):
            new_path = f"{path}.{key}" if path else key
            if ignore and new_path in ignore:
                continue
            val1 = d1.get(key, object())
            val2 = d2.get(key, object())

            if isinstance(val1, dict) and isinstance(val2, dict):
                sub_diff = self._deep_diff(val1, val2, ignore, new_path)
                if sub_diff:
                    diff[new_path] = sub_diff
            elif val1 != val2:
                diff[new_path] = {"saved": val1, "current": val2}
        return diff
