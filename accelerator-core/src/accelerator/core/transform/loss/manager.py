from typing import Any

from accelerator.utilities.api_desc import APIDesc

from ..base import BaseTransform
from ..registry import transforms_registry


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="beta")
class LossTransformManager:
    """Manages transformation pipelines with efficient caching and instance reuse.

    This manager ensures transforms with identical configurations are reused,
    provides appropriate caching for both joint and single transform modes,
    and manages the application of sequences of transforms.

    Attributes:
        transforms (FrozenSet[BaseTransform]): Unique set of transform instances.
        transform_sequence (List[BaseTransform]): Ordered sequence of transforms to apply.
        pred_cache (Dict): Cache for transformed predictions.
        gt_cache (Dict): Cache for transformed ground truths.
        joint_results_cache (Dict): Cache for jointly transformed results.

    """

    def __init__(self):
        """Initialize the transform manager with a list of transforms.

        Args:
            transforms (List[BaseTransform]): List of transform instances to manage.

        """
        self.pred_cache = {}
        self.gt_cache = {}
        self.joint_results_cache = {}

    def apply_transforms(
        self,
        source_key: str,
        target_key: str,
        net_result: dict[str, Any],
        labels: dict[str, Any],
        transform_pipeline: list[BaseTransform],
        **kwargs,
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Apply a sequence of transformations, caching results efficiently.

        Args:
            source_key: Key for the prediction in net_result.
            target_key: Key for the ground truth in labels.
            net_result: Dictionary of model predictions.
            labels: Dictionary of ground truth data.
            transform_pipeline: PLACEHOLDER.
            **kwargs: Additional keyword arguments to pass to transforms.

        Returns:
            Tuple of (transformed_prediction, transformed_ground_truth, list_of_additional_infos)

        """
        current_pred = net_result if source_key is None else net_result[source_key]
        current_gt = labels if target_key is None else labels[target_key]
        additional_infos = []

        for i, transform in enumerate(transform_pipeline):
            transform_path = tuple(t for t in transform_pipeline[:i]) + (transform,)
            if transform._mode == "joint":
                prefix_key = (source_key, target_key, transform_path)
                if prefix_key not in self.joint_results_cache:
                    t_pred, t_gt, info = transform(current_pred, current_gt, **kwargs)
                    self.joint_results_cache[prefix_key] = (t_pred, t_gt, info)
                else:
                    t_pred, t_gt, info = self.joint_results_cache[prefix_key]

                additional_infos.append((transform.name, info))
            else:
                pred_key = (source_key, transform_path)
                gt_key = (target_key, transform_path)

                if pred_key not in self.pred_cache:
                    t_pred, pred_info = transform._apply_single(current_pred, is_prediction_tensor=True, **kwargs)
                    self.pred_cache[pred_key] = (t_pred, pred_info)
                else:
                    print(f"For {i}: {transform.name} reusing {pred_key}")
                    t_pred, pred_info = self.pred_cache[pred_key]

                if gt_key not in self.gt_cache:
                    t_gt, gt_info = transform._apply_single(current_gt, is_prediction_tensor=False, **kwargs)
                    self.gt_cache[gt_key] = (t_gt, gt_info)
                else:
                    print(f"For {i}: {transform.name} reusing {gt_key}")
                    t_gt, gt_info = self.gt_cache[gt_key]

                additional_infos.append((transform.name, (pred_info, gt_info)))

            current_pred = t_pred
            current_gt = t_gt
        return current_pred, current_gt, additional_infos

    @staticmethod
    def _instantiate_transforms(transforms_config: list) -> list[BaseTransform]:
        """Convert config list to transform instances.

        Args:
            transforms_config: List of transform configurations

        Returns:
            List of instantiated transform objects

        Raises:
            ValueError: If an invalid transform entry is provided

        """
        return transforms_registry.instantiate_transform_pipeline(transforms_config)

    def clear_cache(self):
        """Clear all transformation caches."""
        self.pred_cache.clear()
        self.gt_cache.clear()
        self.joint_results_cache.clear()
