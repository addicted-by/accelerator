import abc
from typing import Any, Optional

from accelerator.utilities.api_desc import APIDesc

from ..base import BaseTransform


@APIDesc.developer(dev_info="Ryabykin Alexey r00926208")
@APIDesc.status(status_level="beta")
class BaseLossTransform(BaseTransform, abc.ABC):
    """Abstract base class for transformations applied to predictions and ground truths.

    Subclasses MUST implement EITHER _apply_joint OR _apply_single.
    The class will dynamically determine the mode of operation based on which of these
    methods is overridden. If both or neither are overridden, an error will be raised.

    Attributes:
        config (Dict[str, Any]): Configuration parameters for the transformation.
        name (str): The name of the transformation, derived from the class name
                    or specified in the config.
        _mode (str): Internal flag set to 'joint' or 'single' based on detected
                     implementation.

    """

    def __init__(self, **kwargs):
        """Initialize the transformation and determine its mode of operation.

        Args:
            config (Optional[Dict[str, Any]], optional): Configuration parameters.
                Supports 'name' (str) to override the default name.
                Defaults to None, resulting in an empty config.
            kwargs: PLACEHOLDER

        Raises:
            TypeError: If the subclass implements both _apply_joint and _apply_single,
                       or if it implements neither.

        """
        super().__init__(config=kwargs)

        self._mode: Optional[str] = None

        is_joint_implemented = self._apply_joint.__func__ is not BaseLossTransform._apply_joint
        is_single_implemented = self._apply_single.__func__ is not BaseLossTransform._apply_single

        if is_joint_implemented and is_single_implemented:
            raise TypeError(
                f"Transform '{self.name}' implements both '_apply_joint' and '_apply_single'. "
                "This is ambiguous. Please implement only one."
            )
        elif is_joint_implemented:
            self._mode = "joint"
        elif is_single_implemented:
            self._mode = "single"
        else:
            raise TypeError(f"Transform '{self.name}' must implement either '_apply_joint' or '_apply_single'.")

    def _apply_joint(self, prediction: Any, ground_truth: Any, **kwargs) -> tuple[Any, Any, dict[str, Any]]:
        """Apply transformation considering both inputs together.
        MUST be implemented by subclasses if this is the intended mode of operation.

        Args:
            prediction: The prediction data.
            ground_truth: The ground truth data.
            **kwargs: Additional keyword arguments passed from the apply call.

        Returns:
            Tuple containing:
            - Transformed prediction
            - Transformed ground truth
            - Dictionary with metadata about the joint transformation.

        """
        raise NotImplementedError("This method should be overridden by subclasses choosing joint application.")

    def _apply_single(self, tensor: Any, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Apply transformation to a single tensor independently.
        MUST be implemented by subclasses if this is the intended mode of operation.

        Args:
            tensor: The tensor data (either prediction or ground truth).
            **kwargs: Additional keyword arguments passed from the apply call.

        Returns:
            Tuple containing:
            - Transformed tensor
            - Dictionary with metadata about the single tensor transformation.

        """
        raise NotImplementedError("This method should be overridden by subclasses choosing separate application.")

    @APIDesc.status(status_level="Internal Use Only")
    def apply(self, prediction: Any, ground_truth: Any, **kwargs) -> tuple[Any, Any, dict[str, Any]]:
        """Applies the transformation based on the dynamically detected mode.

        This method acts as a dispatcher, calling either _apply_joint or
        _apply_single based on the detected implementation. It also handles metadata
        aggregation.

        Args:
            prediction: The prediction data to transform.
            ground_truth: The ground truth data to transform.
            **kwargs: Additional keyword arguments passed to the underlying
                      _apply_joint or _apply_single methods.

        Returns:
            Tuple containing:
            - Transformed prediction
            - Transformed ground truth
            - Dictionary with aggregated metadata.

        Raises:
            RuntimeError: If the mode was not correctly determined (should be caught by __init__).
            Exception: Propagates exceptions raised during the execution of
                       the underlying transformation methods.

        """
        meta = {"transform_name": self.name, "apply_mode": self._mode}

        try:
            if self._mode == "single":
                transformed_prediction, pred_meta = self._apply_single(prediction, is_prediction_tensor=True, **kwargs)
                transformed_ground_truth, gt_meta = self._apply_single(
                    ground_truth, is_prediction_tensor=False, **kwargs
                )
                meta["prediction_meta"] = pred_meta
                meta["ground_truth_meta"] = gt_meta
            elif self._mode == "joint":
                transformed_prediction, transformed_ground_truth, joint_meta = self._apply_joint(
                    prediction, ground_truth, **kwargs
                )
                meta.update(joint_meta)
            else:
                raise RuntimeError(
                    f"Transform '{self.name}' has an undetermined application mode. This should not happen."
                )

        except Exception as e:
            print(f"Error during '{meta['apply_mode']}' application in transform '{self.name}': {e}")
            raise e

        return transformed_prediction, transformed_ground_truth, meta

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config}, detected_mode='{self._mode}')"
