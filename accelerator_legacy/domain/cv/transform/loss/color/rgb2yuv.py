from accelerator.domain import cv as cv_utils
from accelerator.runtime.transform import (
    BaseLossTransform, 
    transforms_registry, 
    TensorTransformType
)


@transforms_registry.register_transform(TensorTransformType.LOSS_TRANSFORM)
class RGB2YUV(BaseLossTransform):
    def _apply_single(self, tensor, **kwargs):
        return cv_utils.utils.rgb2yuv(tensor), {}