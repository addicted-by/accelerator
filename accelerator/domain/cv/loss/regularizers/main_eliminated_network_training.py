import copy
import torch 
from textwrap import dedent

from accelerator.runtime.loss import LossWrapper, registry, LossType
from accelerator.runtime.transform import LossTransformManager
from accelerator.typings.base import _DEVICE, BatchTensorType, CONTEXT
from accelerator.utilities.logging import get_logger


log = get_logger(__name__)


@registry.register_loss(LossType.REGULARIZATION, 'main_eliminated_network_training')
class MainEliminatedNetworkTraining(LossWrapper):
    def __init__(
        self,
        *,
        device: _DEVICE,
        prediction_key = None, 
        target_key = None, 
        loss_coefficient = 1, 
        num_layer: int=5,
        **kwargs
    ):
        super().__init__(prediction_key, target_key, loss_coefficient, **kwargs)

        if self._prediction_key not in {None, 'prediction'}:
            raise ValueError(dedent(
                """
                MainEliminatedNerworkTraining uses its own `transforms_pipeline`,
                please, leave `prediction_key` None or `prediction` depending on
                your model `separate_fn`.
                """
            ))

        self.device = device
        self.num_layer = num_layer
        self.lf_lvl = 1
        
        
    def calculate_batch_loss(
        self, 
        net_result, 
        ground_truth,
        *,
        context: CONTEXT,
        inputs: BatchTensorType, 
        transform_manager: LossTransformManager,
        **kwargs
    ):
        if self._transform_pipeline:
            log.warning(f"Be aware that `transforms` will be rewritten inside {self.__class__.__name__}")
        
        # self._transform_pipeline (rgb2yuv -> buildpyramid)
        if context is None or inputs is None:
            raise ValueError(dedent(
                f"""
                {self._name} requires inputs and context.
                Please, pass it!
                """
            ))
        frames, meta, *_ = inputs
        model = context.model
            
        wo_main = copy.deepcopy(frames)
        wo_main[:, 25:50] = 0

        wo_main_net_result = model((wo_main, meta))

        
        net_yuv_pyr, label_yuv_pyr, *_ = transform_manager.apply_transforms(
            'wo_main_prediction',
            'prediction',
            {'wo_main_prediction': wo_main_net_result},
            {'prediction': net_result},
            self._transform_pipeline
        )

        wide_main_fusion_loss = 0.

        for layer in range(self.lf_lvl):
            net_uv = net_yuv_pyr[layer][:, 1:, ...]
            label_uv = label_yuv_pyr[layer][:, 1:, ...]

            wide_main_fusion_loss += torch.mean(torch.abs(net_uv - label_uv))
        

        self._transform_pipeline.clear()
        return wide_main_fusion_loss