import copy
import torch 
import random 
from textwrap import dedent


from accelerator.domain.cv.utils import tv
from accelerator.typings.base import CONTEXT, BatchTensorType, _DEVICE
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.REGULARIZATION, 'ott_regularization')
class OTTRegularization(torch.nn.Module):
    def __init__(
        self,
        *,
        device: _DEVICE,
        mono_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.mono_ratio = mono_ratio

        
    def forward(
        self, 
        net_result, 
        ground_truth,
        *,
        context: CONTEXT,
        inputs: BatchTensorType,
        **kwargs
    ):
        
        if context is None or inputs is None:
            raise ValueError(dedent(
                f"""
                {self._name} requires inputs and context.
                Please, pass it!
                """
            ))
        
        frames, meta, *_ = inputs
        model = context.model
        
        frames_dims = frames.shape
        num_mono_samples = int(round(self.mono_ratio * frames_dims[0]))

        if num_mono_samples <= 0:
            num_mono_samples = 1

        random_sample_idxs = random.sample(range(frames_dims[0]), num_mono_samples)
        random_pixels_x = random.sample(range(frames_dims[2]), num_mono_samples)
        random_pixels_y = random.sample(range(frames_dims[3]), num_mono_samples)

        mono_frames = torch.Tensor().to(self.device)
        for i in range(len(random_sample_idxs)):
            mono_frame = copy.copy(frames[random_sample_idxs[i], :, random_pixels_x[i], random_pixels_y[i]])
            mono_frame = mono_frame[None, :, None, None]
            mono_frame = mono_frame.repeat(1, 1, frames_dims[2], frames_dims[3])
            mono_frames = torch.cat([mono_frames, mono_frame])

        mono_meta = copy.copy(meta[random_sample_idxs])
        
        mono_inputs = (mono_frames, mono_meta)
        
        mono_result = model(*mono_inputs)
        
        if isinstance(mono_result, dict):
            mono_result = mono_result['prediction']

        mono_reg_loss = torch.mean(torch.abs(tv(mono_result)))

        return mono_reg_loss