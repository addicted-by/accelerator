import torch
import torch.nn as nn
from typing import Optional


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

class TrainableScales(nn.Module):
    def __init__(self, shape):
        super(TrainableScales, self).__init__()
        self.shape = shape
        self.scales = nn.Parameter(torch.ones(shape))
    
    def forward(self, x):
        return x * self.scales

class FeatureMergingMLP(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            important_indices=None,
            linear: Optional[nn.Module]=None,
            use_scales=True,
            initialization_cfg=None
        ):
        super(FeatureMergingMLP, self).__init__()
        self.output_channels = output_channels
        self.important_indices = important_indices
        self.initialization_cfg = initialization_cfg
        if isinstance(self.important_indices, (torch.Tensor, list)):
            self.important_indices = self.important_indices[:self.output_channels]
        # TODO: Compare results after TP pruning and such initialization [done]

        if linear is None:
            self.fc = nn.Linear(input_channels, self.output_channels, bias=False)
            self._initialize_weights()
        else:
            print("\t\t\t|||| REUSING ||||")
            self.fc = linear

        if use_scales:
            self.fc = nn.Sequential(
                self.fc,
                TrainableScales(self.output_channels)
            )

    def _initialize_weights(self):
        if isinstance(self.important_indices, (torch.Tensor, list)):
            weight_init = torch.zeros_like(self.fc.weight)
            for idx, preserve_idx in enumerate(self.important_indices):
                weight_init[idx, preserve_idx] = torch.as_tensor(1.0)
        else:
            print(f"INITIALIZING with {self.initialization_cfg['target']}")
            initialization = getattr(torch.nn.init, self.initialization_cfg['target'], None)
            weight_init = eye_like(self.fc.weight)
            if initialization:
                print(f"INITIALIZING with {initialization.__name__}")
                initialization(weight_init, **self.initialization_cfg['kwargs'])
        
        self.fc.weight.data.copy_(weight_init)

    def forward(self, x):
        return self.fc(x)

class FeatureMerging(nn.Module):
    def __init__(
        self,
        size,
        new_size,
        dim,
        pretrain_stage=False,
        important_indices=None,
        name=None,
        use_scales=False,
        linear=None,
        initialization_cfg=None
    ):
        super(FeatureMerging, self).__init__()
        self.size = size
        self.new_size = new_size
        self.dim = dim
        self.name = name
        self.use_scales = use_scales
        self.pretrain_stage = pretrain_stage
        self.mlp = FeatureMergingMLP(
            size, 
            new_size, 
            important_indices, 
            use_scales=self.use_scales if not self.pretrain_stage else False, 
            linear=linear,
            initialization_cfg=initialization_cfg
        )

    def gumbel_trick(self, W):
        sampled = torch.nn.functional.gumbel_softmax(
            self.mlp.fc.weight, 
            tau=1, 
            hard=True
        )
        return W @ sampled.T


    def right_inverse(self, x):
        return x

    def forward(self, W):
        reshaped_W = (
            W.transpose(self.dim, 0)  # transpose to make pruning dim the first dimension
            .reshape(self.size, -1)  # flatten all dimensions except the pruning idm
            .permute(1, 0)           # switch the first and second dimensions to fit MLP
        )

        final_shape = list(W.transpose(self.dim, 0).shape)
        final_shape[0] = self.new_size
        
        if self.pretrain_stage:
            processed_weights = self.gumbel_trick(reshaped_W)
        else:
            processed_weights = self.mlp(reshaped_W)

        return processed_weights.permute(1, 0).reshape(final_shape).transpose(self.dim, 0)

class BiasParametrization(nn.Module):
    def __init__(self, new_size, important_indices=None):
        super(BiasParametrization, self).__init__()
        self.new_size = new_size
        self.important_indices = important_indices
        if self.important_indices is None:
            self.important_indices = torch.arange(new_size)
        self.important_indices = self.important_indices[:new_size]

    def forward(self, b):
        return  b[self.important_indices]




if __name__ == "__main__":
    size = 64
    new_size = 64
    dim = 0

    old_tensor = torch.randint(10, (size, 512, 3, 3))
    dummy_important_indices = [0, 2, 1]
    feature_merging = FeatureMerging(size, new_size, dim, dummy_important_indices)
    new_tensor = feature_merging(old_tensor.float())


    assert torch.allclose(
        old_tensor[dummy_important_indices].float(), 
        new_tensor[:len(dummy_important_indices)]
    )