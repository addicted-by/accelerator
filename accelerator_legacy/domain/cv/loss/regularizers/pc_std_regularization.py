import torch 
from textwrap import dedent


from accelerator.typings.base import CONTEXT
from accelerator.runtime.loss import registry, LossType


@registry.register_loss(LossType.REGULARIZATION, 'pc_std_regularization')
def pc_std_regularization( 
        net_result,
        ground_truth,
        context: CONTEXT, 
        *args,
        **kwargs
    ):
        if context is None:
            raise ValueError(dedent(
                """
                {self._name} requires `context` 
                as well! Please provide it.
                """)
            )
    
        model = context.model
        for n, m in model.named_modules():
            if getattr(m, 'enable_std_regularization', False):
                std_min = torch.std(m.input_quantizer.min)
                std_max = torch.std(m.input_quantizer.max)

                std_regularization_result = std_min + std_max
                model.add_regularization(std_regularization_result)
        
        return torch.tensor(0.) # add_regularization is done