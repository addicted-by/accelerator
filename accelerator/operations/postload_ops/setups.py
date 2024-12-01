import logging
import torch


logger = logging.getLogger(__name__)

class PostLoadOpsHandler:
    def __init__(self):
        ...

    @staticmethod
    def unfreeze_all_biases(model, *args, **kwargs):
        for name, params in model.named_parameters():
            if "bias" in name and not params.requires_grad:
                logger.info(f"{name:_<100} was unfreezed")
                params.requires_grad = True

    @staticmethod
    def unfreeze_all_parameters(model, *args, **kwargs):
        for name, params in model.named_parameters():
            if not params.requires_grad:
                logger.info(f"{name:_<100} was unfreezed")
                params.requires_grad = True

    @staticmethod
    def freeze_weights(model, *args, **kwargs):
        for name, params in model.named_parameters():
            if "weight" in name and params.requires_grad:
                logger.info(f"{name:_<100} was freezed")
                params.requires_grad = False

    @staticmethod
    def freeze_params(model, params_to_freeze, *args, **kwargs):
        for name, params in model.named_parameters():
            if (
                any(to_freeze in name for to_freeze in params_to_freeze)
                and params.requires_grad
            ):
                logger.info(f"{name:_<100} was freezed")
                params.requires_grad = False

    @staticmethod
    def freeze_all_params(model, *args, **kwargs):
        for name, params in model.named_parameters():
            if params.requires_grad:
                logger.info(f"{name:_<100} was freezed")
                params.requires_grad = False
    
    @staticmethod
    def freeze_all_biases(model, *args, **kwargs):
        for name, params in model.named_parameters():
            print(name, 'bias' in name, params.requires_grad)
            if 'bias' in name and params.requires_grad:
                logger.info(f'{name:_<100} was freezed')
                params.requires_grad = False

        return 0


    @staticmethod
    def unfreeze_params(model, params_to_unfreeze, *args, **kwargs):
        for name, params in model.named_parameters():
            if (
                any(to_unfreeze in name for to_unfreeze in params_to_unfreeze)
                and not params.requires_grad
            ):
                logger.info(f"{name:_<100} was unfreezed")
                params.requires_grad = True

    @staticmethod
    def print_optimizing_params(model, *args, **kwargs):
        total_trainable_params = 0.
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("Optimizing", name, sep=" => ")
                total_trainable_params += param.numel()
        
        print(f"Total trainable params: {total_trainable_params}")
    
    @staticmethod
    def save_onnx(model, input_shape, save_path, *args, **kwargs):
        device = next(iter(model.parameters())).device
        input_example = torch.rand(input_shape)
        torch.onnx.export(model, input_example.to(device), save_path, **kwargs)


    @staticmethod
    def do_nothing(model, *args, **kwargs):
        ...
