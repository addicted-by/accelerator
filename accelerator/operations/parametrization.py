from torch.nn.utils import parametrize


def remove_parametrizations(model):
    """
    Function to remove parametrizations from a model
    """
    parametrized_modules = {}

    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            parametrized_modules[name] = []

            for p_name in list(module.parametrizations.keys()):
                orig_parameter = getattr(module.parametrizations, p_name)
                orig_parameter = orig_parameter.original.data.detach().clone()
                parametrized_modules[name].append(
                    (p_name, module.parametrizations[p_name], orig_parameter)
                )
                parametrize.remove_parametrizations(module, p_name, True)
    
    return parametrized_modules

def reapply_parametrizations(model, parametrized_modules, unsafe=True):
    """Function to reapply parameterizations to a model."""

    for name, params in parametrized_modules.items():
        module = dict(model.named_modules())[name]

        for p_name, parametrizations, orig_parameter in params:
            for parametrization in parametrizations:
                parametrize.register_parametrization(
                    module, p_name, parametrization, unsafe=unsafe
                )
            getattr(module.parametrizations, p_name).original.data = orig_parameter
