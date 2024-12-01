# comparison of two models

# logic to compare two models from weights to state dicts, attributes, buffers, etc.

def compare_models(model1, model2):
    ...

def model_info(model, verbosity) -> str:
    # return string of main model information
    # -- class name (verbosity=0)
    # -- num of layers (verbosity=0)
    # -- modules, that are presented in the model (verbosity=0)
    # -- number of parameters (verbosity=0)
    # -- each module separate information (verbosity=-1)
    # -- Weights statistics per-layer (verbosty=-1)
    # -- FLOPs (verbosity=-1)
    # -- BOPs (verbosity=-1)
    # -- MACs (verbosity=-1)
    # -- size (verbosity=-1)

    ...