from .importances import (
    FPGMImportance,
    HessianImportance,
    LAMPImportance,
    # OBDCImportance,
    MagnitudeImportance,
    TaylorImportance,
)

imp_fns = {
    "taylor": TaylorImportance,
    "fpgm": FPGMImportance,
    "magnitude": MagnitudeImportance,
    "lamp": LAMPImportance,
    "hessian": HessianImportance,
    # 'obdc': OBDCImportance
}
