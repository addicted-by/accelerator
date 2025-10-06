from .importances import (
    FPGMImportance, 
    TaylorImportance, 
    LAMPImportance, 
    # OBDCImportance, 
    MagnitudeImportance, 
    HessianImportance
)


imp_fns = {
    'taylor': TaylorImportance,
    'fpgm': FPGMImportance,
    'magnitude': MagnitudeImportance,
    'lamp': LAMPImportance,
    'hessian': HessianImportance,
    # 'obdc': OBDCImportance
}