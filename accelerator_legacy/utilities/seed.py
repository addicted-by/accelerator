def setup_seed(seed: int):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = True

    torch.use_deterministic_algorithms(True)

    # Disable silent mixed-precision paths
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

    # Deterministic cuBLAS (CUDA ≥ 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"