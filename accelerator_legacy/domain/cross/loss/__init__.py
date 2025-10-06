"""Cross-domain loss functions including PyTorch standard losses."""

import torch.nn as nn
from accelerator.runtime.loss import LossType, registry

# Register PyTorch classification losses
registry.add_loss(LossType.CLASSIFICATION.value, nn.CrossEntropyLoss, 'cross_entropy')
registry.add_loss(LossType.CLASSIFICATION.value, nn.NLLLoss, 'nll_loss')
registry.add_loss(LossType.CLASSIFICATION.value, nn.BCELoss, 'bce_loss')
registry.add_loss(LossType.CLASSIFICATION.value, nn.BCEWithLogitsLoss, 'bce_with_logits')
registry.add_loss(LossType.CLASSIFICATION.value, nn.MultiLabelMarginLoss, 'multi_label_margin')
registry.add_loss(LossType.CLASSIFICATION.value, nn.MultiLabelSoftMarginLoss, 'multi_label_soft_margin')
registry.add_loss(LossType.CLASSIFICATION.value, nn.MultiMarginLoss, 'multi_margin')

# Register PyTorch regression losses
registry.add_loss(LossType.REGRESSION.value, nn.MSELoss, 'l2_loss')
registry.add_loss(LossType.REGRESSION.value, nn.L1Loss, 'l1_loss')
registry.add_loss(LossType.REGRESSION.value, nn.SmoothL1Loss, 'smooth_l1')
registry.add_loss(LossType.REGRESSION.value, nn.HuberLoss, 'huber_loss')

# Register other common PyTorch losses
registry.add_loss(LossType.CUSTOM.value, nn.KLDivLoss, 'kl_div')
registry.add_loss(LossType.CUSTOM.value, nn.PoissonNLLLoss, 'poisson_nll')
registry.add_loss(LossType.CUSTOM.value, nn.GaussianNLLLoss, 'gaussian_nll')
registry.add_loss(LossType.CUSTOM.value, nn.HingeEmbeddingLoss, 'hinge_embedding')
registry.add_loss(LossType.CUSTOM.value, nn.MarginRankingLoss, 'margin_ranking')
registry.add_loss(LossType.CUSTOM.value, nn.TripletMarginLoss, 'triplet_margin')
registry.add_loss(LossType.CUSTOM.value, nn.TripletMarginWithDistanceLoss, 'triplet_margin_distance')
registry.add_loss(LossType.CUSTOM.value, nn.CosineEmbeddingLoss, 'cosine_embedding')
registry.add_loss(LossType.CUSTOM.value, nn.CTCLoss, 'ctc_loss')

# Import existing cross-domain losses
from .mae_loss import mae_loss
from .mse_loss import mse_loss, multi_mse_loss_fn