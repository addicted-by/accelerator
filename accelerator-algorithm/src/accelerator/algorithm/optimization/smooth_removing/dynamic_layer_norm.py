# pylint: disable=missing-module-docstring
from typing import Union

import torch

from accelerator.utilities.logging import get_logger

from .base import SmoothAverage

log = get_logger(__name__)


class SmoothLayerNorm(SmoothAverage, torch.nn.LayerNorm):
    """LayerNorm that can smoothly turn into an affine identity mapping."""

    def __init__(
        self,
        normalized_shape: Union[tuple[int, ...], int],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        alpha_init: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
            alpha_init=alpha_init,
        )

    def core_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def identity_forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.elementwise_affine and self.weight is not None:
            out = x * self.weight
            if self.bias is not None:
                out = out + self.bias
            return out
        return x

    def set_standard_mode(self) -> None:  # noqa: D401
        super().set_standard_mode()
        if self.elementwise_affine:
            if self.weight is not None:
                self.weight.requires_grad_(True)
            if self.bias is not None:
                self.bias.requires_grad_(True)
