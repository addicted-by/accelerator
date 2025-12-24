# pylint: disable=missing-module-docstring
from typing import Union

import torch
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple

from .base import SmoothAverage


class SmoothConv1d(SmoothAverage, torch.nn.Conv1d):
    """Con1d that can smoothly turn into an affine identity mapping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        *,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        alpha_init: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype,
        }
        super().__init__(**conv_kwargs, alpha_init=alpha_init)

    def core_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv1d(  # pylint: disable=not-callable
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(  # pylint: disable=not-callable
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    # def identity_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401
    #     if self.weight is not None:
    #         out = input * self.weight
    #         if self.bias is not None:
    #             out = out + self.bias
    #         return out
    #     return input

    def set_standard_mode(self) -> None:  # noqa: D401
        super().set_standard_mode()
        if self.weight is not None:
            self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)


class SmoothConv2d(SmoothAverage, torch.nn.Conv2d):
    """Con1d that can smoothly turn into an affine identity mapping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        alpha_init: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype,
        }
        super().__init__(**conv_kwargs, alpha_init=alpha_init)

    def core_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(  # pylint: disable=not-callable
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(  # pylint: disable=not-callable
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    # def identity_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401
    #     if self.weight is not None:
    #         out = input * self.weight
    #         if self.bias is not None:
    #             out = out + self.bias
    #         return out
    #     return input

    def set_standard_mode(self) -> None:  # noqa: D401
        super().set_standard_mode()
        if self.weight is not None:
            self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)


class SmoothConv3d(SmoothAverage, torch.nn.Conv3d):
    """Con1d that can smoothly turn into an affine identity mapping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        *,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        alpha_init: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype,
        }
        super().__init__(**conv_kwargs, alpha_init=alpha_init)

    def core_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv3d(  # pylint: disable=not-callable
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(  # pylint: disable=not-callable
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    # def identity_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401
    #     if self.weight is not None:
    #         out = input * self.weight
    #         if self.bias is not None:
    #             out = out + self.bias
    #         return out
    #     return input

    def set_standard_mode(self) -> None:  # noqa: D401
        super().set_standard_mode()
        if self.weight is not None:
            self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
