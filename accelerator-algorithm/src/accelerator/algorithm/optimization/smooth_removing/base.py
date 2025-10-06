import abc
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from accelerator.algorithm.base import AccelerationOperationBase
from accelerator.algorithm.registry import AccelerationType, registry
from accelerator.core.callbacks.base import BaseCallback
from accelerator.utilities.logging import get_logger
from accelerator.utilities.model_utils.unwrap import unwrap_model

if TYPE_CHECKING:
    from accelerator import Context


log = get_logger(__name__)


# pylint: disable=missing-function-docstring
@registry.register_acceleration(AccelerationType.OPTIMIZATION)
class SmoothlyRemoveLayer(AccelerationOperationBase):
    def apply(self, context: "Context"):
        model = unwrap_model(context.model).model_core
        for module_name in self.config["layers"]:
            module = model.get_submodule(module_name)
            if not hasattr(module, "set_weighted_avg_mode"):
                log.warning(
                    "Mentioned module does not have the `set_weighted_avg_mode`"
                )
            else:
                log.info(f"{module_name}: Weighted avg mode set!")
                module.set_weighted_avg_mode(
                    total_epochs_for_decay=self.config["epochs_num"]
                )

        context.callbacks.add_callback(SmoothAverageCallback(self.config))


    def reapply(self, model):
        unwrapped_model = unwrap_model(model).model_core
        if self.config:
            for module_name in self.config["layers"]:
                module = unwrapped_model.get_submodule(module_name)
                if not hasattr(module, "set_weighted_avg_mode"):
                    log.warning(
                        "Mentioned module does not have the `set_weighted_avg_mode`"
                    )
                else:
                    log.info(f"{module_name}: Weighted avg mode set!")
                    module.set_weighted_avg_mode(
                        total_epochs_for_decay=self.config["epochs_num"]
                    )
        else:
            raise ValueError(self._not_loaded_error_msg)

    def calibrate(self, context: "Context") -> None:
        log.info("Calibrating alphas... [Not implemented yet]")


class SmoothAverage(nn.Module, abc.ABC):
    """Mixin that lets a module interpolate smoothly between its core
    behaviour and an *identity* mapping.
    """

    def __init__(self, *args, alpha_init: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError("alpha_init must be in [0, 1]")

        self._alpha = torch.nn.Parameter(torch.tensor(alpha_init), requires_grad=False)

        self._mode: int = 0  # 0 → standard, 1 → annealed, 2 → trainable
        self._decay_total_epochs: Optional[int] = None

    def set_standard_mode(self) -> None:
        """Use *core* behaviour only (α = 1)."""
        self._mode = 0
        self._alpha.requires_grad_(False)
        self._alpha.fill_(1.0)

    def set_weighted_avg_mode(self, *, total_epochs_for_decay: int) -> None:
        """Cosine‑anneal α from 1 → 0 over *total_epochs_for_decay* epochs."""
        if total_epochs_for_decay <= 0:
            raise ValueError("total_epochs_for_decay must be positive")
        self._mode = 1
        self._decay_total_epochs = total_epochs_for_decay
        self._alpha.requires_grad_(False)

    def set_trainable_alpha_mode(self, *, initial_alpha: Optional[None] = None) -> None:
        """Make α a learnable parameter (clamped in ``forward``)."""
        self._mode = 2
        if initial_alpha is not None:
            if not 0.0 <= initial_alpha <= 1.0:
                raise ValueError("initial_alpha must be in [0, 1]")
            self._alpha.data.fill_(initial_alpha)
        self._alpha.requires_grad_(True)
        log.info("Trainable alpha mode set!")

    @torch.no_grad()
    def update_alpha(self, *, epoch: int) -> None:
        """Call once per epoch in *annealed* mode to update α."""
        if epoch > self._decay_total_epochs - 1:
            return
        if self._mode != 1 or self._decay_total_epochs is None:
            raise RuntimeError("update_alpha is only valid in weighted‑avg mode")
        new_val = 0.5 * (
            1.0
            + torch.cos(torch.tensor(torch.pi) * epoch / (self._decay_total_epochs - 1))
        )
        self._alpha.fill_(float(new_val))

    @abc.abstractmethod
    def core_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The module’s regular computation."""

    def identity_forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """What counts as *identity* (default: return the input untouched)."""
        return x

    # pylint: disable=missing-function-docstring
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: C0116 ignore[override]
        core_out = self.core_forward(x)

        if self._mode == 0:
            return core_out
        if self._mode not in {1, 2}:
            raise ValueError(f"Unknown mode {self._mode}")

        alpha = self._alpha if self._mode == 1 else torch.clamp(self._alpha, 0.0, 1.0)
        identity_out = self.identity_forward(x)
        return alpha * core_out + (1.0 - alpha) * identity_out


# pylint: disable=missing-class-docstring
class SmoothAverageCallback(BaseCallback):
    def on_train_epoch_begin(self, context):
        model = unwrap_model(context.model).model_core

        epoch = context.training_manager.current_epoch
        if epoch // self.config["epoch_freq"] > self.config["epochs_num"]:
            return

        if epoch % self.config["epoch_freq"] == 0:
            for module_name in self.config["layers"]:
                module: SmoothAverage = model.get_submodule(module_name)
                if isinstance(module, SmoothAverage) and module.mode == 1:
                    before = module.alpha.item()
                    module.update_alpha(epoch=epoch)
                    after = module.alpha.item()
                    msg = f"[Epoch {epoch:03d}] {module_name}: alpha {before:.4f} → {after:.4f}"
                    log.info(msg)
