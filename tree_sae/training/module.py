"""PyTorch Lightning module that trains any SAE of this package."""

from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from ..models.base import BaseSAE, geometric_median, remove_parallel_decoder_grad, unit_norm_decoder
from ..models.tree import TreeSAE
from .losses import auxiliary_loss, reconstruction_loss


def _first_element(batch):
    return batch[0]


class SAETrainingModule(LightningModule):
    """Optimises ``L = sum_l ||x_hat_l - x||^2 + sum_l alpha_l L_aux^l + extra`` (Eqs. 3, 7-9).

    ``x_hat_l`` are the cumulative reconstructions of every privilege layer for the Tree SAE and
    the single full reconstruction for the other SAEs; ``extra`` is the L1 penalty of the ReLU SAE
    and the strict-prefix losses of the Matryoshka SAE.
    """

    def __init__(
        self,
        sae: BaseSAE,
        lr: float = 1e-4,
        auxk_coef: float = 1 / 32,
        use_loss_var: bool = True,
        preprocess: Callable = _first_element,
    ) -> None:
        """
        Args:
            sae: The SAE to train.
            lr: Adam learning rate.
            auxk_coef: Coefficient ``alpha`` of the auxiliary loss.
            use_loss_var: Divide reconstruction losses by ``Var(x)``.
            preprocess: Maps a dataloader batch to model activations. It runs inside the training
                step (i.e. under the trainer's autocast), e.g. a forward pass of the language model.
        """
        super().__init__()
        self.sae = sae
        self.lr = lr
        self.auxk_coef = auxk_coef
        self.use_loss_var = use_loss_var
        self.preprocess = preprocess

    def _shared_step(self, x: Tensor, prefix: str) -> Tensor:
        out = self.sae.forward_training(x)

        logs = {}
        total_mse, total_aux = 0, 0
        for layer, (recons, auxk_recons) in enumerate(zip(out.recons, out.auxk_recons)):
            mse = reconstruction_loss(x, recons, self.use_loss_var)
            aux = auxiliary_loss(x, recons, auxk_recons, self.auxk_coef)
            total_mse += mse
            total_aux += aux
            logs[f"{prefix}loss/mse/{layer}"] = mse
            logs[f"{prefix}loss/aux/{layer}"] = aux
        loss = total_mse + total_aux + out.extra_loss

        if prefix == "" and isinstance(self.sae, TreeSAE):
            self.sae.update_allocation(loss)
            logs.update(self._allocation_logs())
        for layer, dead in enumerate(out.dead_fraction):
            logs[f"{prefix}dead/{layer}"] = dead
        logs[f"{prefix}loss/extra"] = out.extra_loss
        logs[f"{prefix}loss/total"] = loss
        self.log_dict(logs)
        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x = self.preprocess(batch)
        if batch_idx == 0:
            self.sae.b_dec.data = geometric_median(x)
        return self._shared_step(x, prefix="")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(self.preprocess(batch), prefix="val/")

    def on_after_backward(self) -> None:
        unit_norm_decoder(self.sae.decoder)
        remove_parallel_decoder_grad(self.sae.decoder)

    def configure_optimizers(self):
        return torch.optim.Adam(self.sae.parameters(), lr=self.lr, eps=6.25e-10)

    @torch.no_grad()
    def _allocation_logs(self) -> dict[str, float]:
        logs = {}
        for layer in range(1, self.sae.n_layers):
            counts = torch.bincount(self.sae.parent_index(layer), minlength=self.sae.layer_start(layer) + 1)
            logs[f"alloc/{layer}/root_children"] = counts[-1].float()
            logs[f"alloc/{layer}/parents_with_children"] = (counts[:-1] > 0).float().mean()
        return logs
