"""Logistic-regression probes (adapted from Chanin et al., 2024, https://github.com/lasr-spelling/sae-spelling)."""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import tqdm

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_outputs: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @property
    def weights(self) -> torch.Tensor:
        return self.fc.weight

    @property
    def biases(self) -> torch.Tensor:
        return self.fc.bias


def _pos_weights(y: torch.Tensor) -> torch.Tensor:
    n_pos = y.sum(dim=0)
    return (len(y) - n_pos) / n_pos


def train_multi_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_probes: int | None = None,
    batch_size: int = 4096,
    num_epochs: int = 100,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    show_progress: bool = True,
    optimizer: Literal["Adam", "SGD", "AdamW"] = "Adam",
    extra_loss_fn: Callable[[LinearProbe, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    verbose: bool = False,
    device: torch.device | str = DEFAULT_DEVICE,
    map_acts: Callable[[torch.Tensor], torch.Tensor] | None = None,
    probe_dim: int | None = None,
) -> LinearProbe:
    """Train ``num_probes`` one-vs-rest logistic-regression probes at once.

    Args:
        x_train: ``(n_samples, input_dim)`` inputs.
        y_train: ``(n_samples, num_probes)`` multi-hot labels in ``{0, 1}``; positives are re-weighted
            by the negative/positive ratio of every probe.
        map_acts: Optional map applied to every input batch (e.g. SAE encoding); ``probe_dim`` is
            then the dimension of the mapped inputs.
    """
    dtype = x_train.dtype
    num_probes = num_probes or y_train.shape[-1]
    loader = DataLoader(TensorDataset(x_train, y_train.to(dtype=dtype)), batch_size=batch_size, shuffle=True)
    probe = LinearProbe(probe_dim or x_train.shape[-1], num_outputs=num_probes).to(device, dtype=dtype)
    _run_probe_training(
        probe,
        loader,
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=_pos_weights(y_train).to(device)),
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        optimizer_name=optimizer,
        extra_loss_fn=extra_loss_fn,
        verbose=verbose,
        device=device,
        map_acts=map_acts,
    )
    return probe


def train_binary_probe(x_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 256, **kwargs) -> LinearProbe:
    """Single logistic-regression probe; ``y_train`` has shape ``(n_samples,)``."""
    return train_multi_probe(x_train, y_train.unsqueeze(1), num_probes=1, batch_size=batch_size, **kwargs)


def _run_probe_training(
    probe: LinearProbe,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int,
    lr: float,
    end_lr: float,
    weight_decay: float,
    show_progress: bool,
    optimizer_name: Literal["Adam", "SGD", "AdamW"],
    extra_loss_fn: Callable[[LinearProbe, torch.Tensor, torch.Tensor], torch.Tensor] | None,
    verbose: bool,
    device: torch.device | str,
    map_acts: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    probe.train()
    optimizer_cls = {"Adam": optim.Adam, "SGD": optim.SGD, "AdamW": optim.AdamW}[optimizer_name]
    optimizer = optimizer_cls(probe.parameters(), lr=lr, weight_decay=weight_decay)
    # Exponential decay from lr to end_lr over num_epochs.
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp(log(end_lr / lr) / num_epochs))

    epochs = tqdm(range(num_epochs), disable=not show_progress, desc="Epochs")
    for epoch in epochs:
        epoch_loss = 0.0
        for x, y in tqdm(loader, disable=not show_progress, leave=False, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            if map_acts is not None:
                x = map_acts(x)
            optimizer.zero_grad()
            loss = loss_fn(probe(x), y)
            if extra_loss_fn is not None:
                loss += extra_loss_fn(probe, x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / len(loader)
        epochs.set_postfix({"Mean Loss": f"{mean_loss:.8f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
        if verbose:
            print(f"Epoch {epoch + 1}: Mean Loss: {mean_loss:.8f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()
    probe.eval()
