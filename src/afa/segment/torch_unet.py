"""U-Net architecture, masked loss and training loop (requires the ``dl`` extra).

Kept in its own module so that importing :mod:`afa.segment.unet` -- and with it
the rest of the pipeline -- never pulls in torch.

The network is a small, self-contained U-Net rather than a pretrained backbone.
The reasons are specific to this dataset: it is tiny (tens of images), grayscale,
and its texture statistics are nothing like ImageNet's, so a large pretrained
encoder buys little while costing a heavyweight dependency chain and a slow CPU
forward pass. The interface is unchanged if a bigger backbone is swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Compact U-Net for single-channel input and a single output logit map."""

    def __init__(self, *, base: int = 16, depth: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(depth)]

        self.downs = nn.ModuleList()
        ch = in_channels
        for w in widths:
            self.downs.append(_block(ch, w))
            ch = w
        self.bottleneck = _block(ch, ch * 2)
        ch *= 2

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(ch, w, 2, stride=2))
            self.up_convs.append(_block(w * 2, w))
            ch = w
        self.head = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips), strict=True):
            x = up(x)
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)


def masked_dice_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    pos_weight: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE + soft Dice, both evaluated only where ``weight`` is 1.

    ``pos_weight`` compensates the heavy class imbalance: fibrils occupy a low
    single-digit percentage of the pixels, so unweighted BCE is minimized by
    predicting background everywhere.

    Note for anyone comparing runs: changing ``pos_weight`` changes the loss
    function, so the resulting loss values are on different scales and cannot be
    ranked against each other. Compare such runs on a fixed downstream metric.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    bce = (bce * weight).sum() / weight.sum().clamp_min(eps)

    prob = torch.sigmoid(logits) * weight
    tgt = target * weight
    intersection = (prob * tgt).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (prob.sum() + tgt.sum() + eps)
    return bce + dice


@dataclass
class TrainHistory:
    """Per-epoch losses and the best validation loss seen."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1


def train(
    train_dataset,
    val_dataset,
    *,
    out_path: str | Path,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 3e-4,
    base: int = 16,
    depth: int = 4,
    pos_weight: float = 10.0,
    cosine_schedule: bool = False,
    device: str | None = None,
    num_workers: int = 0,
    seed: int = 0,
    log: bool = True,
) -> TrainHistory:
    """Train a U-Net and save the best-validation weights to ``out_path``.

    ``cosine_schedule`` anneals the learning rate to zero over the run. Off by
    default despite being the more principled choice, because measurement did
    not support it: on the held-out split it bought one extra matched fibril and
    0.06 coverage while doubling the median tortuosity error and producing two
    traces that wandered between fibrils (predicted tortuosity 2.5 against a
    true 1.0). Tortuosity is currently the only descriptor valid without a pixel
    size, so that is a bad trade here. See reports/README.md.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=base, depth=depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        if cosine_schedule
        else None
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = TrainHistory()

    for epoch in range(epochs):
        model.train()
        losses = []
        for image, mask, weight in train_loader:
            image, mask, weight = image.to(device), mask.to(device), weight.to(device)
            optimizer.zero_grad()
            loss = masked_dice_bce(model(image), mask, weight, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.train_loss.append(float(np.mean(losses)) if losses else float("nan"))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for image, mask, weight in val_loader:
                image, mask, weight = image.to(device), mask.to(device), weight.to(device)
                val_losses.append(
                    float(masked_dice_bce(model(image), mask, weight, pos_weight=pos_weight))
                )
        val = float(np.mean(val_losses)) if val_losses else float("nan")
        history.val_loss.append(val)
        if scheduler is not None:
            scheduler.step()

        if val < history.best_val:
            history.best_val = val
            history.best_epoch = epoch
            torch.save(
                {"state_dict": model.state_dict(), "base": base, "depth": depth}, out_path
            )
        if log:
            marker = " *" if history.best_epoch == epoch else ""
            print(
                f"epoch {epoch + 1:3d}/{epochs}  "
                f"train {history.train_loss[-1]:.4f}  val {val:.4f}{marker}"
            )

    return history


def load_model(weights: str | Path, device: str = "cpu") -> UNet:
    """Rebuild a :class:`UNet` from a checkpoint saved by :func:`train`."""
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    model = UNet(base=ckpt.get("base", 16), depth=ckpt.get("depth", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
