"""Learned fibril segmenter (U-Net) -- interface and training scaffold.

This module intentionally keeps a thin, documented interface. It is the
recommended detector once the ~20-30 manually traced images are available: the
traces are rasterized into masks and a small U-Net learns to separate fibrils
from background noise far better than any fixed threshold.

Heavy ML deps (torch, segmentation-models-pytorch, albumentations) live behind
the ``dl`` optional-dependency group so the deterministic core installs light.

Recommended recipe
------------------
* Rasterize each manual trace to a binary mask with a fibril-width kernel
  (``rasterize_traces``).
* Tile each micrograph into overlapping patches (e.g. 512x512).
* Augment with flips + 90-degree rotations (fibrils are orientation-invariant).
* Hold out ~5-8 whole images for validation -- never split patches from the same
  image across train/val.
* Train a U-Net (e.g. ``segmentation_models_pytorch.Unet`` with a resnet34
  encoder) using Dice + BCE loss.
* At inference, tile, predict, and stitch to a full-image probability map, then
  hand it to :func:`afa.trace.trace_centerlines`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from afa.io.annotations import Trace


def rasterize_traces(
    traces: list[Trace],
    shape: tuple[int, int],
    *,
    width_px: int = 3,
) -> np.ndarray:
    """Rasterize vector traces into a binary training mask.

    Parameters
    ----------
    traces:
        Manual traces (pixel coordinates).
    shape:
        ``(height, width)`` of the target mask.
    width_px:
        Line thickness (dilation radius) in pixels -- set to the typical fibril
        width so the mask covers the fibril, not just its centerline.
    """
    from skimage.draw import line
    from skimage.morphology import binary_dilation, disk

    mask = np.zeros(shape, dtype=bool)
    for tr in traces:
        pts = np.round(tr.points).astype(int)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=False):
            rr, cc = line(y0, x0, y1, x1)
            valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            mask[rr[valid], cc[valid]] = True
    if width_px > 1:
        mask = binary_dilation(mask, disk(width_px // 2))
    return mask


class UNetSegmenter:
    """A trained U-Net exposed as a full-micrograph fibril-probability predictor.

    Interface parity with :func:`afa.segment.classical.vesselness_probability` is
    the point: both return a ``[0, 1]`` map of the same shape as the input, so
    the tracing and metric stages are identical whichever detector produced it.

    Inference is tiled and blended (see :mod:`afa.segment.tiling`), because a
    micrograph does not fit in one forward pass and butt-joined patches leave
    seams that the skeletonizer turns into false fibril breaks.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        *,
        patch: int = 256,
        overlap: int = 64,
        batch_size: int = 8,
    ) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device
        self.patch = patch
        self.overlap = overlap
        self.batch_size = batch_size
        self._model = None

    def load(self) -> UNetSegmenter:
        """Load the checkpoint. Requires the ``dl`` optional dependencies."""
        if self.weights is None:
            raise ValueError("UNetSegmenter needs a path to trained weights")
        if not self.weights.exists():
            raise FileNotFoundError(f"No U-Net weights at {self.weights}")

        import torch

        from afa.segment.torch_unet import load_model

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = load_model(self.weights, device=self.device)
        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return a ``[0, 1]`` fibril-probability map for a full micrograph.

        The image is normalized the same way as during training, so passing a
        raw micrograph is fine.
        """
        import torch

        from afa.segment.tiling import stitch, tile_positions

        if self._model is None:
            self.load()

        img = np.asarray(image, dtype=np.float32)
        lo, hi = np.percentile(img, [1.0, 99.0])
        # np.percentile returns float64; without the cast the whole map is
        # promoted to double and the forward pass fails on a float32 model.
        img = (
            np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            if hi > lo
            else np.zeros_like(img)
        )

        pad_y = max(self.patch - img.shape[0], 0)
        pad_x = max(self.patch - img.shape[1], 0)
        if pad_y or pad_x:
            img = np.pad(img, ((0, pad_y), (0, pad_x)), mode="reflect")

        positions = tile_positions(img.shape, self.patch, self.overlap)
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(positions), self.batch_size):
                chunk = positions[start:start + self.batch_size]
                batch = np.stack(
                    [img[y:y + self.patch, x:x + self.patch] for y, x in chunk]
                )[:, None]
                tensor = torch.from_numpy(batch).to(self.device)
                out = torch.sigmoid(self._model(tensor)).cpu().numpy()[:, 0]
                probs.extend(out)

        full = stitch(probs, positions, img.shape)
        return full[: image.shape[0], : image.shape[1]].astype(np.float32)
