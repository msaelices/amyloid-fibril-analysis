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
    """Thin wrapper around a U-Net for fibril probability prediction.

    The implementation is deferred until a sample image + traces are available
    so the architecture/patching can be tuned to the real data. The interface is
    fixed so the rest of the pipeline (tracing, metrics) does not change when the
    model lands.
    """

    def __init__(self, weights: str | Path | None = None, device: str = "cpu") -> None:
        self.weights = Path(weights) if weights else None
        self.device = device
        self._model = None

    def load(self) -> UNetSegmenter:
        raise NotImplementedError(
            "U-Net weights loading is not implemented yet. Train a model with the "
            "recipe in this module's docstring, then wire loading here."
        )

    def predict(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - stub
        """Return a [0, 1] fibril-probability map for a full micrograph."""
        raise NotImplementedError(
            "Provide trained weights and implement tiled inference. Until then use "
            "afa.segment.classical.vesselness_probability as the detector."
        )
