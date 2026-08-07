"""Tile a large micrograph into patches and stitch predictions back together.

Micrographs are far too large to segment in one forward pass, so inference runs
on overlapping patches. Overlap is not optional: a model sees no context beyond
a patch edge, so predictions there are unreliable and butt-joined patches leave
visible seams that the skeletonizer turns into spurious fibril breaks.

Patches are therefore blended with a raised-cosine window that falls to zero at
the border, and the per-pixel weights are accumulated so the result is a proper
weighted average regardless of how many patches cover a pixel.

Everything here is plain NumPy and independent of the model.
"""

from __future__ import annotations

import numpy as np


def tile_positions(shape: tuple[int, int], patch: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left ``(y, x)`` corners of patches covering ``shape``.

    The last row/column is pulled flush with the image edge rather than padded,
    so every pixel is covered without inventing content outside the image.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the image.
    patch:
        Side of the (square) patch in pixels.
    overlap:
        Overlap between neighbouring patches in pixels; must be < ``patch``.
    """
    if overlap >= patch:
        raise ValueError(f"overlap ({overlap}) must be smaller than patch ({patch})")
    height, width = shape
    stride = patch - overlap

    def starts(extent: int) -> list[int]:
        if extent <= patch:
            return [0]
        pos = list(range(0, extent - patch + 1, stride))
        if pos[-1] != extent - patch:
            pos.append(extent - patch)
        return pos

    return [(y, x) for y in starts(height) for x in starts(width)]


def blend_window(patch: int, *, taper: float = 0.25) -> np.ndarray:
    """Raised-cosine weight window, 1 in the middle and 0 at the patch border.

    ``taper`` is the fraction of each side spent ramping up/down. A small
    positive floor keeps the weights from being exactly zero, so pixels covered
    by a single patch still get a defined value.
    """
    ramp_len = max(int(round(patch * taper)), 1)
    w = np.ones(patch, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len + 2)[1:-1]))
    w[:ramp_len] = ramp
    w[-ramp_len:] = ramp[::-1]
    window = np.outer(w, w)
    return np.maximum(window, 1e-3).astype(np.float32)


def stitch(
    patches: list[np.ndarray],
    positions: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Blend patch predictions into a full-size map by weighted averaging.

    Parameters
    ----------
    patches:
        Per-patch 2D arrays, same order as ``positions``.
    positions:
        Top-left ``(y, x)`` corners, as returned by :func:`tile_positions`.
    shape:
        ``(height, width)`` of the output.
    window:
        Per-patch weights; defaults to :func:`blend_window` for the patch size.
    """
    if len(patches) != len(positions):
        raise ValueError("patches and positions must have the same length")
    if not patches:
        return np.zeros(shape, dtype=np.float32)

    size = patches[0].shape[0]
    win = blend_window(size) if window is None else window

    total = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    for arr, (y, x) in zip(patches, positions, strict=True):
        total[y:y + size, x:x + size] += arr.astype(np.float32) * win
        weight[y:y + size, x:x + size] += win
    return total / np.maximum(weight, 1e-8)


def extract(image: np.ndarray, positions: list[tuple[int, int]], patch: int) -> np.ndarray:
    """Stack the patches of ``image`` at ``positions`` into ``(n, patch, patch)``."""
    return np.stack([image[y:y + patch, x:x + patch] for y, x in positions])
