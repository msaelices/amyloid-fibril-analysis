"""Match an original ``.mrc`` micrograph to the screenshot it was annotated on.

The first annotation batch was traced on 4x-downsampled 8-bit screenshots rather
than on the micrographs. Recovering which ``.mrc`` produced which screenshot puts
real units on every measurement and, more importantly, opens the door to working
at full resolution.

Two things make the matching harder than it looks, and both cost a wrong answer
before the method below settled:

* **Do not decimate.** These images are dominated by shot noise, so taking every
  n-th pixel aliases the noise and destroys the correlation: a true pair scored
  0.08 that way. Block-averaging is what makes the fibrils comparable at all.
* **Mask the carbon.** The dark hole edge is a large-scale feature shared by
  every micrograph in a grid square, so a smoothed correlation measures *it*
  rather than the fibrils and reports 0.6-0.75 for unrelated images.

With both applied, the separation is unambiguous: true pairs score ~0.95 and the
best non-match seen is 0.54.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

MATCH_THRESHOLD = 0.85


def block_average(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging ``factor`` x ``factor`` blocks, trimming the edge."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return np.asarray(image, dtype=np.float32)
    h, w = image.shape
    h_t, w_t = h // factor * factor, w // factor * factor
    trimmed = np.asarray(image[:h_t, :w_t], dtype=np.float32)
    return trimmed.reshape(h_t // factor, factor, w_t // factor, factor).mean(axis=(1, 3))


def _normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1.0, 99.0])
    return np.clip((image - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)


def carbon_mask(image: np.ndarray, *, sigma: float = 25.0, threshold: float = 0.42) -> np.ndarray:
    """``True`` where the image is ice (usable), ``False`` over the carbon support.

    The support film is the dark, large-scale region at the edge of a hole. It is
    common to every micrograph of the same grid square, so leaving it in makes
    unrelated images look similar.
    """
    return gaussian_filter(_normalize(image), sigma) >= threshold


def correlation(a: np.ndarray, b: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    """Normalized cross-correlation of two same-shape images, optionally masked."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    x, y = _normalize(a), _normalize(b)
    if mask is not None:
        x, y = x[mask], y[mask]
    if x.size == 0:
        return 0.0
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float((x * y).mean())


@dataclass
class MatchResult:
    """Best and runner-up candidates for one micrograph."""

    micrograph: Path
    best_id: str | None
    best_score: float
    runner_up_id: str | None
    runner_up_score: float
    scale: float
    pixel_size_a: float

    @property
    def matched(self) -> bool:
        return self.best_score >= MATCH_THRESHOLD

    @property
    def screenshot_pixel_size_a(self) -> float:
        """Pixel size of the downsampled screenshot, in Angstrom."""
        return self.pixel_size_a * self.scale


def match_micrograph(
    micrograph: np.ndarray,
    screenshots: dict[str, np.ndarray],
    *,
    smooth: float = 3.0,
    factor: int = 4,
) -> list[tuple[float, str]]:
    """Score ``micrograph`` against every screenshot, best first.

    Both images are block-averaged to a common grid, the carbon is masked out,
    and the remainder is lightly smoothed before correlating.
    """
    from PIL import Image

    reference = _normalize(block_average(micrograph, factor))
    mask = carbon_mask(reference)
    ref_smooth = gaussian_filter(reference, smooth)

    scores = []
    for name, shot in screenshots.items():
        resized = np.asarray(
            Image.fromarray(np.asarray(shot, dtype=np.float32)).resize(
                (reference.shape[1], reference.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        scores.append(
            (correlation(ref_smooth, gaussian_filter(_normalize(resized), smooth), mask=mask), name)
        )
    scores.sort(reverse=True)
    return scores
