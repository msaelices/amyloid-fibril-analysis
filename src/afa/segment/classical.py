"""Zero-training classical detector: denoise + multiscale vesselness (Frangi).

This is the day-one baseline so the whole pipeline runs before any model is
trained. On low-SNR cryo-EM it will pick up some noise and dense aggregates;
that is expected -- the learned detector in :mod:`afa.segment.unet` is the path
to higher precision/recall.
"""

from __future__ import annotations

import numpy as np


def vesselness_probability(
    image: np.ndarray,
    *,
    invert: bool = True,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    denoise: bool = True,
) -> np.ndarray:
    """Return a [0, 1] fibril-likelihood map from a raw micrograph.

    Parameters
    ----------
    image:
        2D micrograph.
    invert:
        Fibrils in cryo-EM are usually darker than background; Frangi detects
        bright ridges, so we invert by default.
    sigmas:
        Scales (in pixels) of the ridge filter -- roughly the half-widths of the
        fibrils to detect.
    denoise:
        Apply an edge-preserving denoise before ridge detection.
    """
    from skimage.filters import frangi
    from skimage.restoration import denoise_bilateral

    img = image.astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    if invert:
        img = 1.0 - img
    if denoise:
        img = denoise_bilateral(img, sigma_color=0.1, sigma_spatial=2)

    prob = frangi(img, sigmas=sigmas, black_ridges=False)
    prob = prob / (prob.max() + 1e-9)
    return prob.astype(np.float32)
