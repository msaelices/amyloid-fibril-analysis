"""Render traced centerlines over the raw micrograph for visual validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_overlay(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    out_path: str | Path,
    *,
    labels: list[str] | None = None,
    linewidth: float = 1.2,
    dpi: int = 200,
) -> Path:
    """Draw ``centerlines`` on top of ``image`` and save a PNG.

    Parameters
    ----------
    image:
        2D micrograph (grayscale).
    centerlines:
        List of ``(N, 2)`` ``(x, y)`` polylines in pixel coordinates.
    out_path:
        Where to write the PNG.
    labels:
        Optional per-fibril labels drawn near each trace start.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    disp = image.astype(np.float32)
    lo, hi = np.percentile(disp, [1, 99])
    disp = np.clip((disp - lo) / (hi - lo + 1e-9), 0, 1)

    fig, ax = plt.subplots(figsize=(disp.shape[1] / 200, disp.shape[0] / 200))
    ax.imshow(disp, cmap="gray", interpolation="nearest")

    # `cm.get_cmap` was removed in matplotlib 3.9; `colormaps[...].resampled`
    # is the supported equivalent (available since 3.6).
    colors = matplotlib.colormaps["hsv"].resampled(max(len(centerlines), 1))
    for i, cl in enumerate(centerlines):
        cl = np.asarray(cl)
        ax.plot(cl[:, 0], cl[:, 1], "-", lw=linewidth, color=colors(i))
        if labels is not None and i < len(labels):
            ax.text(cl[0, 0], cl[0, 1], str(labels[i]), color="yellow", fontsize=6)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path
