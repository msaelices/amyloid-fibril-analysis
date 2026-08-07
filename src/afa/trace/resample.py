"""Arc-length resampling and smoothing of centerlines.

Curvature is very sensitive to pixel-stepped, jagged centerlines, so before any
morphology is computed a polyline should be resampled to a uniform arc-length
step and lightly smoothed.
"""

from __future__ import annotations

import numpy as np


def _arc_length(points: np.ndarray) -> np.ndarray:
    seg = np.diff(points, axis=0)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
    return d


def resample_polyline(points: np.ndarray, *, step: float = 1.0) -> np.ndarray:
    """Resample a polyline to approximately uniform arc-length spacing.

    Parameters
    ----------
    points:
        ``(N, 2)`` ordered vertices.
    step:
        Target spacing between output vertices (same units as ``points``).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    s = _arc_length(pts)
    total = s[-1]
    if total <= 0:
        return pts[:1]
    n = max(int(np.ceil(total / step)) + 1, 2)
    s_new = np.linspace(0.0, total, n)
    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([x, y])


def smooth_polyline(points: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Smooth a polyline with a moving average, keeping the endpoints fixed.

    ``window`` is the (odd) number of points in the averaging window. Endpoints
    are preserved so length and tortuosity stay meaningful.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if window < 3 or len(pts) <= window:
        return pts
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    pad = window // 2
    out = pts.copy()
    for c in range(2):
        padded = np.pad(pts[:, c], pad, mode="edge")
        out[:, c] = np.convolve(padded, kernel, mode="valid")
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out
