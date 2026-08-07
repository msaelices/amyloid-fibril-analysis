"""Morphology metrics computed from an ordered fibril centerline.

All metrics are pure functions of the polyline geometry. The polyline should be
given in **physical units** (e.g. nanometres); pass ``pixel_size`` to convert a
pixel-space polyline on the fly. Curvature is sensitive to discretization, so
callers should resample + smooth the centerline first (see
:mod:`afa.trace.resample`).

Definitions
-----------
Given ordered points ``p_i`` with segment vectors ``d_i = p_{i+1} - p_i``,
segment lengths ``Δs_i = |d_i|`` and tangent angles ``θ_i = atan2(d_i.y, d_i.x)``:

* length              ``L = Σ Δs_i``
* tortuosity          ``L / |p_last - p_first|``
* turning angle       ``Δθ_i = wrap(θ_{i+1} - θ_i)`` in ``(-π, π]``
* curvature at vertex ``κ_i = Δθ_i / s_i`` with ``s_i = (Δs_{i-1}+Δs_i)/2``
* max curvature       ``max |κ_i|``
* mean abs curvature  ``mean |κ_i|`` (a.k.a. local direction change per length)
* total abs turning   ``Σ |Δθ_i|`` (radians)
* turning per length  ``Σ |Δθ_i| / L``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FibrilMetrics:
    """Container for the morphology descriptors of one fibril."""

    length: float
    tortuosity: float
    max_curvature: float
    mean_abs_curvature: float
    total_abs_turning: float
    total_abs_turning_per_length: float
    local_dir_change_per_length: float
    n_points: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval (-pi, pi]."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _clean_polyline(points: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate vertices that would create zero-length segments."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("A polyline needs at least 2 points")
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0.0, axis=1)
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("Polyline collapsed to a single point after de-duplication")
    return pts


def compute_metrics(points: np.ndarray, *, pixel_size: float = 1.0) -> FibrilMetrics:
    """Compute all morphology metrics for a single ordered centerline.

    Parameters
    ----------
    points:
        ``(N, 2)`` array of ordered ``(x, y)`` vertices.
    pixel_size:
        Physical size of one pixel (e.g. nm/pixel). Lengths scale by this and
        curvature by ``1/pixel_size``. Pass ``1.0`` if ``points`` are already in
        physical units.
    """
    pts = _clean_polyline(points) * float(pixel_size)

    seg = np.diff(pts, axis=0)                      # (M, 2), M = N-1
    seg_len = np.linalg.norm(seg, axis=1)           # (M,)
    length = float(seg_len.sum())

    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    tortuosity = length / chord if chord > 0 else float("inf")

    if len(seg) < 2:
        # A straight two-point segment: no turning, no curvature.
        return FibrilMetrics(
            length=length,
            tortuosity=tortuosity,
            max_curvature=0.0,
            mean_abs_curvature=0.0,
            total_abs_turning=0.0,
            total_abs_turning_per_length=0.0,
            local_dir_change_per_length=0.0,
            n_points=int(len(pts)),
        )

    theta = np.arctan2(seg[:, 1], seg[:, 0])        # (M,)
    dtheta = _wrap_to_pi(np.diff(theta))            # (M-1,) turning at interior vertices
    abs_turn = np.abs(dtheta)

    # Local arc length associated with each interior vertex.
    s_vertex = 0.5 * (seg_len[:-1] + seg_len[1:])   # (M-1,)
    curvature = dtheta / np.where(s_vertex > 0, s_vertex, np.nan)
    abs_curv = np.abs(curvature)
    abs_curv = abs_curv[np.isfinite(abs_curv)]

    total_abs_turning = float(abs_turn.sum())
    max_curvature = float(np.nanmax(abs_curv)) if abs_curv.size else 0.0
    mean_abs_curvature = float(np.nanmean(abs_curv)) if abs_curv.size else 0.0

    return FibrilMetrics(
        length=length,
        tortuosity=tortuosity,
        max_curvature=max_curvature,
        mean_abs_curvature=mean_abs_curvature,
        total_abs_turning=total_abs_turning,
        total_abs_turning_per_length=total_abs_turning / length if length > 0 else 0.0,
        local_dir_change_per_length=mean_abs_curvature,
        n_points=int(len(pts)),
    )
