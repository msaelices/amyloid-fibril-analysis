"""Snap a hand-drawn trace sideways onto the fibril it annotates.

Why this exists
---------------
The manual traces in the first annotation batch were drawn deliberately *beside*
each fibril, not on top of it, so that the drawn line would not hide the fibril
underneath. The geometry is therefore an approximately parallel (offset) curve:
it has the right topology and roughly the right shape, but it does not lie on
the fibril centerline.

That matters twice over:

* **Training masks.** Rasterizing the drawn curve labels empty background beside
  the fibril. A model trained on it learns the wrong thing entirely.
* **Metrics.** An offset curve is not metrically equal to the curve it follows.
  Arc length differs, and curvature differs systematically: for an offset ``d``
  the curvature becomes ``k / (1 - d*k)``. Curvature is the quantity most
  affected, and it is one of the requested descriptors.

So the drawn curve is treated as an *initialization*, and this module moves it
onto the real ridge before anything downstream uses it.

Method
------
For each vertex of the (resampled) drawn polyline, candidate positions are
sampled along the local **normal** within ``+/- max_shift`` pixels, scored by a
fibril-likelihood map (any 2D array where higher means "more fibril", e.g.
:func:`afa.segment.classical.vesselness_probability`). The offset sequence is
then chosen by dynamic programming over the whole trace, maximizing total
likelihood minus a penalty on abrupt offset changes.

The global optimization matters: a per-point argmax jumps between neighbouring
fibrils wherever the ridge response is locally ambiguous, while the DP keeps the
offset sequence coherent along the trace and rides through faint stretches and
crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afa.trace.resample import resample_polyline, smooth_polyline


@dataclass
class SnapResult:
    """Outcome of snapping one trace onto the ridge map.

    Attributes
    ----------
    points:
        ``(N, 2)`` snapped centerline in ``(x, y)`` pixel coordinates.
    offsets:
        Per-vertex signed shift applied along the normal, in pixels.
    score_before, score_after:
        Mean ridge response along the original and the snapped curve. The ratio
        is the honest confidence signal: a trace that did not gain response has
        probably not landed on a fibril and should be reviewed by hand.
    """

    points: np.ndarray
    offsets: np.ndarray
    score_before: float
    score_after: float

    @property
    def gain(self) -> float:
        """Multiplicative improvement in mean ridge response."""
        return self.score_after / self.score_before if self.score_before > 0 else float("inf")


def _unit_normals(points: np.ndarray) -> np.ndarray:
    """Unit normals of a polyline, from central-difference tangents."""
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``field`` at ``(x, y)`` coordinates, 0 outside."""
    from scipy.ndimage import map_coordinates

    return map_coordinates(
        field, [xy[..., 1].ravel(), xy[..., 0].ravel()], order=1, mode="constant", cval=0.0
    ).reshape(xy.shape[:-1])


def _viterbi(
    scores: np.ndarray,
    offsets: np.ndarray,
    jump_penalty: float,
    anchor_penalty: float,
) -> np.ndarray:
    """Best offset index per vertex, maximizing score minus offset-change penalty.

    ``scores`` is ``(n_vertices, n_offsets)``. Returns the index sequence.
    """
    n, k = scores.shape
    transition = -jump_penalty * np.abs(offsets[:, None] - offsets[None, :])  # (k, k)
    # Prefer the smallest move that explains the ridge. Tiny by design: it only
    # decides otherwise-flat cases, where the alternative is that the trace
    # slides to the edge of the search band on no evidence at all.
    scores = scores - anchor_penalty * np.abs(offsets)[None, :]

    best = scores[0].copy()
    back = np.empty((n, k), dtype=np.int32)
    back[0] = np.arange(k)
    for i in range(1, n):
        total = best[:, None] + transition  # from-offset (rows) -> to-offset (cols)
        prev = np.argmax(total, axis=0)
        best = total[prev, np.arange(k)] + scores[i]
        back[i] = prev

    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmax(best))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def snap_to_ridge(
    points: np.ndarray,
    ridge: np.ndarray,
    *,
    max_shift: float = 25.0,
    shift_step: float = 1.0,
    resample_step: float = 1.0,
    jump_penalty: float = 0.1,
    anchor_penalty: float = 1e-3,
    smooth_window: int = 9,
) -> SnapResult:
    """Move a drawn polyline onto the nearby fibril ridge.

    Parameters
    ----------
    points:
        ``(N, 2)`` drawn polyline in ``(x, y)`` pixel coordinates.
    ridge:
        2D fibril-likelihood map; higher means more fibril-like. Typically the
        output of :func:`afa.segment.classical.vesselness_probability`.
    max_shift:
        Largest sideways displacement considered, in pixels. Set it a little
        above the largest offset the annotator used; too large invites snapping
        onto a neighbouring fibril. On the first annotation batch the drawn
        offset has a median magnitude of about 10 px, hence a default of 25.
    shift_step:
        Granularity of the offset search, in pixels.
    resample_step:
        Arc-length spacing the trace is resampled to before snapping.
    jump_penalty:
        Cost per pixel of offset change between consecutive vertices. Higher
        keeps the snapped curve more rigidly parallel to the drawn one; lower
        lets it follow the ridge more freely.
    anchor_penalty:
        Cost per pixel of absolute displacement from the drawn curve. Kept tiny
        on purpose: without it, a stretch of image with no ridge signal has no
        preferred offset and the trace slides to the edge of the search band.
    smooth_window:
        Moving-average window applied to the snapped curve (odd, in vertices).

    Returns
    -------
    SnapResult
    """
    base = resample_polyline(np.asarray(points, dtype=float), step=resample_step)
    if len(base) < 3:
        score = float(_sample(ridge, base).mean())
        return SnapResult(base, np.zeros(len(base)), score, score)

    normals = _unit_normals(base)
    offsets = np.arange(-max_shift, max_shift + shift_step / 2, shift_step)

    # candidates[i, j] = vertex i displaced by offsets[j] along its normal
    candidates = base[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    scores = _sample(ridge, candidates)  # (n_vertices, n_offsets)

    path = _viterbi(scores, offsets, jump_penalty, anchor_penalty)
    chosen = offsets[path]
    snapped = base + chosen[:, None] * normals
    snapped = smooth_polyline(snapped, window=smooth_window)

    return SnapResult(
        points=snapped,
        offsets=chosen,
        score_before=float(_sample(ridge, base).mean()),
        score_after=float(_sample(ridge, snapped).mean()),
    )
