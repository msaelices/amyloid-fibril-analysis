"""Join centerline fragments across gaps in the detected mask.

Why this exists
---------------
Junction linking (:mod:`afa.trace.tracer`) can only follow a fibril through a
place where the skeleton is *connected*. It cannot cross a gap. On real
micrographs the detected mask breaks wherever a fibril fades, is occluded, or
dips below threshold, and each break ends the chain at a free endpoint.

Measured on held-out images, that is the dominant failure: the median traced
fragment is 38 px against a median manual fibril of roughly 480 px, and pushing
the junction angle all the way to 90 degrees still only reaches 50 px, because
the chains are not stopping at junctions -- they are stopping at nothing at all.

Method
------
Each fragment contributes two endpoints, with an outgoing tangent estimated over
its last few vertices. Two endpoints are joinable when the gap between them is
short, each tangent points at the other endpoint, and the two fragments are
roughly collinear rather than merely near each other. Feasible pairs are joined
greedily, cheapest first, and the pass repeats so that a fibril broken into many
pieces is rebuilt in stages.

Requiring *both* tangents to agree with the connecting direction is what stops
two fibrils that happen to end near each other from being welded into one.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _endpoint_direction(points: np.ndarray, at_start: bool, k: int = 8) -> np.ndarray:
    """Unit tangent pointing *out* of the polyline at one of its ends."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return np.zeros(2)
    span = min(k, n - 1)
    seg = pts[0] - pts[span] if at_start else pts[-1] - pts[-1 - span]
    norm = np.linalg.norm(seg)
    return seg / norm if norm > 0 else np.zeros(2)


def _endpoints(centerlines: list[np.ndarray]) -> list[tuple[int, bool, np.ndarray, np.ndarray]]:
    """``(chain index, at_start, position, outgoing direction)`` for every end."""
    ends = []
    for i, c in enumerate(centerlines):
        pts = np.asarray(c, dtype=float)
        ends.append((i, True, pts[0], _endpoint_direction(pts, at_start=True)))
        ends.append((i, False, pts[-1], _endpoint_direction(pts, at_start=False)))
    return ends


def _gap_evidence(
    evidence: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray, samples: int = 16
) -> float:
    """Mean of ``evidence`` sampled along the straight segment between two ends."""
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = pos_a[None, :] * (1 - t) + pos_b[None, :] * t
    vals = map_coordinates(
        evidence, [pts[:, 1], pts[:, 0]], order=1, mode="constant", cval=0.0
    )
    return float(vals.mean())


def bridge_gaps(
    centerlines: list[np.ndarray],
    *,
    max_gap_px: float = 40.0,
    max_angle_deg: float = 30.0,
    max_rounds: int = 8,
    evidence: np.ndarray | None = None,
    min_evidence: float = 0.25,
) -> tuple[list[np.ndarray], list[bool]]:
    """Join fragments whose ends face each other across a short gap.

    Parameters
    ----------
    centerlines:
        Ordered ``(N, 2)`` polylines in ``(x, y)`` pixel coordinates.
    max_gap_px:
        Longest gap that may be bridged.
    max_angle_deg:
        Each fragment's outgoing tangent must lie within this angle of the line
        joining the two endpoints, and the two tangents must be that collinear
        with each other.
    max_rounds:
        Passes over the fragment set. Each pass joins disjoint pairs, so a
        fibril in a dozen pieces needs several.
    evidence:
        Detector output (a ``[0, 1]`` probability or likelihood map) used to
        check that the image actually supports the bridge. **Strongly
        recommended**: geometry alone cannot tell a break inside one fibril from
        the empty space between two different fibrils that happen to lie on the
        same line, and it welds the latter. Measured against the manual traces
        at a 60 px gap, 38% of purely geometric joins were between distinct
        fibrils.
    min_evidence:
        Minimum mean ``evidence`` along the connecting segment. Ignored when
        ``evidence`` is ``None``.

    Returns
    -------
    ``(chains, merged)`` where ``merged[i]`` is ``True`` if ``chains[i]`` is a
    join of two or more inputs. Callers that resample or smooth the result must
    do so only for those: re-running it over untouched fragments smooths them a
    second time and halves their curvature.
    """
    if max_gap_px <= 0 or len(centerlines) < 2:
        return list(centerlines), [False] * len(centerlines)

    min_cos = float(np.cos(np.deg2rad(max_angle_deg)))
    chains = [np.asarray(c, dtype=float) for c in centerlines]
    merged_flags = [False] * len(chains)

    for _ in range(max_rounds):
        ends = _endpoints(chains)
        candidates = []
        for a in range(len(ends)):
            ia, start_a, pos_a, dir_a = ends[a]
            for b in range(a + 1, len(ends)):
                ib, start_b, pos_b, dir_b = ends[b]
                if ia == ib:
                    continue
                delta = pos_b - pos_a
                gap = float(np.linalg.norm(delta))
                if gap > max_gap_px or gap == 0.0:
                    continue
                u = delta / gap
                # Each end must point at the other, and the two must be
                # collinear. Necessary but not sufficient: two different fibrils
                # lying end to end on one line pass all three tests, which is
                # what `evidence` is for.
                if float(np.dot(dir_a, u)) < min_cos:
                    continue
                if float(np.dot(dir_b, -u)) < min_cos:
                    continue
                if float(np.dot(dir_a, -dir_b)) < min_cos:
                    continue
                if evidence is not None:
                    if _gap_evidence(evidence, pos_a, pos_b) < min_evidence:
                        continue
                candidates.append((gap, ia, start_a, ib, start_b))

        if not candidates:
            break

        candidates.sort()
        used_ends: set[tuple[int, bool]] = set()
        used_chains: set[int] = set()
        merged: list[np.ndarray] = []
        consumed: set[int] = set()

        for _gap, ia, start_a, ib, start_b in candidates:
            if (ia, start_a) in used_ends or (ib, start_b) in used_ends:
                continue
            if ia in used_chains or ib in used_chains:
                continue
            first = chains[ia][::-1] if start_a else chains[ia]
            second = chains[ib] if start_b else chains[ib][::-1]
            merged.append(np.vstack([first, second]))
            used_ends |= {(ia, start_a), (ib, start_b)}
            used_chains |= {ia, ib}
            consumed |= {ia, ib}

        if not consumed:
            break
        kept = [i for i in range(len(chains)) if i not in consumed]
        merged_flags = [True] * len(merged) + [merged_flags[i] for i in kept]
        chains = merged + [chains[i] for i in kept]

    return chains, merged_flags
