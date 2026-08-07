"""Skeleton -> graph -> ordered centerlines, with orientation-aware crossings.

The tracer takes a binary fibril mask (from a threshold on the classical
vesselness map, from a U-Net probability map, or from a burned-in overlay) and
returns a list of ordered centerlines (one per fibril).

Pipeline
--------
1. Skeletonize the mask.
2. Build a graph of the skeleton with ``skan`` (nodes = endpoints/junctions,
   edges = branch paths).
3. Walk the graph, resolving junctions (crossings) by **orientation
   continuity**: at a junction, the branch whose direction best continues the
   incoming branch is chosen, so a fibril is followed through a crossing instead
   of being truncated.
4. Resample + smooth each centerline (see :mod:`afa.trace.resample`).

Notes
-----
``skan`` is used for robust skeleton-to-graph conversion. The junction logic
here is intentionally simple and greedy; publication-quality tracing typically
keeps a human in the loop to confirm ambiguous crossings.
"""

from __future__ import annotations

import numpy as np

from afa.trace.resample import resample_polyline, smooth_polyline


def _branch_direction(coords: np.ndarray, at_start: bool, k: int = 5) -> np.ndarray:
    """Unit direction of a branch near one of its ends."""
    if at_start:
        seg = coords[min(k, len(coords) - 1)] - coords[0]
    else:
        seg = coords[-1] - coords[max(-k - 1, -len(coords))]
    norm = np.linalg.norm(seg)
    return seg / norm if norm > 0 else np.zeros(2)


def trace_centerlines(
    mask: np.ndarray,
    *,
    resample_step: float = 1.0,
    smooth_window: int = 5,
    min_branch_px: int = 10,
    link_crossings: bool = True,
    max_link_angle_deg: float = 35.0,
) -> list[np.ndarray]:
    """Trace ordered centerlines from a binary fibril mask.

    Parameters
    ----------
    mask:
        2D boolean array; ``True`` where fibril is present.
    resample_step, smooth_window:
        Passed to the resample/smooth step (pixel units).
    min_branch_px:
        Drop skeleton branches shorter than this (removes spurs/noise).
    link_crossings:
        If ``True``, greedily link branches across junctions by orientation.
    max_link_angle_deg:
        Two branches at a junction are linked only if the turn between them is
        below this angle (i.e. they are roughly collinear).

    Returns
    -------
    list of ``(N, 2)`` arrays of ``(x, y)`` centerline vertices.
    """
    from skan import Skeleton, summarize
    from skimage.morphology import skeletonize

    skel_img = skeletonize(np.asarray(mask, dtype=bool))
    if skel_img.sum() == 0:
        return []

    skeleton = Skeleton(skel_img)
    summary = summarize(skeleton, separator="_")

    # Collect branch paths as (x, y) polylines in image coordinates.
    branches: list[dict] = []
    for i in range(skeleton.n_paths):
        coords_rc = skeleton.path_coordinates(i)  # (row, col)
        if len(coords_rc) < 2:
            continue
        length_px = float(summary.loc[i, "branch_distance"])
        if length_px < min_branch_px:
            continue
        coords_xy = coords_rc[:, ::-1].astype(float)  # -> (x, y)
        branches.append(
            {
                "coords": coords_xy,
                "src": int(summary.loc[i, "node_id_src"]),
                "dst": int(summary.loc[i, "node_id_dst"]),
                "used": False,
            }
        )

    if not branches:
        return []

    if not link_crossings:
        return [_finish(b["coords"], resample_step, smooth_window) for b in branches]

    return _link_and_finish(branches, resample_step, smooth_window, max_link_angle_deg)


def _finish(coords: np.ndarray, step: float, window: int) -> np.ndarray:
    return smooth_polyline(resample_polyline(coords, step=step), window=window)


def _link_and_finish(
    branches: list[dict],
    step: float,
    window: int,
    max_link_angle_deg: float,
) -> list[np.ndarray]:
    """Greedily chain branches that share a node and continue smoothly."""
    max_cos = np.cos(np.deg2rad(180.0 - max_link_angle_deg))  # near -1 for small angle

    # Index branches by the nodes they touch.
    from collections import defaultdict

    by_node: dict[int, list[int]] = defaultdict(list)
    for idx, b in enumerate(branches):
        by_node[b["src"]].append(idx)
        by_node[b["dst"]].append(idx)

    centerlines: list[np.ndarray] = []

    for start in branches:
        if start["used"]:
            continue
        start["used"] = True
        chain = list(start["coords"])
        # Extend from the destination end forward.
        _extend(chain, start, "dst", branches, by_node, max_cos, forward=True)
        # Extend from the source end backward.
        _extend(chain, start, "src", branches, by_node, max_cos, forward=False)
        centerlines.append(_finish(np.asarray(chain), step, window))

    return centerlines


def _extend(chain, current, node_key, branches, by_node, max_cos, *, forward):
    """Walk from ``current`` branch across a junction, appending collinear branches."""
    while True:
        node = current[node_key]
        candidates = [j for j in by_node[node] if not branches[j]["used"]]
        if not candidates:
            return
        # Direction of the current chain approaching the node.
        tail = np.asarray(chain[-5:] if forward else chain[:5])
        ref_dir = _branch_direction(tail, at_start=not forward)

        best, best_cos = None, -2.0
        for j in candidates:
            cand = branches[j]
            at_start = cand["src"] == node
            cdir = _branch_direction(cand["coords"], at_start=at_start)
            # Collinear continuation => ref_dir and cdir roughly opposite.
            cos = float(np.dot(ref_dir, -cdir))
            if cos > best_cos:
                best, best_cos, best_at_start = j, cos, at_start
        if best is None or best_cos < max_cos:
            return

        cand = branches[best]
        cand["used"] = True
        seg = cand["coords"] if best_at_start else cand["coords"][::-1]
        if forward:
            chain.extend(seg[1:])
        else:
            for p in seg[1:]:
                chain.insert(0, p)
        # Continue from the far end of the appended branch.
        current = {"src": cand["dst"] if best_at_start else cand["src"],
                   "dst": cand["src"] if best_at_start else cand["dst"]}
        node_key = "dst" if forward else "src"
