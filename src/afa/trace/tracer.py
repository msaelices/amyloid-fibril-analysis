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
    merge_junction_px: float = 15.0,
) -> list[np.ndarray]:
    """Trace ordered centerlines from a binary fibril mask.

    Parameters
    ----------
    mask:
        2D boolean array; ``True`` where fibril is present.
    resample_step, smooth_window:
        Passed to the resample/smooth step (pixel units).
    min_branch_px:
        Drop free-ended skeleton branches shorter than this (removes spurs and
        noise). Junction-to-junction branches are exempt: they are structure,
        not noise.
    merge_junction_px:
        Junction-to-junction branches shorter than this are treated as the
        bridge of a single crossing and their two endpoints collapsed into one
        node. Set it to roughly the fibril width; too large merges genuinely
        distinct junctions.
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

    raw = []
    for i in range(skeleton.n_paths):
        coords_rc = skeleton.path_coordinates(i)  # (row, col)
        if len(coords_rc) < 2:
            continue
        raw.append(
            {
                "coords": coords_rc[:, ::-1].astype(float),  # -> (x, y)
                "src": int(summary.loc[i, "node_id_src"]),
                "dst": int(summary.loc[i, "node_id_dst"]),
                "length": float(summary.loc[i, "branch_distance"]),
                # skan branch types: 1 = junction-to-endpoint, 2 =
                # junction-to-junction. Only the former can be a noise spur.
                "internal": int(summary.loc[i, "branch_type"]) == 2,
                "used": False,
            }
        )

    node_of = _merge_close_junctions(raw, merge_junction_px)

    branches: list[dict] = []
    for b in raw:
        # A short junction-to-junction branch is the bridge of a crossing, not a
        # spur; it has been absorbed into a merged node and must not survive as
        # a separate fibril. Length filtering applies only to free-ended
        # branches, which are the actual noise.
        if b["internal"] and b["length"] < merge_junction_px:
            continue
        if not b["internal"] and b["length"] < min_branch_px:
            continue
        branches.append(
            {
                "coords": b["coords"],
                "src": node_of[b["src"]],
                "dst": node_of[b["dst"]],
                "used": False,
            }
        )

    if not branches:
        return []

    if not link_crossings:
        return [_finish(b["coords"], resample_step, smooth_window) for b in branches]

    return _link_and_finish(branches, resample_step, smooth_window, max_link_angle_deg)


def _merge_close_junctions(raw: list[dict], merge_junction_px: float) -> dict[int, int]:
    """Map each skeleton node to a representative, collapsing crossing bridges.

    Skeletonizing an X does not produce one node: it produces two Y-junctions
    joined by a short bridge. Left alone, the four fibril halves attach to two
    different nodes and no two halves of the same fibril ever share one, so the
    crossing can never be linked through. Worse, dropping that bridge as a
    "short branch" is what disconnects them.

    Union-find over short junction-to-junction branches collapses the pair into
    a single node, after which all four halves meet at one place and the
    orientation test can pair them up.
    """
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for b in raw:
        parent.setdefault(b["src"], b["src"])
        parent.setdefault(b["dst"], b["dst"])
        if b["internal"] and b["length"] < merge_junction_px:
            ra, rb = find(b["src"]), find(b["dst"])
            if ra != rb:
                parent[rb] = ra

    return {node: find(node) for node in parent}


def _finish(coords: np.ndarray, step: float, window: int) -> np.ndarray:
    return smooth_polyline(resample_polyline(coords, step=step), window=window)


def _link_and_finish(
    branches: list[dict],
    step: float,
    window: int,
    max_link_angle_deg: float,
) -> list[np.ndarray]:
    """Greedily chain branches that share a node and continue smoothly.

    Convention that makes the angle test work: ``ref_dir`` and ``cdir`` both
    point *away* from the shared junction. Two branches that continue each other
    straight through are then antiparallel, so ``dot(ref_dir, -cdir) == 1`` and a
    turn of ``theta`` degrees gives ``cos(theta)``. The acceptance threshold is
    therefore ``cos(max_link_angle_deg)``.
    """
    max_cos = np.cos(np.deg2rad(max_link_angle_deg))  # near +1 for a small turn

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
        # Direction of the current chain at the node, pointing AWAY from it so
        # that it is directly comparable with the candidates' directions.
        tail = np.asarray(chain[-5:] if forward else chain[:5])
        ref_dir = _branch_direction(tail, at_start=not forward)
        if forward:
            ref_dir = -ref_dir

        best, best_cos, best_at_start = None, -2.0, False
        for j in candidates:
            cand = branches[j]
            at_start = cand["src"] == node
            cdir = _branch_direction(cand["coords"], at_start=at_start)
            if not at_start:
                # The branch meets the node by its far end, so that direction
                # points into the node; flip it to point away like the others.
                cdir = -cdir
            # Both vectors now point away from the node, so a straight
            # continuation is antiparallel and this is the cosine of the turn.
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
