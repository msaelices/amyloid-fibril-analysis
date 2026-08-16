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

from collections import defaultdict

import numpy as np

from afa.trace.bridge import bridge_gaps
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
    mask_opening_px: int = 0,
    link_crossings: bool = True,
    max_link_angle_deg: float = 60.0,
    merge_junction_px: float = 15.0,
    bridge_gap_px: float = 0.0,
    bridge_angle_deg: float = 30.0,
    evidence: np.ndarray | None = None,
    min_bridge_evidence: float = 0.25,
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
    mask_opening_px:
        Radius of a morphological opening applied to the mask *before*
        skeletonizing. Zero disables it.

        This attacks the cause rather than the symptom. Skeletonizing the raw
        thresholded mask produces a hairball: measured on held-out images, 655
        to 1122 branches with a median length of 5 to 12 px, against a median
        manual fibril of 430 px. The mask edge is ragged, and every bump becomes
        a branch, so the linking stage is asked to rebuild one fibril out of
        roughly fifty fragments.

        Opening erodes then dilates, removing those bumps and the thin spurious
        connections between neighbouring fibrils. Closing, the opposite
        operation and the intuitive guess, measured neutral to worse.
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
    bridge_gap_px:
        After junction linking, join fragments whose ends face each other across
        a gap of at most this many pixels (see :mod:`afa.trace.bridge`). Zero
        disables it, which is the default: without ``evidence`` the join rests
        on geometry alone, and two different fibrils lying end to end on one
        line satisfy every geometric test. The pipeline enables it and supplies
        evidence; a bare call on a mask does not.
    bridge_angle_deg:
        Collinearity tolerance for that join.
    evidence:
        Detector output backing the mask (a probability map). Supplying it lets
        the bridging step check that the image actually supports each join.
    min_bridge_evidence:
        Minimum mean ``evidence`` along a bridge for it to be made.

    Returns
    -------
    list of ``(N, 2)`` arrays of ``(x, y)`` centerline vertices.
    """
    from skan import Skeleton, summarize  # noqa: PLC0415
    from skimage.morphology import binary_opening, disk, skeletonize  # noqa: PLC0415

    binary = np.asarray(mask, dtype=bool)
    if mask_opening_px > 0:
        binary = binary_opening(binary, disk(mask_opening_px))
    skel_img = skeletonize(binary)
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

    if link_crossings:
        centerlines = _link_and_finish(
            branches, resample_step, smooth_window, max_link_angle_deg
        )
    else:
        centerlines = [_finish(b["coords"], resample_step, smooth_window) for b in branches]

    if bridge_gap_px > 0:
        centerlines, merged = bridge_gaps(
            centerlines,
            max_gap_px=bridge_gap_px,
            max_angle_deg=bridge_angle_deg,
            evidence=evidence,
            min_evidence=min_bridge_evidence,
        )
        # Only the joined chains need resampling and smoothing again. Running it
        # over the untouched ones smooths them a second time, which leaves their
        # length alone but roughly halves their curvature -- silently rescaling
        # a reported metric for traces that were never bridged at all.
        centerlines = [
            _finish(c, resample_step, smooth_window) if was_merged else c
            for c, was_merged in zip(centerlines, merged, strict=True)
        ]
    return centerlines


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


def _end_direction(branch: dict, at_start: bool, k: int = 5) -> np.ndarray:
    """Unit direction of a branch at one end, pointing AWAY from the node there."""
    d = _branch_direction(branch["coords"], at_start=at_start, k=k)
    return d if at_start else -d


def _best_matching(quality: dict[tuple[int, int], float], ends: list[int]) -> list[tuple[int, int]]:
    """Maximum-weight pairing of branch ends meeting at one node.

    Exhaustive for the degrees that occur in practice (3 to 6 branches at a
    crossing), greedy above that. Junction degree is tiny, so the exhaustive
    search is cheap and removes the ordering sensitivity that made the greedy
    walk's result depend on which branch it happened to start from.
    """
    if len(ends) > 8:
        chosen, used = [], set()
        for (a, b), _ in sorted(quality.items(), key=lambda kv: -kv[1]):
            if a not in used and b not in used:
                chosen.append((a, b))
                used |= {a, b}
        return chosen

    best_score, best_pairs = 0.0, []

    def recurse(remaining: tuple[int, ...], pairs: list, score: float) -> None:
        nonlocal best_score, best_pairs
        if score > best_score:
            best_score, best_pairs = score, list(pairs)
        if len(remaining) < 2:
            return
        head, rest = remaining[0], remaining[1:]
        # Either leave `head` unpaired (a chain terminates here) ...
        recurse(rest, pairs, score)
        # ... or pair it with any admissible partner.
        for i, other in enumerate(rest):
            key = (head, other) if head < other else (other, head)
            if key in quality:
                pairs.append(key)
                recurse(rest[:i] + rest[i + 1:], pairs, score + quality[key])
                pairs.pop()

    recurse(tuple(ends), [], 0.0)
    return best_pairs


def _link_and_finish(
    branches: list[dict],
    step: float,
    window: int,
    max_link_angle_deg: float,
) -> list[np.ndarray]:
    """Chain branches by pairing their ends optimally at each junction.

    Convention that makes the angle test work: every direction points *away*
    from the shared node, so two branches continuing each other straight are
    antiparallel and ``-dot(a, b)`` is the cosine of the turn. A pair is
    admissible when that exceeds ``cos(max_link_angle_deg)``.

    The pairing is solved per node rather than walked greedily. A fibril
    crossing a dense field has to win a decision at every junction it passes,
    and under a greedy walk the winner depends on which branch the iteration
    started from, so a fibril could lose a crossing to a branch that merely got
    there first.
    """
    min_cos = float(np.cos(np.deg2rad(max_link_angle_deg)))

    # Every branch has two ends; end 2i is branch i's start, 2i+1 its end.
    by_node: dict[int, list[int]] = defaultdict(list)
    for idx, b in enumerate(branches):
        by_node[b["src"]].append(2 * idx)
        by_node[b["dst"]].append(2 * idx + 1)

    partner: dict[int, int] = {}
    for ends in by_node.values():
        if len(ends) < 2:
            continue
        dirs = {e: _end_direction(branches[e // 2], at_start=(e % 2 == 0)) for e in ends}
        quality = {}
        for i, a in enumerate(ends):
            for b in ends[i + 1:]:
                if a // 2 == b // 2:
                    continue  # a branch may not be joined to itself
                cos_turn = float(-np.dot(dirs[a], dirs[b]))
                if cos_turn >= min_cos:
                    # Normalized so an admissible pair scores in (0, 1]: 1 for a
                    # perfectly straight continuation, ~0 at the angle limit.
                    # Raw cosines would let two marginal links outscore one
                    # straight one, and would go negative past 90 degrees, where
                    # the empty matching then wins and linking silently stops.
                    quality[(a, b)] = (cos_turn - min_cos) / max(1.0 - min_cos, 1e-9)
        for a, b in _best_matching(quality, ends):
            partner[a] = b
            partner[b] = a

    # Walk the pairings to build chains. Start from unpaired ends first so that
    # open fibrils come out whole; whatever remains is a closed loop.
    centerlines: list[np.ndarray] = []
    visited: set[int] = set()

    def walk(start_end: int) -> list[np.ndarray]:
        pieces, end = [], start_end
        while True:
            branch = end // 2
            if branch in visited:
                break
            visited.add(branch)
            coords = branches[branch]["coords"]
            pieces.append(coords if end % 2 == 0 else coords[::-1])
            far = branch * 2 + (1 if end % 2 == 0 else 0)
            nxt = partner.get(far)
            if nxt is None:
                break
            end = nxt
        return pieces

    for e in range(2 * len(branches)):
        if e in partner or e // 2 in visited:
            continue
        pieces = walk(e)
        if pieces:
            centerlines.append(_finish(np.vstack(pieces), step, window))

    for idx in range(len(branches)):
        if idx in visited:
            continue
        pieces = walk(2 * idx)
        if pieces:
            centerlines.append(_finish(np.vstack(pieces), step, window))

    return centerlines
