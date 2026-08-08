"""Tests for junction linking in the skeleton tracer.

The whole point of the linking step is that a fibril passing under another is
followed *through* the crossing instead of being truncated or spliced onto the
wrong branch. These tests pin the angle convention, which was silently inverted:
collinear continuations were rejected and near-180-degree doubling back accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from afa.trace.tracer import trace_centerlines


def _draw(mask: np.ndarray, p0, p1, width: int = 1) -> None:
    from skimage.draw import line

    rr, cc = line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            y, x = np.clip(rr + dy, 0, mask.shape[0] - 1), np.clip(cc + dx, 0, mask.shape[1] - 1)
            mask[y, x] = True


def _lengths(centerlines):
    return sorted(
        float(np.linalg.norm(np.diff(np.asarray(c), axis=0), axis=1).sum())
        for c in centerlines
    )


def test_straight_fibril_survives_a_spur():
    """A sharp spur must not be linked; the straight run stays one centerline."""
    mask = np.zeros((300, 400), dtype=bool)
    _draw(mask, (20, 200), (380, 200))     # straight fibril
    _draw(mask, (200, 200), (182, 100))    # steep spur off its middle

    out = trace_centerlines(mask, min_branch_px=15, link_crossings=True)
    lengths = _lengths(out)

    # The long fibril is recovered whole, not doubled back on itself.
    assert max(lengths) == pytest.approx(360, abs=25)
    assert max(lengths) < 500, f"a doubled-back chain appeared: {lengths}"


def test_crossing_fibrils_are_followed_through():
    """Two fibrils crossing at a shallow angle stay two, each end to end.

    Their true lengths are deliberately different (400 vs ~284 px) so that the
    old failure mode is detectable: splicing half of one onto half of the other
    produced two near-equal middling lengths instead.
    """
    mask = np.zeros((400, 500), dtype=bool)
    _draw(mask, (50, 200), (450, 200))     # horizontal, 400 px
    _draw(mask, (150, 80), (330, 300))     # diagonal crossing it, ~284 px

    out = trace_centerlines(mask, min_branch_px=15, link_crossings=True)
    lengths = _lengths(out)

    assert len(out) == 2, f"expected 2 fibrils, got {len(out)}: {lengths}"
    assert lengths[0] == pytest.approx(284, abs=30), lengths
    assert lengths[1] == pytest.approx(400, abs=30), lengths


def test_linking_off_leaves_the_crossing_fragmented():
    """Control: without linking the same X yields four branch fragments.

    Gap bridging is disabled too. It is a separate step that joins collinear
    fragments without needing the skeleton to be connected, so with it on the
    two halves of a fibril are rejoined even when junction linking is off, and
    this control would not measure what it means to.
    """
    mask = np.zeros((400, 500), dtype=bool)
    _draw(mask, (50, 200), (450, 200))
    _draw(mask, (150, 80), (330, 300))

    out = trace_centerlines(
        mask, min_branch_px=15, link_crossings=False, bridge_gap_px=0.0
    )

    assert len(out) == 4


def test_bridging_rejoins_a_fibril_the_mask_breaks_in_two():
    """A break in the mask disconnects the skeleton; only bridging can cross it."""
    mask = np.zeros((200, 500), dtype=bool)
    _draw(mask, (40, 100), (230, 100))
    _draw(mask, (270, 100), (460, 100))   # same fibril, 40 px of it missing

    without = trace_centerlines(mask, min_branch_px=15, bridge_gap_px=0.0)
    with_bridge = trace_centerlines(mask, min_branch_px=15, bridge_gap_px=60.0)

    assert len(without) == 2
    assert len(with_bridge) == 1
    assert _lengths(with_bridge)[0] == pytest.approx(420, abs=25)


def test_single_fibril_is_untouched():
    mask = np.zeros((200, 400), dtype=bool)
    _draw(mask, (20, 100), (380, 100))

    out = trace_centerlines(mask, min_branch_px=15, link_crossings=True)

    assert len(out) == 1
    assert _lengths(out)[0] == pytest.approx(360, abs=15)


def test_bridging_does_not_resmooth_untouched_centerlines():
    """Enabling bridging must not change traces that were never bridged.

    _finish was being re-applied to the whole list, which leaves length alone but
    roughly halves curvature -- silently rescaling a reported metric.
    """
    from afa.morphology.metrics import compute_metrics

    mask = np.zeros((300, 500), dtype=bool)
    _draw(mask, (40, 60), (300, 240))
    _draw(mask, (60, 250), (460, 160))

    off = trace_centerlines(mask, min_branch_px=20, bridge_gap_px=0.0)
    # A tolerance at which no join is geometrically possible.
    on = trace_centerlines(mask, min_branch_px=20, bridge_gap_px=1e-9)

    assert len(off) == len(on)
    for a, b in zip(sorted(off, key=len), sorted(on, key=len), strict=True):
        ma, mb = compute_metrics(a), compute_metrics(b)
        assert ma.length == pytest.approx(mb.length, rel=1e-6)
        assert ma.max_curvature == pytest.approx(mb.max_curvature, rel=1e-6)
        assert ma.total_abs_turning == pytest.approx(mb.total_abs_turning, rel=1e-6)


def test_a_wide_angle_tolerance_loosens_linking_instead_of_disabling_it():
    """Past 90 degrees the raw cosine goes negative and the empty matching won."""
    from afa.trace.tracer import _best_matching

    # Four ends meeting at 110 degrees, well inside a 120 degree tolerance.
    min_cos = float(np.cos(np.deg2rad(120.0)))
    cos_turn = float(np.cos(np.deg2rad(110.0)))
    weight = (cos_turn - min_cos) / (1.0 - min_cos)
    quality = {(0, 2): weight, (4, 6): weight}

    assert weight > 0
    assert len(_best_matching(quality, [0, 2, 4, 6])) == 2


def test_one_straight_continuation_beats_two_marginal_ones():
    """Summed raw cosines let two links at the limit outscore a straight one."""
    from afa.trace.tracer import _best_matching

    min_cos = float(np.cos(np.deg2rad(60.0)))
    def w(turn_deg):
        return (float(np.cos(np.deg2rad(turn_deg))) - min_cos) / (1.0 - min_cos)

    quality = {(0, 2): w(1.0), (0, 4): w(59.0), (2, 6): w(59.0)}

    assert _best_matching(quality, [0, 2, 4, 6]) == [(0, 2)]
