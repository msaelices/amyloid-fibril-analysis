"""Tests for joining centerline fragments across gaps in the mask."""

from __future__ import annotations

import numpy as np
import pytest

from afa.trace.bridge import bridge_gaps


def _seg(x0, x1, y, n=40):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, float(y))])


def _lengths(chains):
    return sorted(
        float(np.linalg.norm(np.diff(np.asarray(c), axis=0), axis=1).sum()) for c in chains
    )


def test_rejoins_a_fibril_broken_into_pieces():
    pieces = [_seg(0, 90, 0), _seg(110, 200, 0), _seg(220, 300, 0)]

    out = bridge_gaps(pieces, max_gap_px=30.0, max_angle_deg=20.0)

    assert len(out) == 1
    assert _lengths(out)[0] == pytest.approx(300, abs=5)


def test_leaves_a_gap_that_is_too_long():
    pieces = [_seg(0, 90, 0), _seg(300, 400, 0)]
    assert len(bridge_gaps(pieces, max_gap_px=30.0)) == 2


def test_does_not_weld_two_fibrils_that_merely_end_near_each_other():
    """Ends that are close but point across each other must stay separate."""
    horizontal = _seg(0, 100, 0)
    vertical = np.column_stack([np.full(40, 110.0), np.linspace(0, 100, 40)])

    out = bridge_gaps([horizontal, vertical], max_gap_px=30.0, max_angle_deg=30.0)

    assert len(out) == 2


def test_does_not_join_parallel_neighbours():
    """Two fibrils side by side are close but not collinear."""
    a = _seg(0, 100, 0)
    b = _seg(0, 100, 12)

    out = bridge_gaps([a, b], max_gap_px=40.0, max_angle_deg=30.0)

    assert len(out) == 2


def test_joins_regardless_of_fragment_orientation():
    """Fragments may be stored end-to-start; the join must handle every pairing."""
    forward = _seg(0, 90, 0)
    backward = _seg(200, 110, 0)  # same fibril, reversed order

    out = bridge_gaps([forward, backward], max_gap_px=30.0, max_angle_deg=20.0)

    assert len(out) == 1
    assert _lengths(out)[0] == pytest.approx(200, abs=5)


def test_follows_a_gentle_curve_across_a_gap():
    xs = np.linspace(0, 300, 300)
    ys = 0.0006 * (xs - 150) ** 2
    curve = np.column_stack([xs, ys])
    pieces = [curve[:120], curve[160:]]

    out = bridge_gaps(pieces, max_gap_px=50.0, max_angle_deg=25.0)

    assert len(out) == 1


def test_disabled_and_degenerate_inputs_are_returned_unchanged():
    pieces = [_seg(0, 90, 0), _seg(110, 200, 0)]
    assert len(bridge_gaps(pieces, max_gap_px=0.0)) == 2
    assert bridge_gaps([], max_gap_px=30.0) == []
    assert len(bridge_gaps([_seg(0, 90, 0)], max_gap_px=30.0)) == 1


def test_a_fibril_in_many_pieces_needs_several_rounds():
    pieces = [_seg(i * 40, i * 40 + 30, 0) for i in range(8)]

    one_round = bridge_gaps(pieces, max_gap_px=20.0, max_angle_deg=20.0, max_rounds=1)
    many = bridge_gaps(pieces, max_gap_px=20.0, max_angle_deg=20.0, max_rounds=8)

    assert len(one_round) == 4
    assert len(many) == 1
