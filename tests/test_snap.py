"""Tests for snapping hand-drawn offset traces onto the fibril ridge.

The manual traces were drawn beside the fibrils rather than on them, so this
step decides whether the ground truth is usable at all. These tests use
synthetic ridges where the correct answer is known exactly.
"""

from __future__ import annotations

import numpy as np

from afa.trace.snap import snap_to_ridge


def _ridge_image(shape=(200, 300), sigma=3.0):
    """A bright horizontal ridge at y = 100, Gaussian in cross-section."""
    yy = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    return np.exp(-((yy - 100.0) ** 2) / (2 * sigma**2)).astype(np.float32)


def test_snaps_straight_offset_trace_onto_ridge():
    ridge = _ridge_image()
    drawn = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 115.0)])  # 15 px below

    res = snap_to_ridge(drawn, ridge, max_shift=25.0)

    assert np.abs(res.points[:, 1] - 100.0).max() < 1.5
    assert res.score_after > res.score_before
    assert res.gain > 5.0


def test_snaps_from_either_side():
    ridge = _ridge_image()
    above = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 85.0)])

    res = snap_to_ridge(above, ridge, max_shift=25.0)

    assert np.abs(res.points[:, 1] - 100.0).max() < 1.5
    assert np.all(res.offsets > 0) or np.all(res.offsets < 0)  # consistent side


def test_follows_a_curved_ridge():
    shape = (240, 300)
    xs = np.arange(shape[1], dtype=float)
    centre = 120.0 + 40.0 * np.sin(xs / 60.0)
    yy = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    ridge = np.exp(-((yy - centre[None, :]) ** 2) / (2 * 3.0**2)).astype(np.float32)

    drawn = np.column_stack([xs[40:260:4], centre[40:260:4] + 14.0])
    res = snap_to_ridge(drawn, ridge, max_shift=25.0)

    truth = np.interp(res.points[:, 0], xs, centre)
    assert np.abs(res.points[:, 1] - truth).max() < 3.0


def test_leaves_an_already_centred_trace_alone():
    ridge = _ridge_image()
    drawn = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 100.0)])

    res = snap_to_ridge(drawn, ridge, max_shift=25.0)

    assert np.abs(res.offsets).max() <= 1.0
    assert np.abs(res.points[:, 1] - 100.0).max() < 1.0


def test_offset_spread_is_narrow_for_a_clean_snap():
    ridge = _ridge_image()
    drawn = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 115.0)])

    res = snap_to_ridge(drawn, ridge, max_shift=25.0)

    assert res.offset_spread < 1.0


def test_offset_spread_flags_a_trace_that_crosses_to_another_fibril():
    """gain cannot see this; offset_spread can."""
    shape = (260, 300)
    yy = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    ridge = np.exp(-((yy - 150.0) ** 2) / 18.0).astype(np.float32)
    # A competing fibril that only exists over the right half of the image.
    ridge[:, 150:] = np.maximum(
        ridge[:, 150:], np.exp(-((yy[:, 150:] - 128.0) ** 2) / 18.0) * 1.4
    )
    drawn = np.column_stack([np.arange(20, 280, 4.0), np.full(65, 140.0)])

    res = snap_to_ridge(drawn, ridge, max_shift=25.0)

    assert res.gain > 2.0, "the wrong-fibril case still looks great by gain alone"
    assert res.offset_spread > 3.0


def test_snapping_keeps_the_offset_it_was_drawn_at_rather_than_collapsing():
    """The anchor must not drag the trace back to where it was drawn."""
    ridge = _ridge_image()
    drawn = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 112.0)])

    res = snap_to_ridge(drawn, ridge, max_shift=25.0, anchor_penalty=0.01)

    assert np.abs(res.points[:, 1] - 100.0).max() < 2.0
    assert np.median(np.abs(res.offsets)) > 8.0


def test_flat_field_leaves_geometry_unchanged():
    """With no ridge to find, the trace must not wander off."""
    flat = np.full((200, 300), 0.5, dtype=np.float32)
    drawn = np.column_stack([np.arange(50, 250, 5.0), np.full(40, 115.0)])

    res = snap_to_ridge(drawn, flat, max_shift=25.0, jump_penalty=0.02)

    assert np.abs(res.offsets).max() < 1e-6
    assert np.isclose(res.gain, 1.0, atol=1e-3)
