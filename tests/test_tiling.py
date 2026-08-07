"""Tests for patch tiling and prediction stitching."""

from __future__ import annotations

import numpy as np
import pytest

from afa.segment.tiling import blend_window, extract, stitch, tile_positions


def test_positions_cover_every_pixel():
    shape = (300, 421)
    patch, overlap = 128, 32
    positions = tile_positions(shape, patch, overlap)

    covered = np.zeros(shape, dtype=bool)
    for y, x in positions:
        covered[y:y + patch, x:x + patch] = True
    assert covered.all()
    assert all(y + patch <= shape[0] and x + patch <= shape[1] for y, x in positions)


def test_single_patch_when_image_is_smaller():
    assert tile_positions((64, 64), 128, 32) == [(0, 0)]


def test_overlap_must_be_smaller_than_patch():
    with pytest.raises(ValueError):
        tile_positions((256, 256), 64, 64)


def test_blend_window_is_zero_free_and_peaks_in_the_middle():
    w = blend_window(64)
    assert w.shape == (64, 64)
    assert (w > 0).all()
    assert w[32, 32] == pytest.approx(1.0)
    assert w[0, 0] < 0.05


def test_stitch_reconstructs_a_constant_map():
    shape = (200, 260)
    patch, overlap = 64, 16
    positions = tile_positions(shape, patch, overlap)
    patches = [np.full((patch, patch), 0.7, dtype=np.float32) for _ in positions]

    out = stitch(patches, positions, shape)

    assert out.shape == shape
    assert np.allclose(out, 0.7, atol=1e-5)


def test_stitch_reconstructs_a_smooth_gradient_without_seams():
    shape = (192, 192)
    patch, overlap = 64, 32
    ramp = np.linspace(0, 1, shape[1], dtype=np.float32)[None, :]
    field = ramp * np.ones((shape[0], 1), np.float32)
    positions = tile_positions(shape, patch, overlap)
    patches = list(extract(field, positions, patch))

    out = stitch(patches, positions, shape)

    assert np.abs(out - field).max() < 1e-4


def test_stitch_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        stitch([np.zeros((8, 8))], [(0, 0), (0, 8)], (16, 16))
