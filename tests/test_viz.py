"""Tests for overlay rendering.

Mostly a smoke test: rendering pulls in matplotlib APIs that have churned across
releases (``cm.get_cmap`` was removed in 3.9), and a silent failure here breaks
the visual-validation step that every other conclusion depends on.
"""

from __future__ import annotations

import numpy as np

from afa.viz import save_overlay


def test_save_overlay_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.normal(0.5, 0.1, size=(64, 64)).astype(np.float32)
    centerlines = [
        np.column_stack([np.arange(10, 50), np.full(40, 20.0)]),
        np.column_stack([np.full(40, 30.0), np.arange(10, 50)]),
    ]

    out = save_overlay(image, centerlines, tmp_path / "overlay.png", labels=["44", "45"])

    assert out.exists()
    assert out.stat().st_size > 0


def test_save_overlay_handles_no_centerlines(tmp_path):
    image = np.zeros((32, 32), dtype=np.float32)
    out = save_overlay(image, [], tmp_path / "empty.png")
    assert out.exists()
