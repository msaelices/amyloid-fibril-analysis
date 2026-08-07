"""Analytic tests for the morphology metrics.

Each shape has a known closed-form answer, so these tests pin the metric
definitions to ground truth rather than to a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from afa.morphology.metrics import _wrap_to_pi, compute_metrics


def test_straight_line():
    pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
    m = compute_metrics(pts)
    assert m.length == pytest.approx(100.0, rel=1e-9)
    assert m.tortuosity == pytest.approx(1.0, rel=1e-9)
    assert m.max_curvature == pytest.approx(0.0, abs=1e-9)
    assert m.total_abs_turning == pytest.approx(0.0, abs=1e-9)


def test_right_angle_turn():
    # An L shape: 10 right, then 10 up. One 90-degree turn.
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
    m = compute_metrics(pts)
    assert m.length == pytest.approx(20.0, rel=1e-9)
    assert m.total_abs_turning == pytest.approx(np.pi / 2, rel=1e-9)
    # chord = sqrt(200); tortuosity = 20 / sqrt(200)
    assert m.tortuosity == pytest.approx(20.0 / np.sqrt(200.0), rel=1e-9)


def test_circle_curvature():
    r = 50.0
    phi = np.arange(0, np.pi / 2, 0.02)  # quarter circle
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    m = compute_metrics(pts)
    # Curvature of a circle is 1/r everywhere.
    assert m.mean_abs_curvature == pytest.approx(1.0 / r, rel=0.02)
    assert m.max_curvature == pytest.approx(1.0 / r, rel=0.05)
    # Arc length ~ r * (pi/2).
    assert m.length == pytest.approx(r * (phi[-1] - phi[0]), rel=0.01)


def test_pixel_size_scaling():
    pts = np.column_stack([np.linspace(0, 100, 60), np.zeros(60)])
    m1 = compute_metrics(pts, pixel_size=1.0)
    m2 = compute_metrics(pts, pixel_size=2.0)
    # Length scales linearly with pixel size; tortuosity is dimensionless.
    assert m2.length == pytest.approx(2.0 * m1.length, rel=1e-9)
    assert m2.tortuosity == pytest.approx(m1.tortuosity, rel=1e-9)


def test_curvature_scales_inversely_with_pixel_size():
    r = 40.0
    phi = np.arange(0, 1.0, 0.02)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    m1 = compute_metrics(pts, pixel_size=1.0)
    m2 = compute_metrics(pts, pixel_size=2.0)
    # Physical radius doubles => curvature halves.
    assert m2.mean_abs_curvature == pytest.approx(0.5 * m1.mean_abs_curvature, rel=1e-6)


def test_turning_per_length_consistency():
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)
    m = compute_metrics(pts)
    assert m.total_abs_turning_per_length == pytest.approx(
        m.total_abs_turning / m.length, rel=1e-9
    )


def test_duplicate_points_are_handled():
    pts = np.array([[0, 0], [0, 0], [5, 0], [5, 0], [10, 0]], dtype=float)
    m = compute_metrics(pts)
    assert m.length == pytest.approx(10.0, rel=1e-9)
    assert m.max_curvature == pytest.approx(0.0, abs=1e-9)


def test_wrap_to_pi():
    ang = np.array([0.0, np.pi + 0.1, -np.pi - 0.1, 3 * np.pi])
    wrapped = _wrap_to_pi(ang)
    assert np.all(wrapped > -np.pi - 1e-9)
    assert np.all(wrapped <= np.pi + 1e-9)
    assert wrapped[1] == pytest.approx(-np.pi + 0.1, rel=1e-6)
