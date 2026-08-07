"""Tests for arc-length resampling and per-patient aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from afa.stats import summarize_per_patient
from afa.trace.resample import resample_polyline, smooth_polyline


def test_resample_uniform_spacing():
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)  # length 20
    rs = resample_polyline(pts, step=1.0)
    seg = np.linalg.norm(np.diff(rs, axis=0), axis=1)
    # Spacing should be close to the requested step and roughly uniform.
    assert np.allclose(seg, seg[0], atol=0.2)
    total = np.linalg.norm(np.diff(rs, axis=0), axis=1).sum()
    assert abs(total - 20.0) < 0.5


def test_smooth_preserves_endpoints():
    pts = np.column_stack([np.arange(20), np.zeros(20)]).astype(float)
    sm = smooth_polyline(pts, window=5)
    assert np.allclose(sm[0], pts[0])
    assert np.allclose(sm[-1], pts[-1])


def test_summarize_per_patient_basic():
    df = pd.DataFrame(
        {
            "patient_id": ["P1"] * 4 + ["P2"] * 3,
            "image_id": ["a", "a", "b", "b", "c", "c", "c"],
            "length": [10.0, 12.0, 14.0, 16.0, 20.0, 22.0, 24.0],
            "tortuosity": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            "max_curvature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "mean_abs_curvature": [0.01] * 7,
            "total_abs_turning": [1.0] * 7,
            "total_abs_turning_per_length": [0.1] * 7,
            "local_dir_change_per_length": [0.01] * 7,
        }
    )
    summary = summarize_per_patient(df)
    p1_len = summary[(summary.patient_id == "P1") & (summary.metric == "length")].iloc[0]
    assert p1_len["n"] == 4
    assert p1_len["mean"] == 13.0
    assert p1_len["ci95_low"] < p1_len["mean"] < p1_len["ci95_high"]


def test_bootstrap_ci_runs():
    df = pd.DataFrame(
        {
            "patient_id": ["P1"] * 6,
            "image_id": ["a", "a", "b", "b", "c", "c"],
            "length": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    summary = summarize_per_patient(df, metrics=["length"], bootstrap=True)
    row = summary.iloc[0]
    assert row["ci95_low"] <= row["mean"] <= row["ci95_high"]
