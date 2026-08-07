"""Tests for the automatic-vs-manual validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from afa.validate import (
    centerline_coverage,
    centerline_distance,
    compare_metrics,
    coverage_report,
    dice,
    iou,
    match_traces,
    summarize_errors,
)


def _line(x0, x1, y, n=60):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, float(y))])


def test_dice_and_iou_extremes():
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    assert dice(a, a) == pytest.approx(1.0)
    assert iou(a, a) == pytest.approx(1.0)
    assert dice(a, np.zeros_like(a)) == pytest.approx(0.0)
    assert dice(np.zeros_like(a), np.zeros_like(a)) == pytest.approx(1.0)


def test_dice_half_overlap():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True   # 16 px
    b[2:6, 0:4] = True   # 16 px, 8 shared
    assert dice(a, b) == pytest.approx(2 * 8 / 32)
    assert iou(a, b) == pytest.approx(8 / 24)


def test_centerline_distance_is_symmetric_and_penalizes_fragments():
    full = _line(0, 100, 0)
    shifted = _line(0, 100, 5)
    assert centerline_distance(full, shifted) == pytest.approx(5.0, abs=0.1)
    assert centerline_distance(full, shifted) == pytest.approx(
        centerline_distance(shifted, full)
    )

    # A short fragment lying on the line is close one way but far the other.
    fragment = _line(0, 10, 0)
    assert centerline_distance(fragment, full) > 15.0


def test_match_pairs_each_manual_fibril_once():
    gt = [_line(0, 100, 0), _line(0, 100, 60)]
    pred = [_line(0, 100, 2), _line(0, 100, 61), _line(0, 100, 300)]

    m = match_traces(pred, gt, max_distance=10.0)

    assert len(m.pairs) == 2
    assert m.recall == pytest.approx(1.0)
    assert m.unmatched_pred == [2]
    assert m.precision_lower_bound == pytest.approx(2 / 3)
    assert m.detections_per_matched_fibril == pytest.approx(3 / 2)


def test_match_rejects_far_assignments():
    gt = [_line(0, 100, 0)]
    pred = [_line(0, 100, 80)]
    m = match_traces(pred, gt, max_distance=10.0)
    assert m.pairs == []
    assert m.recall == pytest.approx(0.0)
    assert m.unmatched_gt == [0]


def test_match_maximizes_the_number_of_matches_not_total_distance():
    """A min-total assignment can drop a feasible pair to save distance elsewhere.

    These three centerlines produce a cost matrix where the cheapest total
    assignment includes one pair just over the threshold, which is then
    discarded, reporting 2 matches where 3 are feasible.
    """
    gt = [_line(0, 100, 0), _line(0, 100, 400), _line(0, 100, 800)]
    pred = [_line(0, 100, 9), _line(0, 100, 409), _line(0, 100, 806)]

    m = match_traces(pred, gt, max_distance=15.0)

    assert len(m.pairs) == 3
    assert m.recall == pytest.approx(1.0)


def test_match_handles_empty_sides():
    gt = [_line(0, 100, 0)]
    assert match_traces([], gt).recall == pytest.approx(0.0)
    assert np.isnan(match_traces([], []).recall)


def test_compare_metrics_reports_zero_error_for_identical_traces():
    gt = [_line(0, 100, 0)]
    m = match_traces(list(gt), gt, max_distance=1.0)
    comp = compare_metrics(list(gt), gt, m)

    assert len(comp) == 1
    assert comp["length_abs_error"].iloc[0] == pytest.approx(0.0)
    assert comp["length_gt"].iloc[0] == pytest.approx(100.0)

    errors = summarize_errors(comp)
    length_row = errors[errors.metric == "length"].iloc[0]
    assert length_row["mean_abs_error"] == pytest.approx(0.0)
    assert length_row["n"] == 1


def test_compare_metrics_scales_with_pixel_size():
    gt = [_line(0, 100, 0)]
    m = match_traces(list(gt), gt, max_distance=1.0)
    comp = compare_metrics(list(gt), gt, m, pixel_size=0.5)
    assert comp["length_gt"].iloc[0] == pytest.approx(50.0)


def test_coverage_is_total_for_an_exact_detection():
    gt = _line(0, 100, 0)
    assert centerline_coverage(gt, [gt], tolerance=2.0) == pytest.approx(1.0)


def test_coverage_is_zero_for_a_distant_detection():
    gt = _line(0, 100, 0)
    assert centerline_coverage(gt, [_line(0, 100, 50)], tolerance=5.0) == pytest.approx(0.0)
    assert centerline_coverage(gt, [], tolerance=5.0) == pytest.approx(0.0)


def test_coverage_sees_a_fragmented_detection_that_matching_rejects():
    """The point of coverage: fragments cover the fibril but never match it.

    Each fragment is short relative to the fibril, so the symmetric centerline
    distance from the whole fibril to any single fragment is large even though
    together they lie along its entire length.
    """
    gt = _line(0, 400, 0, n=400)
    fragments = [_line(start, start + 24, 0) for start in range(0, 400, 25)]

    assert centerline_coverage(gt, fragments, tolerance=2.0) > 0.95
    assert match_traces(fragments, [gt], max_distance=15.0).recall == pytest.approx(0.0)


def test_coverage_is_partial_for_half_a_fibril():
    gt = _line(0, 100, 0)
    assert centerline_coverage(gt, [_line(0, 50, 0)], tolerance=2.0) == pytest.approx(0.5, abs=0.03)


def test_coverage_report_returns_one_value_per_fibril():
    gt = [_line(0, 100, 0), _line(0, 100, 200)]
    report = coverage_report(gt, [_line(0, 100, 0)], tolerance=2.0)
    assert report.shape == (2,)
    assert report[0] == pytest.approx(1.0)
    assert report[1] == pytest.approx(0.0)


def test_summarize_errors_on_empty_comparison():
    import pandas as pd

    assert summarize_errors(pd.DataFrame()).empty
