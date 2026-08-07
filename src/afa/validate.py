"""Compare automatic traces against the manual ground truth.

Two levels, because they answer different questions:

* **Pixel level** (Dice / IoU) -- does the detector light up the right pixels?
* **Fibril level** -- does each manual fibril end up as one traced object, and do
  its morphology numbers come out right? This is what the study actually needs;
  a detector can score well on Dice while fragmenting every fibril into pieces,
  which destroys length and tortuosity.

A caveat that must travel with every number produced here: **the annotation is
partial**. Only some fibrils in each micrograph were traced, so a detection with
no manual counterpart is usually a real fibril nobody drew, not a false alarm.
Recall is therefore meaningful and precision is a lower bound; the code reports
both but labels precision for what it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afa.morphology.metrics import compute_metrics
from afa.stats import METRIC_COLUMNS


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient of two boolean masks (1.0 = identical)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def centerline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-point distance between two polylines, in pixels.

    Symmetric on purpose: the one-directional mean is small whenever ``a`` is a
    short fragment sitting on top of a long ``b``, which is exactly the failure
    mode (fragmentation) this is meant to expose.
    """
    from scipy.spatial import cKDTree

    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    a_to_b = cKDTree(b).query(a)[0].mean()
    b_to_a = cKDTree(a).query(b)[0].mean()
    return float(0.5 * (a_to_b + b_to_a))


def centerline_coverage(
    ground_truth: np.ndarray,
    predicted: list[np.ndarray],
    *,
    tolerance: float = 6.0,
    step: float = 1.0,
) -> float:
    """Fraction of one manual centerline lying within ``tolerance`` of any detection.

    This is the fragmentation-tolerant companion to :func:`match_traces`. A
    detector that shatters every fibril into short pieces scores zero one-to-one
    matches while still covering the fibril completely, and the two numbers
    together say which failure is happening: low coverage means the fibril was
    missed, high coverage with zero matches means it was found but fragmented.
    """
    from scipy.spatial import cKDTree

    from afa.trace.resample import resample_polyline

    gt = resample_polyline(np.asarray(ground_truth, dtype=float), step=step)
    if not predicted:
        return 0.0
    points = np.vstack([np.asarray(p, dtype=float).reshape(-1, 2) for p in predicted])
    distances = cKDTree(points).query(gt)[0]
    return float(np.mean(distances <= tolerance))


def coverage_report(
    ground_truth: list[np.ndarray],
    predicted: list[np.ndarray],
    *,
    tolerance: float = 6.0,
) -> np.ndarray:
    """Per-manual-fibril coverage, as returned by :func:`centerline_coverage`."""
    return np.array(
        [centerline_coverage(g, predicted, tolerance=tolerance) for g in ground_truth]
    )


@dataclass
class MatchResult:
    """Outcome of matching predicted centerlines to manual ones."""

    pairs: list[tuple[int, int, float]]      # (pred_index, gt_index, distance_px)
    unmatched_pred: list[int]
    unmatched_gt: list[int]
    n_pred: int
    n_gt: int

    @property
    def recall(self) -> float:
        """Fraction of manual fibrils that were found."""
        return len(self.pairs) / self.n_gt if self.n_gt else float("nan")

    @property
    def precision_lower_bound(self) -> float:
        """Fraction of detections matching a manual trace.

        A lower bound, not precision: the annotation is partial, so unmatched
        detections include real fibrils that were never traced.
        """
        return len(self.pairs) / self.n_pred if self.n_pred else float("nan")

    @property
    def fragmentation(self) -> float:
        """Detections per matched manual fibril; 1.0 means one trace per fibril."""
        return self.n_pred / len(self.pairs) if self.pairs else float("nan")


def match_traces(
    predicted: list[np.ndarray],
    ground_truth: list[np.ndarray],
    *,
    max_distance: float = 15.0,
) -> MatchResult:
    """One-to-one match of predicted to manual centerlines by proximity.

    Uses the Hungarian algorithm on the pairwise symmetric centerline distance,
    then drops assignments further apart than ``max_distance``.

    Parameters
    ----------
    predicted, ground_truth:
        Lists of ``(N, 2)`` ``(x, y)`` polylines in pixel coordinates.
    max_distance:
        Largest mean centerline separation still considered the same fibril.
    """
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = len(predicted), len(ground_truth)
    if n_pred == 0 or n_gt == 0:
        return MatchResult([], list(range(n_pred)), list(range(n_gt)), n_pred, n_gt)

    cost = np.empty((n_pred, n_gt), dtype=float)
    for i, p in enumerate(predicted):
        for j, g in enumerate(ground_truth):
            cost[i, j] = centerline_distance(p, g)

    rows, cols = linear_sum_assignment(cost)
    pairs = [
        (int(i), int(j), float(cost[i, j]))
        for i, j in zip(rows, cols, strict=True)
        if cost[i, j] <= max_distance
    ]
    matched_pred = {i for i, _, _ in pairs}
    matched_gt = {j for _, j, _ in pairs}
    return MatchResult(
        pairs=pairs,
        unmatched_pred=[i for i in range(n_pred) if i not in matched_pred],
        unmatched_gt=[j for j in range(n_gt) if j not in matched_gt],
        n_pred=n_pred,
        n_gt=n_gt,
        )


def compare_metrics(
    predicted: list[np.ndarray],
    ground_truth: list[np.ndarray],
    match: MatchResult,
    *,
    pixel_size: float = 1.0,
) -> pd.DataFrame:
    """Per-matched-fibril morphology of prediction vs manual trace.

    Returns one row per matched pair with ``<metric>_pred``, ``<metric>_gt`` and
    ``<metric>_abs_error`` columns, plus the centerline distance.
    """
    rows = []
    for pred_idx, gt_idx, distance in match.pairs:
        mp = compute_metrics(predicted[pred_idx], pixel_size=pixel_size).as_dict()
        mg = compute_metrics(ground_truth[gt_idx], pixel_size=pixel_size).as_dict()
        row: dict[str, float] = {
            "pred_index": pred_idx,
            "gt_index": gt_idx,
            "centerline_distance_px": distance,
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_pred"] = mp[metric]
            row[f"{metric}_gt"] = mg[metric]
            row[f"{metric}_abs_error"] = abs(mp[metric] - mg[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_errors(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`compare_metrics` output into one row per metric."""
    rows = []
    for metric in METRIC_COLUMNS:
        pred_col, gt_col, err_col = f"{metric}_pred", f"{metric}_gt", f"{metric}_abs_error"
        if err_col not in comparison.columns or comparison.empty:
            continue
        gt = comparison[gt_col].to_numpy(dtype=float)
        err = comparison[err_col].to_numpy(dtype=float)
        finite = np.isfinite(err) & np.isfinite(gt)
        rows.append(
            {
                "metric": metric,
                "n": int(finite.sum()),
                "gt_mean": float(gt[finite].mean()) if finite.any() else float("nan"),
                "mean_abs_error": float(err[finite].mean()) if finite.any() else float("nan"),
                "median_abs_error": float(np.median(err[finite])) if finite.any() else float("nan"),
                "mean_rel_error": (
                    float(np.mean(err[finite] / np.abs(gt[finite])))
                    if finite.any() and np.all(gt[finite] != 0)
                    else float("nan")
                ),
                "bias": (
                    float(np.mean(comparison[pred_col].to_numpy(dtype=float)[finite] - gt[finite]))
                    if finite.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
