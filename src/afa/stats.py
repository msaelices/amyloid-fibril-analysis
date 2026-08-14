"""Per-patient aggregation: mean, SD and 95% confidence interval per metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "length",
    "tortuosity",
    "max_curvature",
    "mean_abs_curvature",
    "total_abs_turning",
    "total_abs_turning_per_length",
    "local_dir_change_per_length",
]


def _t_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """t-based (1-alpha) confidence interval for the mean."""
    from scipy import stats  # noqa: PLC0415

    v = values[np.isfinite(values)]
    n = v.size
    if n < 2:
        m = float(v.mean()) if n == 1 else float("nan")
        return (m, m)
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return (mean - tcrit * se, mean + tcrit * se)


def _bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling at the image level when ``groups`` given.

    Resampling whole images (not individual fibrils) respects the fact that
    fibrils within an image are not independent.
    """
    rng = np.random.default_rng(seed)
    v = values[np.isfinite(values)]
    if v.size < 2:
        m = float(v.mean()) if v.size == 1 else float("nan")
        return (m, m)

    means = np.empty(n_boot)
    if groups is None:
        for b in range(n_boot):
            means[b] = rng.choice(v, size=v.size, replace=True).mean()
    else:
        g = groups[np.isfinite(values)]
        uniq = np.unique(g)
        for b in range(n_boot):
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            sample = np.concatenate([v[g == c] for c in chosen])
            means[b] = sample.mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def summarize_per_patient(
    per_image: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    bootstrap: bool = False,
    group_col: str = "image_id",
) -> pd.DataFrame:
    """Aggregate per-fibril metrics into per-patient statistics.

    Parameters
    ----------
    per_image:
        DataFrame with at least a ``patient_id`` column plus the metric columns.
    metrics:
        Which metric columns to summarize (defaults to :data:`METRIC_COLUMNS`).
    bootstrap:
        If ``True``, use an image-level percentile bootstrap for the CI;
        otherwise a t-based CI.
    group_col:
        Column identifying the image, used for the bootstrap grouping.

    Returns
    -------
    One row per (patient, metric) with ``n, mean, sd, ci95_low, ci95_high``.
    """
    metrics = metrics or [m for m in METRIC_COLUMNS if m in per_image.columns]
    rows = []
    for patient, grp in per_image.groupby("patient_id", sort=True):
        for metric in metrics:
            vals = grp[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if bootstrap and group_col in grp.columns:
                lo, hi = _bootstrap_ci(vals, grp[group_col].to_numpy())
            else:
                lo, hi = _t_ci(vals)
            rows.append(
                {
                    "patient_id": patient,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": float(finite.mean()) if finite.size else float("nan"),
                    "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)
