"""End-to-end orchestration: micrograph -> traces -> metrics rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from afa.config import Config
from afa.io.annotations import Trace, load_traces
from afa.io.mrc import MrcImage, load_mrc
from afa.morphology.metrics import compute_metrics
from afa.segment.classical import vesselness_probability
from afa.trace.resample import resample_polyline, smooth_polyline
from afa.trace.tracer import trace_centerlines


def _rows_from_centerlines(
    centerlines: list[np.ndarray],
    *,
    image_id: str,
    patient_id: str,
    pixel_size_nm: float,
    ids: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, cl in enumerate(centerlines):
        fid = ids[i] if ids is not None and i < len(ids) else f"{i:04d}"
        m = compute_metrics(cl, pixel_size=pixel_size_nm)
        row = {"image_id": image_id, "patient_id": patient_id, "filament_id": fid}
        row.update(m.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_traces(
    mrc_path: str | Path,
    traces_path: str | Path,
    *,
    patient_id: str,
    image_id: str | None = None,
    config: Config | None = None,
) -> tuple[pd.DataFrame, MrcImage, list[np.ndarray]]:
    """Compute metrics directly from existing manual traces (no detection).

    Returns the per-fibril DataFrame, the loaded image, and the smoothed
    centerlines (for overlay rendering).
    """
    config = config or Config()
    img = load_mrc(mrc_path, pixel_size_a=config.pixel_size_a)
    traces: list[Trace] = load_traces(traces_path)
    image_id = image_id or Path(mrc_path).stem

    centerlines, ids = [], []
    for tr in traces:
        cl = smooth_polyline(
            resample_polyline(tr.points, step=config.trace.resample_step),
            window=config.trace.smooth_window,
        )
        centerlines.append(cl)
        ids.append(tr.filament_id)

    df = _rows_from_centerlines(
        centerlines,
        image_id=image_id,
        patient_id=patient_id,
        pixel_size_nm=img.pixel_size_nm,
        ids=ids,
    )
    return df, img, centerlines


def trace_and_measure(
    mrc_path: str | Path,
    *,
    patient_id: str,
    image_id: str | None = None,
    config: Config | None = None,
) -> tuple[pd.DataFrame, MrcImage, list[np.ndarray]]:
    """Detect + trace fibrils automatically, then compute metrics."""
    config = config or Config()
    img = load_mrc(mrc_path, pixel_size_a=config.pixel_size_a)
    image_id = image_id or Path(mrc_path).stem

    if config.detect.method == "unet":
        from afa.segment.unet import UNetSegmenter

        seg = UNetSegmenter(weights=config.detect.unet_weights).load()
        prob = seg.predict(img.data)
    else:
        prob = vesselness_probability(
            img.data, invert=config.detect.invert, sigmas=tuple(config.detect.sigmas)
        )

    mask = prob > config.detect.prob_threshold
    centerlines = trace_centerlines(
        mask,
        resample_step=config.trace.resample_step,
        smooth_window=config.trace.smooth_window,
        min_branch_px=config.trace.min_branch_px,
        link_crossings=config.trace.link_crossings,
        max_link_angle_deg=config.trace.max_link_angle_deg,
        merge_junction_px=config.trace.merge_junction_px,
        bridge_gap_px=config.trace.bridge_gap_px,
        bridge_angle_deg=config.trace.bridge_angle_deg,
    )
    df = _rows_from_centerlines(
        centerlines,
        image_id=image_id,
        patient_id=patient_id,
        pixel_size_nm=img.pixel_size_nm,
    )
    return df, img, centerlines
