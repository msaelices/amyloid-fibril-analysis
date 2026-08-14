"""Load manual fibril traces from several possible sources.

Supported inputs
----------------
* ImageJ/FIJI ROIs: a ``.roi`` file or a ``RoiSet.zip`` (polyline / freehand /
  segmented-line ROIs). Requires the ``roifile`` package.
* CSV of points: columns ``filament_id, x, y`` (one row per vertex, ordered).
* Burned-in traces: a bright overlay drawn on the image (e.g. white dashed
  lines). Extracted by intensity threshold -- lower fidelity, use only if no
  vector annotation exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trace:
    """A single fibril trace as an ordered polyline in pixel coordinates.

    ``points`` is an ``(N, 2)`` array of ``(x, y)`` vertices.
    """

    filament_id: str
    points: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)
        if len(self.points) < 2:
            raise ValueError(f"Trace {self.filament_id} needs >=2 points")


def load_traces(path: str | Path, *, source: str = "auto") -> list[Trace]:
    """Load traces from a file, auto-detecting the format by extension."""
    path = Path(path)
    if source == "auto":
        suffix = path.suffix.lower()
        if suffix == ".zip" or suffix == ".roi":
            source = "imagej"
        elif suffix == ".csv":
            source = "csv"
        else:
            raise ValueError(
                f"Cannot auto-detect annotation format for {path!r}; pass source=..."
            )
    if source == "imagej":
        return _load_imagej(path)
    if source == "csv":
        return _load_csv(path)
    raise ValueError(f"Unknown annotation source {source!r}")


def _load_imagej(path: Path) -> list[Trace]:
    import roifile  # noqa: PLC0415

    rois = roifile.ImagejRoi.fromfile(str(path))
    if isinstance(rois, roifile.ImagejRoi):
        rois = [rois]
    traces: list[Trace] = []
    for i, roi in enumerate(rois):
        coords = np.asarray(roi.coordinates(), dtype=float)  # (N, 2) as (x, y)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue
        name = getattr(roi, "name", None) or f"f{i:03d}"
        traces.append(Trace(filament_id=str(name), points=coords))
    if not traces:
        raise ValueError(f"No polyline ROIs found in {path}")
    return traces


def _load_csv(path: Path) -> list[Trace]:
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = {"filament_id", "x", "y"}
    if not required.issubset(cols):
        raise ValueError(f"CSV must have columns {required}; got {list(df.columns)}")
    traces: list[Trace] = []
    for fid, grp in df.groupby(cols["filament_id"], sort=False):
        pts = grp[[cols["x"], cols["y"]]].to_numpy(dtype=float)
        if len(pts) >= 2:
            traces.append(Trace(filament_id=str(fid), points=pts))
    return traces


def extract_burned_in(
    image: np.ndarray,
    *,
    threshold: float | None = None,
    min_length: int = 20,
) -> np.ndarray:
    """Extract a bright burned-in overlay (e.g. white traces) as a binary mask.

    This is a fallback for when only rasterized annotations exist. It returns a
    boolean mask of the overlay; turning that mask into ordered polylines is the
    job of :mod:`afa.trace.tracer`. Fidelity is limited -- prefer vector ROIs.
    """
    from skimage.filters import threshold_otsu  # noqa: PLC0415
    from skimage.morphology import remove_small_objects  # noqa: PLC0415

    img = image.astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    thr = threshold if threshold is not None else float(threshold_otsu(img))
    mask = img > thr
    mask = remove_small_objects(mask, min_size=min_length)
    return mask
