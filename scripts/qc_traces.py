"""Render a contact sheet of snapped traces for a human to check.

The manual traces were drawn beside each fibril, so `afa.trace.snap` moves them
onto the ridge. Those snapped centerlines are simultaneously the training labels
and the validation ground truth, so one bad trace corrupts both at once, and
nothing automatic has ever looked at them one by one.

Each panel shows one trace: the drawn curve in red, the snapped result in green,
over a crop of its own micrograph. Panels are ordered most-suspicious first,
using the spread of the chosen offsets -- a trace whose offset wanders changed
fibril part way along, which `gain` cannot see because landing on *a* ridge
scores just as well as landing on the right one.

Reject list: write one trace id per line to `data/dataset/rejected_traces.txt`.

Usage:
    python scripts/qc_traces.py --out outputs/qc --images slide01 slide02
    python scripts/qc_traces.py --out outputs/qc          # all of them
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from afa.io.annotations import load_traces  # noqa: E402
from afa.segment.classical import vesselness_probability  # noqa: E402
from afa.segment.dataset import load_grayscale  # noqa: E402
from afa.trace.snap import snap_to_ridge  # noqa: E402

PER_PAGE = 6


def _crop(shape: tuple[int, int], pts: np.ndarray, margin: int = 60) -> tuple[slice, slice]:
    """Bounding box of a trace, padded, clipped to the image."""
    x0, y0 = pts.min(axis=0) - margin
    x1, y1 = pts.max(axis=0) + margin
    return (
        slice(max(int(y0), 0), min(int(y1), shape[0])),
        slice(max(int(x0), 0), min(int(x1), shape[1])),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=Path("outputs/qc"))
    ap.add_argument("--images", nargs="*", default=None, help="image ids; default all")
    ap.add_argument("--max-shift", type=float, default=25.0)
    args = ap.parse_args()

    ids = args.images or sorted(p.stem for p in (args.data / "images").glob("*.png"))
    args.out.mkdir(parents=True, exist_ok=True)

    panels = []
    for image_id in ids:
        trace_path = args.data / "traces" / f"{image_id}.csv"
        if not trace_path.exists():
            continue
        image = load_grayscale(args.data / "images" / f"{image_id}.png")
        ridge = vesselness_probability(image, invert=True, sigmas=(2.0, 3.0, 4.0, 5.0))
        for trace in load_traces(trace_path):
            result = snap_to_ridge(trace.points, ridge, max_shift=args.max_shift)
            panels.append(
                {
                    "id": trace.filament_id,
                    "image": image,
                    "drawn": np.asarray(trace.points, dtype=float),
                    "snapped": result.points,
                    "spread": result.offset_spread,
                    "gain": result.gain,
                    "offset": float(np.median(result.offsets)),
                }
            )
        print(f"  {image_id}: {sum(1 for p in panels if p['id'].startswith(image_id))} traces")

    # Most suspicious first: a wide offset spread means the trace changed its
    # mind about which fibril it was on.
    panels.sort(key=lambda p: -p["spread"])

    with open(args.out / "traces.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "filament_id", "offset_spread_px", "median_offset_px", "ridge_gain"])
        for rank, p in enumerate(panels, start=1):
            w.writerow([
                rank, p["id"], f"{p['spread']:.1f}",
                f"{p['offset']:+.1f}", f"{p['gain']:.1f}",
            ])

    pages = 0
    for start in range(0, len(panels), PER_PAGE):
        chunk = panels[start:start + PER_PAGE]
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for ax, p in zip(axes.ravel(), chunk, strict=False):
            ys, xs = _crop(p["image"].shape, np.vstack([p["drawn"], p["snapped"]]))
            ax.imshow(p["image"][ys, xs], cmap="gray", interpolation="nearest")
            ax.plot(p["drawn"][:, 0] - xs.start, p["drawn"][:, 1] - ys.start,
                    "-", lw=1.4, color="red", alpha=0.85, label="drawn")
            ax.plot(p["snapped"][:, 0] - xs.start, p["snapped"][:, 1] - ys.start,
                    "-", lw=1.6, color="lime", label="snapped")
            flag = "  <-- CHECK" if p["spread"] > 8 else ""
            ax.set_title(f"{p['id']}   spread {p['spread']:.0f} px   "
                         f"offset {p['offset']:+.0f}   gain {p['gain']:.1f}x{flag}",
                         fontsize=10)
            ax.set_axis_off()
        for ax in axes.ravel()[len(chunk):]:
            ax.set_axis_off()
        axes.ravel()[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"snapped traces {start + 1}-{start + len(chunk)} of {len(panels)}, "
            f"most suspicious first   (red = as drawn, green = snapped onto the ridge)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        page = args.out / f"qc_{start // PER_PAGE + 1:02d}.png"
        fig.savefig(page, dpi=95)
        plt.close(fig)
        pages += 1

    flagged = sum(1 for p in panels if p["spread"] > 8)
    print(f"\n{len(panels)} traces over {pages} page(s) -> {args.out}/")
    print(f"{flagged} flagged (offset spread > 8 px); they sort first")
    print(f"reject by writing ids to {args.data / 'rejected_traces.txt'}")


if __name__ == "__main__":
    main()
