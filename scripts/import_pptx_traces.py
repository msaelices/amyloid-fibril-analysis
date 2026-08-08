"""One-off importer: PowerPoint deck with vector traces -> standard dataset.

This is deliberately a script and NOT part of the ``afa`` package. PowerPoint is
not an input format of this project; it just happens to be how the first batch
of manual traces arrived. The output is the format the pipeline already speaks:
a PNG per image plus a CSV of ``filament_id,x,y`` points per image.

The traces are stored as PowerPoint freeform shapes (``a:custGeom``) drawn on
top of the micrograph, so the underlying pixels are untouched and the geometry
is exact. Cubic Bezier segments are flattened to polylines; slide coordinates
(EMU) are mapped to image pixels through the picture's position and size.

Usage:
    python scripts/import_pptx_traces.py trazadas.pptx --out data/dataset
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/package/2006/relationships}"

BEZIER_SAMPLES = 24  # points per cubic segment; the polyline is resampled later


def _cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, BEZIER_SAMPLES)[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def _xfrm(shape: ET.Element) -> tuple[float, float, float, float]:
    xf = shape.find(f".//{A}xfrm")
    if xf is None:
        raise ValueError("shape without a:xfrm")
    off, ext = xf.find(A + "off"), xf.find(A + "ext")
    return (
        float(off.get("x")),  # type: ignore[arg-type,union-attr]
        float(off.get("y")),  # type: ignore[arg-type,union-attr]
        float(ext.get("cx")),  # type: ignore[arg-type,union-attr]
        float(ext.get("cy")),  # type: ignore[arg-type,union-attr]
    )


def shape_polyline(shape: ET.Element) -> np.ndarray:
    """Flatten one freeform shape into an ordered polyline in slide EMU units."""
    ox, oy, cx, cy = _xfrm(shape)
    path = shape.find(f".//{A}path")
    if path is None:
        raise ValueError("custGeom without a:path")
    pw, ph = float(path.get("w") or 0), float(path.get("h") or 0)
    sx = cx / pw if pw else 0.0
    sy = cy / ph if ph else 0.0

    def pt(el: ET.Element) -> np.ndarray:
        return np.array([ox + float(el.get("x")) * sx, oy + float(el.get("y")) * sy])  # type: ignore[arg-type]

    chunks: list[np.ndarray] = []
    cur: np.ndarray | None = None
    for child in path:
        tag = child.tag.split("}")[-1]
        if tag == "moveTo":
            cur = pt(child.find(A + "pt"))  # type: ignore[arg-type]
            chunks.append(cur[None, :])
        elif tag == "lnTo":
            nxt = pt(child.find(A + "pt"))  # type: ignore[arg-type]
            chunks.append(nxt[None, :])
            cur = nxt
        elif tag == "cubicBezTo":
            c1, c2, end = (pt(p) for p in child.findall(A + "pt"))
            chunks.append(_cubic(cur, c1, c2, end)[1:])  # type: ignore[arg-type]
            cur = end
        elif tag == "close":
            raise ValueError("closed path: not an open fibril trace")
    return np.vstack(chunks)


def _slide_image(z: zipfile.ZipFile, n: int) -> tuple[bytes, str]:
    rels = ET.fromstring(z.read(f"ppt/slides/_rels/slide{n}.xml.rels"))
    targets = [
        r.get("Target", "").replace("../", "ppt/")
        for r in rels.iter(R + "Relationship")
        if "image" in (r.get("Type") or "")
    ]
    if len(targets) != 1:
        raise ValueError(f"slide {n}: expected exactly 1 image, found {len(targets)}")
    return z.read(targets[0]), targets[0]


def parse_slide(z: zipfile.ZipFile, n: int) -> dict:
    """Extract the micrograph plus its traces and number labels, in pixel coords."""
    from PIL import Image

    blob, target = _slide_image(z, n)
    width, height = Image.open(io.BytesIO(blob)).size

    root = ET.fromstring(z.read(f"ppt/slides/slide{n}.xml"))
    pics = list(root.iter(P + "pic"))
    if len(pics) != 1:
        raise ValueError(f"slide {n}: expected exactly 1 picture, found {len(pics)}")
    pox, poy, pcx, pcy = _xfrm(pics[0])

    def to_pixels(pts: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [(pts[:, 0] - pox) / pcx * width, (pts[:, 1] - poy) / pcy * height]
        )

    traces, labels = [], []
    for sp in root.iter(P + "sp"):
        text = "".join(t.text or "" for t in sp.iter(A + "t")).strip()
        if sp.find(f".//{A}custGeom") is not None:
            traces.append(to_pixels(shape_polyline(sp)))
        elif text:
            ox, oy, _, _ = _xfrm(sp)
            labels.append((text, to_pixels(np.array([[ox, oy]]))[0]))

    return {
        "slide": n,
        "image_bytes": blob,
        "image_name": Path(target).name,
        "size": (width, height),
        "traces": traces,
        "labels": labels,
    }


def assign_labels(traces: list[np.ndarray], labels: list[tuple[str, np.ndarray]]) -> list[dict]:
    """Attach each PowerPoint number to its nearest trace, flagging ambiguity.

    The numbers are text boxes placed by hand next to a fibril, so the mapping is
    a guess based on proximity. Any trace claimed by two numbers, or left without
    one, is reported rather than silently resolved.
    """
    rows = []
    for text, pos in labels:
        dists = [float(np.linalg.norm(tr - pos, axis=1).min()) for tr in traces]
        best = int(np.argmin(dists)) if dists else -1
        rows.append({"label": text, "trace_index": best, "distance_px": dists[best]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pptx", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/dataset"))
    args = ap.parse_args()

    images_dir = args.out / "images"
    traces_dir = args.out / "traces"
    images_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.pptx)
    n_slides = sum(
        1
        for x in z.namelist()
        if x.startswith("ppt/slides/slide") and x.endswith(".xml")
    )

    label_rows: list[dict] = []
    problems: list[str] = []
    n_traces = 0

    for n in range(1, n_slides + 1):
        info = parse_slide(z, n)
        stem = f"slide{n:02d}"
        (images_dir / f"{stem}.png").write_bytes(info["image_bytes"])

        traces = info["traces"]
        if not traces:
            problems.append(f"{stem}: no traces")
            continue

        with open(traces_dir / f"{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filament_id", "x", "y"])
            for k, pts in enumerate(traces):
                fid = f"{stem}_{k:02d}"
                for x, y in pts:
                    w.writerow([fid, f"{x:.3f}", f"{y:.3f}"])
        n_traces += len(traces)

        assigned = assign_labels(traces, info["labels"])
        claimed = [a["trace_index"] for a in assigned]
        for a in assigned:
            label_rows.append({"image_id": stem, **a})
        duplicated = {i for i in claimed if claimed.count(i) > 1}
        unlabelled = set(range(len(traces))) - set(claimed)
        if duplicated:
            problems.append(
                f"{stem}: traces {sorted(duplicated)} claimed by more than one number"
            )
        if unlabelled:
            problems.append(f"{stem}: traces {sorted(unlabelled)} have no number nearby")

    with open(args.out / "pptx_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "label", "trace_index", "distance_px"])
        w.writeheader()
        w.writerows(label_rows)

    print(f"{n_slides} slides -> {n_traces} traces written to {args.out}/")
    print(f"PowerPoint numbers: {len(label_rows)} (see {args.out}/pptx_labels.csv)")
    if problems:
        print(f"\n{len(problems)} numbering issue(s) to review by hand:")
        for p in problems:
            print(f"  {p}")


if __name__ == "__main__":
    main()
