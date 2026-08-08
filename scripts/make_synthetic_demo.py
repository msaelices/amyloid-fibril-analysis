"""Generate a synthetic micrograph with known fibrils and run the pipeline.

Useful as an end-to-end smoke test and a demo of the expected inputs/outputs
before real .mrc data is wired in.

Usage:
    python scripts/make_synthetic_demo.py --out outputs_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def synthetic_fibrils(shape=(512, 512), n=6, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    traces = []
    for _ in range(n):
        # A gently curved fibril as a quadratic Bezier-ish arc.
        p0 = rng.uniform([40, 40], [shape[1] - 40, shape[0] - 40])
        p2 = rng.uniform([40, 40], [shape[1] - 40, shape[0] - 40])
        ctrl = (p0 + p2) / 2 + rng.uniform(-80, 80, size=2)
        t = np.linspace(0, 1, 200)[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p2
        traces.append(pts)
        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < shape[0] and 0 <= xi < shape[1]:
                img[yi, xi] = 1.0
    from skimage.morphology import binary_dilation, disk

    img = binary_dilation(img > 0, disk(2)).astype(np.float32)
    img = 1.0 - img  # fibrils darker than background
    img += rng.normal(0, 0.25, size=shape).astype(np.float32)  # heavy noise, low SNR
    return img, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs_demo"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import mrcfile

    from afa.pipeline import trace_and_measure
    from afa.stats import summarize_per_patient
    from afa.viz import save_overlay

    img, traces = synthetic_fibrils()
    mrc_path = args.out / "demo.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as m:
        m.set_data(img.astype(np.float32))
        m.voxel_size = 5.0  # 5 A/pixel

    df, mrc_img, centerlines = trace_and_measure(mrc_path, patient_id="DEMO")
    df.to_csv(args.out / "per_image.csv", index=False)
    save_overlay(mrc_img.data, centerlines, args.out / "overlay_demo.png")
    summary = summarize_per_patient(df)
    summary.to_csv(args.out / "per_patient.csv", index=False)

    print(f"Ground-truth fibrils: {len(traces)}; detected: {len(centerlines)}")
    print(df.to_string(index=False))
    print(f"\nOutputs written to {args.out}/")


if __name__ == "__main__":
    main()
