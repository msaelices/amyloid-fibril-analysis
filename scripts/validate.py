"""Validate automatic tracing against the manual traces on held-out images.

Reports, per held-out micrograph and in aggregate:

* pixel-level Dice and IoU of the predicted fibril mask vs the manual one;
* fibril-level recall, fragmentation, and a lower bound on precision;
* per-metric error of the morphology descriptors on matched fibrils.

Usage:
    python scripts/validate.py --data data/dataset --weights models/unet.pt \
        --out outputs/validation
    python scripts/validate.py --data data/dataset --detector classical \
        --out outputs/validation_classical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from afa.segment.classical import vesselness_probability
from afa.segment.dataset import load_labelled_images, split_images
from afa.segment.unet import UNetSegmenter
from afa.trace.tracer import trace_centerlines
from afa.validate import (
    compare_metrics,
    coverage_report,
    dice,
    iou,
    match_traces,
    summarize_errors,
)
from afa.viz import save_overlay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--weights", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--detector", choices=["unet", "classical"], default="unet")
    ap.add_argument("--out", type=Path, default=Path("outputs/validation"))
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--split", choices=["test", "val", "train"], default="test",
                    help="which held-out split to report on; val was selected against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("data/dataset/cache"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--width-px", type=int, default=7)
    ap.add_argument("--match-distance", type=float, default=15.0)
    ap.add_argument("--bridge-gap-px", type=float, default=60.0,
                    help="join fragments across gaps this long, checked against the map")
    ap.add_argument("--coverage-tolerance", type=float, default=6.0,
                    help="px from a manual centerline still counted as covered")
    ap.add_argument("--pixel-size-nm", type=float, default=1.0,
                    help="nm per pixel; 1.0 leaves lengths in pixels")
    ap.add_argument("--overlays", action="store_true", help="write per-image overlays")
    args = ap.parse_args()

    image_ids = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    train_ids, val_ids, test_ids = split_images(
        image_ids, n_val=args.n_val, n_test=args.n_test, seed=args.seed
    )
    chosen = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    if not chosen:
        raise SystemExit(f"the {args.split} split is empty; raise --n-{args.split}")
    print(f"reporting on the {args.split} split, {len(chosen)} images: {', '.join(chosen)}")
    if args.split == "val":
        print("WARNING: the checkpoint was selected on these images; scores are optimistic")
    print()

    items = load_labelled_images(
        args.data / "images",
        args.data / "traces",
        image_ids=chosen,
        cache_dir=args.cache,
        width_px=args.width_px,
    )

    predictor = None
    if args.detector == "unet":
        predictor = UNetSegmenter(weights=args.weights).load()

    args.out.mkdir(parents=True, exist_ok=True)
    per_image, comparisons = [], []

    for item in items:
        if predictor is not None:
            prob = predictor.predict(item.image)
        else:
            prob = vesselness_probability(
                item.image, invert=True, sigmas=(2.0, 3.0, 4.0, 5.0)
            )
            prob = prob / (prob.max() + 1e-9)

        pred_mask = prob > args.threshold
        # Score only where the manual label is defined: unannotated fibrils are
        # unknown, not background, so counting them as errors is meaningless.
        valid = ~item.ignore
        predicted = trace_centerlines(
            pred_mask,
            min_branch_px=20,
            bridge_gap_px=args.bridge_gap_px,
            evidence=prob,
        )
        match = match_traces(predicted, item.centerlines, max_distance=args.match_distance)

        cover = coverage_report(item.centerlines, predicted, tolerance=args.coverage_tolerance)
        row = {
            "image_id": item.image_id,
            # Two Dice values, always reported together. The labelled one skips
            # pixels whose truth is unknown, which is the defensible thing to do
            # under partial annotation -- but the ignore mask is by construction
            # the ridge-like pixels, i.e. exactly where a fibril detector's
            # excess predictions land, so it flatters any over-predicting model.
            # The unrestricted one is pessimistic for the opposite reason: it
            # counts untraced real fibrils as errors. The truth is between them.
            "dice_labelled": dice(pred_mask & valid, item.mask & valid),
            "dice_all_pixels": dice(pred_mask, item.mask),
            "ignored_fraction": float(item.ignore.mean()),
            "predicted_fraction": float(pred_mask.mean()),
            "mask_fraction": float(item.mask.mean()),
            "iou": iou(pred_mask & valid, item.mask & valid),
            "n_manual": match.n_gt,
            "n_detected": match.n_pred,
            "matched": len(match.pairs),
            "recall": match.recall,
            "mean_coverage": float(cover.mean()) if cover.size else float("nan"),
            "fibrils_80pct_covered": int((cover >= 0.8).sum()),
            "precision_lower_bound": match.precision_lower_bound,
            "detections_per_matched_fibril": match.detections_per_matched_fibril,
        }
        per_image.append(row)
        print(
            f"{item.image_id}: dice {row['dice_labelled']:.3f} labelled / "
            f"{row['dice_all_pixels']:.3f} all px "
            f"(ignored {100 * row['ignored_fraction']:.0f}%)  "
            f"coverage={row['mean_coverage']:.2f} "
            f"recall={row['recall']:.2f} ({len(match.pairs)}/{match.n_gt}) "
            f"detected={match.n_pred}"
        )

        comp = compare_metrics(
            predicted, item.centerlines, match, pixel_size=args.pixel_size_nm
        )
        comp.insert(0, "image_id", item.image_id)
        comparisons.append(comp)

        if args.overlays:
            save_overlay(
                item.image, predicted, args.out / f"overlay_{item.image_id}.png"
            )

    per_image_df = pd.DataFrame(per_image)
    per_image_df.to_csv(args.out / "per_image_validation.csv", index=False)

    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    comparison.to_csv(args.out / "matched_fibrils.csv", index=False)
    errors = summarize_errors(comparison)
    errors.to_csv(args.out / "metric_errors.csv", index=False)

    def _mean(col: str) -> float:
        return float(per_image_df[col].mean()) if len(per_image_df) else float("nan")

    summary = {
        "detector": args.detector,
        "split": args.split,
        "n_images": len(per_image_df),
        "mean_dice_labelled": _mean("dice_labelled"),
        "mean_dice_all_pixels": _mean("dice_all_pixels"),
        "mean_ignored_fraction": _mean("ignored_fraction"),
        "mean_iou": _mean("iou"),
        "total_manual": int(per_image_df["n_manual"].sum()) if len(per_image_df) else 0,
        "total_matched": int(per_image_df["matched"].sum()) if len(per_image_df) else 0,
        "overall_recall": (
            float(per_image_df["matched"].sum() / per_image_df["n_manual"].sum())
            if len(per_image_df) and per_image_df["n_manual"].sum()
            else float("nan")
        ),
        "mean_coverage": _mean("mean_coverage"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}\ndetector: {args.detector}  |  split: {args.split}")
    print(
        f"mean Dice {summary['mean_dice_labelled']:.3f} on labelled pixels, "
        f"{summary['mean_dice_all_pixels']:.3f} on all pixels "
        f"({100 * summary['mean_ignored_fraction']:.0f}% of each image is unlabelled "
        f"and excluded from the first)"
    )
    print(f"mean coverage of manual fibrils {summary['mean_coverage']:.2f}")
    print(
        f"one-to-one fibril recall {summary['overall_recall']:.2f} "
        f"({summary['total_matched']}/{summary['total_manual']})"
    )
    if summary["mean_coverage"] > 0.5 and summary["overall_recall"] < 0.2:
        print("  -> fibrils ARE found but fragmented; the tracer, not the detector, is the limit")
    if not errors.empty:
        print("\nmorphology error on matched fibrils:")
        print(errors.to_string(index=False))
    print(f"\nwritten to {args.out}/")
    if np.isclose(args.pixel_size_nm, 1.0):
        print("NOTE: lengths/curvature are in PIXELS (--pixel-size-nm not set)")


if __name__ == "__main__":
    main()
