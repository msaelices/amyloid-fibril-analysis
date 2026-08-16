"""Train and evaluate over k folds, so every image is scored by a model that never saw it.

Why this exists: a single held-out split leaves 20 fibrils, and at that size a
change from 3/20 to 7/20 does not reach significance (McNemar p = 0.22). Real
improvements and noise are indistinguishable, which makes every other experiment
unmeasurable. Rotating the test set and pooling the folds evaluates all 41 images
and 111 fibrils instead, without any new annotation.

Costs one training run per fold. Results, including the per-fold detail and the
pooled figures, land in ``reports/``.

Usage:
    python scripts/cross_validate.py --folds 5 --epochs 50
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from afa.segment.dataset import PatchDataset, kfold_splits, load_labelled_images
from afa.segment.unet import UNetSegmenter
from afa.trace.tracer import trace_centerlines
from afa.validate import compare_metrics, coverage_report, dice, match_traces, summarize_errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--models", type=Path, default=Path("models/cv"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--cache", type=Path, default=Path("data/dataset/cache"))
    ap.add_argument("--name", default="cv5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--width-px", type=int, default=7)
    ap.add_argument("--bridge-gap-px", type=float, default=60.0)
    ap.add_argument("--pixel-size-nm", type=float, default=0.3299,
                    help="calibrated from the matched .mrc files; see issue #4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Deferred: torch is an optional dependency and the rest of this script's
    # imports must work without it.
    from afa.segment.torch_unet import train  # noqa: PLC0415

    image_ids = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    folds = kfold_splits(image_ids, n_folds=args.folds, n_val=args.n_val, seed=args.seed)
    print(f"{len(image_ids)} images, {args.folds} folds\n")

    items = {
        it.image_id: it
        for it in load_labelled_images(
            args.data / "images", args.data / "traces",
            cache_dir=args.cache, width_px=args.width_px,
        )
    }
    args.models.mkdir(parents=True, exist_ok=True)

    per_image_rows, comparisons, fold_rows = [], [], []

    for k, (train_ids, val_ids, test_ids) in enumerate(folds):
        print(f"--- fold {k + 1}/{len(folds)}: "
              f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test ---")
        weights = args.models / f"{args.name}_fold{k}.pt"

        history = train(
            PatchDataset([items[i] for i in train_ids], patch=args.patch,
                         overlap=args.overlap, augment_data=True, seed=args.seed),
            PatchDataset([items[i] for i in val_ids], patch=args.patch,
                         overlap=args.overlap, augment_data=False, seed=args.seed),
            out_path=weights, epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, seed=args.seed, log=False,
        )
        print(f"    best val {history.best_val:.4f} at epoch {history.best_epoch + 1}")

        seg = UNetSegmenter(weights=weights).load()
        matched = total = 0
        for image_id in test_ids:
            item = items[image_id]
            prob = seg.predict(item.image)
            pred = prob > args.threshold
            valid = ~item.ignore
            cls = trace_centerlines(pred, min_branch_px=20,
                                    bridge_gap_px=args.bridge_gap_px, evidence=prob)
            m = match_traces(cls, item.centerlines, max_distance=15.0)
            cover = coverage_report(item.centerlines, cls, tolerance=6.0)
            matched += len(m.pairs)
            total += m.n_gt
            per_image_rows.append({
                "fold": k, "image_id": image_id,
                "dice_labelled": dice(pred & valid, item.mask & valid),
                "dice_all_pixels": dice(pred, item.mask),
                "coverage": float(cover.mean()) if cover.size else float("nan"),
                "n_manual": m.n_gt, "n_detected": m.n_pred, "matched": len(m.pairs),
            })
            comp = compare_metrics(cls, item.centerlines, m, pixel_size=args.pixel_size_nm)
            comp.insert(0, "image_id", image_id)
            comp.insert(0, "fold", k)
            comparisons.append(comp)

        fold_rows.append({
            "fold": k, "best_val_loss": history.best_val, "best_epoch": history.best_epoch,
            "matched": matched, "n_manual": total,
        })
        print(f"    test: {matched}/{total} fibrils matched\n")

    per_image = pd.DataFrame(per_image_rows)
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    errors = summarize_errors(comparison)

    args.reports.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.reports / f"{args.name}_per_image.csv", index=False)
    comparison.to_csv(args.reports / f"{args.name}_matched_fibrils.csv", index=False)
    errors.to_csv(args.reports / f"{args.name}_metric_errors.csv", index=False)

    pooled = {
        "name": args.name,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folds": args.folds,
        "n_images": int(len(per_image)),
        "n_fibrils": int(per_image["n_manual"].sum()),
        "matched": int(per_image["matched"].sum()),
        "recall": float(per_image["matched"].sum() / per_image["n_manual"].sum()),
        "mean_dice_labelled": float(per_image["dice_labelled"].mean()),
        "mean_dice_all_pixels": float(per_image["dice_all_pixels"].mean()),
        "mean_coverage": float(per_image["coverage"].mean()),
        "mean_detected": float(per_image["n_detected"].mean()),
        "per_fold": fold_rows,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.reports / f"{args.name}_summary.json").write_text(json.dumps(pooled, indent=2))

    print("=" * 62)
    print(f"pooled over {pooled['n_images']} images and {pooled['n_fibrils']} fibrils")
    print(f"  recall          {pooled['recall']:.3f} "
          f"({pooled['matched']}/{pooled['n_fibrils']})")
    print(f"  coverage        {pooled['mean_coverage']:.3f}")
    print(f"  dice labelled   {pooled['mean_dice_labelled']:.3f}   "
          f"all px {pooled['mean_dice_all_pixels']:.3f}")
    print(f"  objects/image   {pooled['mean_detected']:.0f}")
    spread = [f["matched"] / f["n_manual"] for f in fold_rows if f["n_manual"]]
    print(f"  recall per fold {', '.join(f'{s:.2f}' for s in spread)} "
          f"(sd {np.std(spread):.3f})")
    if not errors.empty:
        print("\nmorphology error on matched fibrils (nm):")
        print(errors.to_string(index=False))
    print(f"\nwritten to {args.reports}/{args.name}_*")


if __name__ == "__main__":
    main()
