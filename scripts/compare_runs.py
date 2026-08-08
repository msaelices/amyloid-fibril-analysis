"""Rank trained checkpoints on a fixed metric, independent of the loss they used.

Why not just compare `best_val_loss` from the run reports: changing `pos_weight`
changes the loss function itself, so its values land on different scales and
ranking them would be meaningless. Anything that varies the objective has to be
compared on a metric held fixed across all runs.

Everything here is measured on the **validation** split. That is the split model
selection already uses, so choosing hyperparameters on it is consistent, but it
also means these numbers are optimistic by construction. Run
``scripts/validate.py --split test`` on the winner for an honest figure.

Usage:
    python scripts/compare_runs.py models/sweep_*.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path, nargs="+")
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--cache", type=Path, default=Path("data/dataset/cache"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--width-px", type=int, default=7)
    args = ap.parse_args()

    from afa.segment.dataset import load_labelled_images, split_images
    from afa.segment.unet import UNetSegmenter
    from afa.trace.tracer import trace_centerlines
    from afa.validate import coverage_report, dice, match_traces

    image_ids = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    _, val_ids, _ = split_images(
        image_ids, n_val=args.n_val, n_test=args.n_test, seed=args.seed
    )
    items = load_labelled_images(
        args.data / "images", args.data / "traces",
        image_ids=val_ids, cache_dir=args.cache, width_px=args.width_px,
    )
    print(f"comparing {len(args.weights)} runs on {len(items)} validation images\n")

    rows = []
    for path in args.weights:
        seg = UNetSegmenter(weights=path).load()
        dices, covs, matched, total, objs = [], [], 0, 0, []
        for item in items:
            prob = seg.predict(item.image)
            pred = prob > args.threshold
            valid = ~item.ignore
            dices.append(dice(pred & valid, item.mask & valid))
            cls = trace_centerlines(pred, min_branch_px=20)
            objs.append(len(cls))
            covs.append(coverage_report(item.centerlines, cls, tolerance=6.0).mean())
            m = match_traces(cls, item.centerlines, max_distance=15.0)
            matched += len(m.pairs)
            total += m.n_gt

        report_path = args.reports / f"{path.stem}.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        rows.append(
            {
                "run": path.stem,
                "pos_weight": report.get("args", {}).get("pos_weight", "?"),
                "schedule": "no" if report.get("args", {}).get("no_schedule") == "True" else "yes",
                "dice": sum(dices) / len(dices),
                "coverage": sum(covs) / len(covs),
                "recall": matched / total if total else float("nan"),
                "objects": sum(objs) / len(objs),
                "best_val_loss": report.get("best_val_loss"),
            }
        )
        print(f"  {path.stem} done")

    rows.sort(key=lambda r: -r["coverage"])
    print(f"\n{'run':<16}{'pos_w':>7}{'sched':>7}{'dice':>7}{'cover':>7}"
          f"{'recall':>8}{'objs':>7}{'val_loss':>10}")
    for r in rows:
        loss = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "-"
        print(f"{r['run']:<16}{r['pos_weight']:>7}{r['schedule']:>7}{r['dice']:>7.3f}"
              f"{r['coverage']:>7.2f}{r['recall']:>8.2f}{r['objects']:>7.0f}{loss:>10}")
    print("\nval_loss is shown for reference only; it is NOT comparable across "
          "different pos_weight values, which is the reason this script exists.")

    out = args.reports / "sweep_comparison.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
