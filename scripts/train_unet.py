"""Train the fibril U-Net from micrographs plus their manual traces.

Usage:
    python scripts/train_unet.py --data data/dataset --out models/unet.pt \
        --epochs 40 --n-val 8

Expects ``<data>/images/<id>.png`` and ``<data>/traces/<id>.csv`` (columns
``filament_id,x,y``). Traces are snapped onto the fibril ridge and rasterized
into masks; unannotated ridge-like pixels are marked ignore rather than
background (see :mod:`afa.segment.dataset` for why).

Weights go to ``models/`` which is gitignored, but the run report goes to
``reports/`` which is tracked: losses, split ids, hyperparameters and the commit
that produced them are the record of what was tried, and they contain no patient
data. ``reports/training_runs.jsonl`` accumulates one line per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from afa.segment.dataset import PatchDataset, load_labelled_images, split_images

# Summary fields appended to the run log. The per-epoch curves stay in the
# individual report; this is the one-line-per-run history.
SUMMARY_FIELDS = (
    "run",
    "finished_at",
    "commit",
    "best_val_loss",
    "best_epoch",
    "final_train_loss",
    "final_val_loss",
    "n_train_patches",
    "n_val_patches",
    "n_fibrils",
)


def _git_commit() -> str | None:
    """Short SHA of the code that produced the run, so a result is traceable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def write_report(report: dict, reports_dir: Path) -> None:
    """Write the full report, and append a summary line to the run log.

    Two files on purpose. The per-run JSON holds the epoch-by-epoch curves and
    is overwritten when a run of the same name is repeated. The JSONL is only
    ever appended to, so the record of what was tried survives re-runs -- which
    is the part that was previously lost, since reports lived under models/ and
    that directory is not tracked.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{report['run']}.json").write_text(json.dumps(report, indent=2))

    summary = {k: report.get(k) for k in SUMMARY_FIELDS}
    summary["args"] = report.get("args", {})
    with open(reports_dir / "training_runs.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=8, help="images for model selection")
    ap.add_argument("--n-test", type=int, default=8,
                    help="images never seen by training OR checkpoint selection")
    ap.add_argument("--base", type=int, default=16, help="U-Net base width")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=10.0,
                    help="BCE weight on fibril pixels; changes the loss scale")
    ap.add_argument("--cosine-schedule", action="store_true",
                    help="anneal the learning rate; measured worse here, see reports/README.md")
    ap.add_argument("--width-px", type=int, default=7, help="rasterized fibril width")
    ap.add_argument("--no-snap", action="store_true", help="traces already centered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("data/dataset/cache"))
    ap.add_argument("--reports", type=Path, default=Path("reports"),
                    help="tracked directory for run reports (weights stay in models/)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Deferred: torch is an optional dependency, and tests/test_train_report.py
    # imports write_report from this module without it installed.
    from afa.segment.torch_unet import train  # noqa: PLC0415

    image_ids = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    train_ids, val_ids, test_ids = split_images(
        image_ids, n_val=args.n_val, n_test=args.n_test, seed=args.seed
    )
    print(
        f"{len(image_ids)} images -> {len(train_ids)} train / "
        f"{len(val_ids)} val / {len(test_ids)} test"
    )
    print(f"val (selection):  {', '.join(val_ids)}")
    print(f"test (untouched): {', '.join(test_ids)}")

    print("\nbuilding labels (snapping traces onto the ridge)...")
    items = load_labelled_images(
        args.data / "images",
        args.data / "traces",
        cache_dir=args.cache,
        width_px=args.width_px,
        snap=not args.no_snap,
    )
    by_id = {it.image_id: it for it in items}
    train_items = [by_id[i] for i in train_ids if i in by_id]
    val_items = [by_id[i] for i in val_ids if i in by_id]

    total_fibrils = sum(len(it.centerlines) for it in items)
    frac_pos = sum(it.mask.mean() for it in items) / max(len(items), 1)
    frac_ign = sum(it.ignore.mean() for it in items) / max(len(items), 1)
    print(
        f"{total_fibrils} fibrils | mean {100 * frac_pos:.2f}% positive pixels, "
        f"{100 * frac_ign:.2f}% ignored"
    )

    train_ds = PatchDataset(
        train_items, patch=args.patch, overlap=args.overlap, augment_data=True, seed=args.seed
    )
    val_ds = PatchDataset(
        val_items, patch=args.patch, overlap=args.overlap, augment_data=False, seed=args.seed
    )
    print(f"patches: {len(train_ds)} train / {len(val_ds)} val\n")
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("no usable patches; lower --patch or check the traces")

    history = train(
        train_ds,
        val_ds,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base=args.base,
        depth=args.depth,
        pos_weight=args.pos_weight,
        cosine_schedule=args.cosine_schedule,
        device=args.device,
        seed=args.seed,
    )

    report = {
        "run": args.out.stem,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "best_val_loss": history.best_val,
        "best_epoch": history.best_epoch,
        "final_train_loss": history.train_loss[-1] if history.train_loss else None,
        "final_val_loss": history.val_loss[-1] if history.val_loss else None,
        "n_train_patches": len(train_ds),
        "n_val_patches": len(val_ds),
        "n_fibrils": total_fibrils,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    write_report(report, args.reports)
    print(f"\nbest val loss {history.best_val:.4f} at epoch {history.best_epoch + 1}")
    print(f"weights -> {args.out}")
    print(f"report  -> {args.reports / (report['run'] + '.json')}")
    print(f"history -> {args.reports / 'training_runs.jsonl'}")


if __name__ == "__main__":
    main()
