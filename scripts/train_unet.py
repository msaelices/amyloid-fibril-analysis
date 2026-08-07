"""Train the fibril U-Net from micrographs plus their manual traces.

Usage:
    python scripts/train_unet.py --data data/dataset --out models/unet.pt \
        --epochs 40 --n-val 8

Expects ``<data>/images/<id>.png`` and ``<data>/traces/<id>.csv`` (columns
``filament_id,x,y``). Traces are snapped onto the fibril ridge and rasterized
into masks; unannotated ridge-like pixels are marked ignore rather than
background (see :mod:`afa.segment.dataset` for why).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=Path("models/unet.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=8, help="whole images held out")
    ap.add_argument("--base", type=int, default=16, help="U-Net base width")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width-px", type=int, default=7, help="rasterized fibril width")
    ap.add_argument("--no-snap", action="store_true", help="traces already centered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("data/dataset/cache"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from afa.segment.dataset import PatchDataset, load_labelled_images, split_images
    from afa.segment.torch_unet import train

    image_ids = sorted(p.stem for p in (args.data / "images").glob("*.png"))
    train_ids, val_ids = split_images(image_ids, n_val=args.n_val, seed=args.seed)
    print(f"{len(image_ids)} images -> {len(train_ids)} train / {len(val_ids)} val")
    print(f"validation images: {', '.join(val_ids)}")

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
        device=args.device,
        seed=args.seed,
    )

    report = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "best_val_loss": history.best_val,
        "best_epoch": history.best_epoch,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    report_path = Path(args.out).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nbest val loss {history.best_val:.4f} at epoch {history.best_epoch + 1}")
    print(f"weights -> {args.out}\nreport  -> {report_path}")


if __name__ == "__main__":
    main()
