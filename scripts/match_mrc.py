"""Match original .mrc micrographs to the screenshots they were annotated on.

The annotation batch was traced on 4x-downsampled screenshots. This recovers
which micrograph produced which screenshot, so measurements can be reported in
nm and, later, the model can be trained at full resolution.

Writes/updates a mapping CSV: image_id, mrc path, pixel sizes, match score.
Unmatched micrographs are reported and not written, since a wrong pairing would
silently corrupt the dataset.

Usage:
    python scripts/match_mrc.py ~/Downloads/*.mrc
    python scripts/match_mrc.py ~/Downloads --out data/dataset/mrc_map.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from afa.io.match import MATCH_THRESHOLD, MatchResult, match_micrograph
from afa.io.mrc import load_mrc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", type=Path, nargs="+", help=".mrc files or a directory")
    ap.add_argument("--data", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out", type=Path, default=Path("data/dataset/mrc_map.csv"))
    ap.add_argument("--factor", type=int, default=4, help="screenshot downsample factor")
    ap.add_argument("--threshold", type=float, default=None,
                    help="minimum correlation to accept a pairing")
    args = ap.parse_args()

    threshold = args.threshold if args.threshold is not None else MATCH_THRESHOLD

    files: list[Path] = []
    for p in args.paths:
        files.extend(sorted(p.glob("*.mrc")) if p.is_dir() else [p])
    if not files:
        raise SystemExit("no .mrc files found")

    screenshots = {
        p.stem: np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        for p in sorted((args.data / "images").glob("*.png"))
    }
    if not screenshots:
        raise SystemExit(f"no screenshots under {args.data / 'images'}")
    print(f"{len(files)} micrograph(s) against {len(screenshots)} screenshots\n")

    existing: dict[str, dict] = {}
    if args.out.exists():
        with open(args.out) as fh:
            existing = {row["image_id"]: row for row in csv.DictReader(fh)}

    results = []
    for path in files:
        img = load_mrc(path)
        scores = match_micrograph(img.data, screenshots, factor=args.factor)
        best, runner = scores[0], (scores[1] if len(scores) > 1 else (0.0, None))
        shot = screenshots[best[1]]
        result = MatchResult(
            micrograph=path,
            best_id=best[1],
            best_score=best[0],
            runner_up_id=runner[1],
            runner_up_score=runner[0],
            scale=img.data.shape[1] / shot.shape[1],
            pixel_size_a=img.pixel_size_a,
        )
        results.append(result)

        if result.best_score >= threshold:
            print(
                f"  {path.name}\n    -> {result.best_id}  r={result.best_score:.3f} "
                f"(next {result.runner_up_score:.3f})  "
                f"screenshot {result.screenshot_pixel_size_a:.3f} A/px"
            )
        else:
            print(
                f"  {path.name}\n    -> NO MATCH (best {result.best_id} r="
                f"{result.best_score:.3f}, below {threshold})"
            )

    matched = [r for r in results if r.best_score >= threshold]
    for r in matched:
        existing[r.best_id] = {
            "image_id": r.best_id,
            "mrc_path": str(r.micrograph.resolve()),
            "mrc_pixel_size_a": f"{r.pixel_size_a:.4f}",
            "screenshot_pixel_size_a": f"{r.screenshot_pixel_size_a:.4f}",
            "scale": f"{r.scale:.4f}",
            "match_score": f"{r.best_score:.4f}",
        }

    if matched:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image_id", "mrc_path", "mrc_pixel_size_a",
            "screenshot_pixel_size_a", "scale", "match_score",
        ]
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for image_id in sorted(existing):
                writer.writerow(existing[image_id])

    print(
        f"\n{len(matched)}/{len(results)} matched. "
        f"{len(existing)}/{len(screenshots)} screenshots now have a micrograph."
    )
    if matched:
        print(f"mapping -> {args.out}")
    sizes = {r.screenshot_pixel_size_a for r in matched}
    if len(sizes) > 1:
        print(
            f"WARNING: screenshot pixel size is not constant across matches: "
            f"{sorted(round(s, 3) for s in sizes)}"
        )


if __name__ == "__main__":
    main()
