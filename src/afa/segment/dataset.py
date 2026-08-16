"""Build training data from micrographs plus their manual traces.

The annotation set has two properties that shape everything here.

**The drawn traces sit beside the fibrils, not on them.** They are snapped onto
the ridge first (:mod:`afa.trace.snap`); rasterizing the raw drawn curve would
label empty background.

**The annotation is partial.** Only a handful of fibrils per micrograph were
traced, while many more are clearly visible. Treating every unlabelled pixel as
background teaches the model to suppress exactly what it is supposed to find. So
pixels that look strongly fibril-like but were not annotated are marked
*ignore* and excluded from the loss, rather than asserted to be background. This
is a deliberate trade: the model gets fewer, cleaner negatives instead of many
wrong ones.

Splitting is by whole image, never by patch: patches from one micrograph overlap
and share noise statistics, so mixing them across train/val leaks and inflates
the validation score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk

from afa.io.annotations import Trace, load_traces
from afa.segment.classical import vesselness_probability
from afa.segment.tiling import tile_positions
from afa.segment.unet import rasterize_traces
from afa.trace.snap import snap_to_ridge


@dataclass
class LabelledImage:
    """One micrograph with its supervision targets, all in pixel space."""

    image_id: str
    image: np.ndarray        # float32, normalized to [0, 1]
    mask: np.ndarray         # bool, True on annotated fibrils
    ignore: np.ndarray       # bool, True where the label is unknown
    centerlines: list[np.ndarray]

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape  # type: ignore[return-value]


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load a PNG/TIFF micrograph as float32 in [0, 1] via percentile clipping."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(arr)
    # np.percentile returns float64, which would promote the whole image.
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_labels(
    image: np.ndarray,
    traces: list[Trace],
    *,
    width_px: int = 7,
    snap: bool = True,
    max_shift: float = 25.0,
    ignore_quantile: float = 0.98,
    ignore_dilation: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Turn drawn traces into a fibril mask plus an ignore mask.

    Parameters
    ----------
    image:
        Normalized micrograph.
    traces:
        Manual traces in pixel coordinates (drawn beside the fibril).
    width_px:
        Rasterized fibril width; set it to the typical apparent fibril width.
    snap:
        Snap each trace onto the ridge before rasterizing. Turn off only if the
        traces are already centered on the fibrils.
    max_shift:
        Search half-width for snapping, in pixels.
    ignore_quantile:
        Pixels above this quantile of the vesselness map that are not part of an
        annotated fibril are marked ignore rather than background.
    ignore_dilation:
        Radius by which the ignore region is grown, so the soft edges of an
        unannotated fibril are excluded too.

    Returns
    -------
    ``(mask, ignore, centerlines)``
    """
    ridge = vesselness_probability(image, invert=True, sigmas=(2.0, 3.0, 4.0, 5.0))

    centerlines = []
    for tr in traces:
        pts = snap_to_ridge(tr.points, ridge, max_shift=max_shift).points if snap else tr.points
        centerlines.append(np.asarray(pts, dtype=float))

    snapped = [
        Trace(filament_id=t.filament_id, points=c)
        for t, c in zip(traces, centerlines, strict=True)
    ]
    mask = rasterize_traces(snapped, image.shape, width_px=width_px)  # type: ignore[arg-type]

    # Anything ridge-like that nobody annotated is "unknown", not "background".
    threshold = float(np.quantile(ridge, ignore_quantile))
    ignore = (ridge > threshold) & ~mask
    if ignore_dilation > 0:
        ignore = binary_dilation(ignore, disk(ignore_dilation))
    ignore &= ~mask
    return mask, ignore, centerlines


def _cache_key(
    images_dir: Path, traces_dir: Path, ids: list[str], label_kwargs: dict
) -> str:
    """Cache key covering the labelling parameters *and* the inputs themselves.

    Keying on the keyword arguments alone was not enough: re-importing the
    annotations rewrites the trace CSVs without changing any parameter, and the
    stale masks were served back silently. The trace files are hashed by content
    (they are small); images by size and mtime, since they are large and are not
    edited in place.
    """
    h = hashlib.sha256()
    for k, v in sorted(label_kwargs.items()):
        h.update(f"{k}={v!r};".encode())
    for image_id in ids:
        trace_path = traces_dir / f"{image_id}.csv"
        image_path = images_dir / f"{image_id}.png"
        if trace_path.exists():
            h.update(trace_path.read_bytes())
        if image_path.exists():
            stat = image_path.stat()
            h.update(f"{image_id}:{stat.st_size}:{int(stat.st_mtime)};".encode())
    return h.hexdigest()[:16]


def load_labelled_images(
    images_dir: str | Path,
    traces_dir: str | Path,
    *,
    image_ids: list[str] | None = None,
    cache_dir: str | Path | None = None,
    **label_kwargs,
) -> list[LabelledImage]:
    """Load every ``<id>.png`` in ``images_dir`` that has a ``<id>.csv`` of traces.

    Label building runs a multiscale ridge filter and a dynamic program per
    trace, which costs tens of seconds per micrograph. Pass ``cache_dir`` to
    persist the result; the key covers the labelling parameters *and* the
    content of the trace files, so re-importing the annotations invalidates the
    cache instead of silently serving stale masks.
    """
    images_dir, traces_dir = Path(images_dir), Path(traces_dir)
    ids = image_ids or sorted(p.stem for p in images_dir.glob("*.png"))

    key = _cache_key(images_dir, traces_dir, ids, label_kwargs)
    cache = Path(cache_dir) / key if cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    out = []
    for image_id in ids:
        trace_path = traces_dir / f"{image_id}.csv"
        if not trace_path.exists():
            continue
        image = load_grayscale(images_dir / f"{image_id}.png")

        cached = cache / f"{image_id}.npz" if cache else None
        if cached is not None and cached.exists():
            blob = np.load(cached, allow_pickle=False)
            mask, ignore = blob["mask"], blob["ignore"]
            centerlines = [blob[k] for k in sorted(blob) if k.startswith("cl_")]
        else:
            traces = load_traces(trace_path)
            mask, ignore, centerlines = build_labels(image, traces, **label_kwargs)
            if cached is not None:
                np.savez_compressed(
                    cached,
                    mask=mask,
                    ignore=ignore,
                    **{f"cl_{i:03d}": c for i, c in enumerate(centerlines)},
                )
        out.append(LabelledImage(image_id, image, mask, ignore, centerlines))
    return out


def split_images(
    image_ids: list[str], *, n_val: int, n_test: int = 0, seed: int = 0
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split whole images into train / val / test.

    Three ways, not two, and the distinction is not pedantry: the training loop
    keeps the checkpoint that scores best on the validation images, so those
    images have been selected on and any score reported over them is optimistic.
    Reporting belongs on the test split, which nothing ever selects against.
    """
    if n_val + n_test >= len(image_ids):
        raise ValueError(
            f"n_val + n_test ({n_val + n_test}) must be < number of images ({len(image_ids)})"
        )
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(sorted(image_ids))[rng.permutation(len(image_ids))])
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def kfold_splits(
    image_ids: list[str], *, n_folds: int = 5, n_val: int = 6, seed: int = 0
) -> list[tuple[list[str], list[str], list[str]]]:
    """Train/val/test triples where every image is tested exactly once.

    A single held-out split leaves too few fibrils to measure anything: with 20,
    a change from 3/20 to 7/20 does not reach significance, so a real improvement
    cannot be told from noise. Rotating the test set over ``n_folds`` folds and
    pooling the results evaluates *every* image with a model that never saw it,
    which is the only way to get the sample size up without more annotation.

    The validation images are drawn afresh per fold, so no image is permanently
    the one that selects checkpoints.

    Returns
    -------
    One ``(train, val, test)`` triple per fold, each sorted.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ids = sorted(image_ids)
    if n_folds > len(ids):
        raise ValueError(f"n_folds ({n_folds}) exceeds the number of images ({len(ids)})")

    rng = np.random.default_rng(seed)
    shuffled = [str(x) for x in np.asarray(ids)[rng.permutation(len(ids))]]
    blocks = [shuffled[i::n_folds] for i in range(n_folds)]

    folds = []
    for i, test in enumerate(blocks):
        rest = [x for x in shuffled if x not in set(test)]
        if n_val >= len(rest):
            raise ValueError(
                f"n_val ({n_val}) leaves no training images in fold {i} ({len(rest)} available)"
            )
        order = np.random.default_rng(seed + 1 + i).permutation(len(rest))
        rotated = [rest[j] for j in order]
        folds.append((sorted(rotated[n_val:]), sorted(rotated[:n_val]), sorted(test)))
    return folds


def sample_patches(
    item: LabelledImage,
    *,
    patch: int = 256,
    overlap: int = 64,
    min_positive: int = 20,
) -> list[tuple[int, int]]:
    """Patch corners worth training on.

    Patches with almost no annotated fibril are dropped: with partial
    annotation an "empty" patch is more likely to hold an untraced fibril than
    genuine background, so it is not a trustworthy negative.
    """
    positions = tile_positions(item.shape, patch, overlap)
    return [
        (y, x)
        for y, x in positions
        if item.mask[y:y + patch, x:x + patch].sum() >= min_positive
    ]


def augment(
    image: np.ndarray,
    mask: np.ndarray,
    ignore: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip/rotate by multiples of 90 degrees; fibrils have no preferred orientation."""
    k = int(rng.integers(4))
    out = [np.rot90(a, k) for a in (image, mask, ignore)]
    if rng.random() < 0.5:
        out = [np.fliplr(a) for a in out]
    if rng.random() < 0.5:
        out = [np.flipud(a) for a in out]
    return tuple(np.ascontiguousarray(a) for a in out)  # type: ignore[return-value]


class PatchDataset:
    """Torch ``Dataset`` of ``(image, mask, weight)`` patches.

    ``weight`` is 0 on ignored pixels and 1 elsewhere, so the loss can skip
    pixels whose label is unknown.
    """

    def __init__(
        self,
        items: list[LabelledImage],
        *,
        patch: int = 256,
        overlap: int = 64,
        min_positive: int = 20,
        augment_data: bool = True,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.patch = patch
        self.augment_data = augment_data
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int, int]] = [
            (i, y, x)
            for i, item in enumerate(items)
            for y, x in sample_patches(
                item, patch=patch, overlap=overlap, min_positive=min_positive
            )
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        # Deferred: torch is an optional dependency (the ``dl`` extra) and the
        # deterministic pipeline must import and run without it.
        import torch  # noqa: PLC0415

        i, y, x = self.index[idx]
        item = self.items[i]
        sl = (slice(y, y + self.patch), slice(x, x + self.patch))
        image, mask, ignore = item.image[sl], item.mask[sl], item.ignore[sl]
        if self.augment_data:
            image, mask, ignore = augment(image, mask, ignore, self.rng)
        return (
            torch.from_numpy(image[None].astype(np.float32)),
            torch.from_numpy(mask[None].astype(np.float32)),
            torch.from_numpy((~ignore)[None].astype(np.float32)),
        )
