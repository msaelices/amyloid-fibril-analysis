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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from afa.io.annotations import Trace, load_traces
from afa.segment.tiling import tile_positions


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
    from PIL import Image

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
    from skimage.morphology import binary_dilation, disk

    from afa.segment.classical import vesselness_probability
    from afa.segment.unet import rasterize_traces
    from afa.trace.snap import snap_to_ridge

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
    persist the result; the cache key includes the labelling parameters, so
    changing them rebuilds rather than silently reusing stale masks.
    """
    images_dir, traces_dir = Path(images_dir), Path(traces_dir)
    ids = image_ids or sorted(p.stem for p in images_dir.glob("*.png"))

    key = "_".join(f"{k}={v}" for k, v in sorted(label_kwargs.items())) or "default"
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


def split_images(image_ids: list[str], *, n_val: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministically hold out ``n_val`` whole images for validation."""
    if n_val >= len(image_ids):
        raise ValueError(f"n_val ({n_val}) must be < number of images ({len(image_ids)})")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(sorted(image_ids))[rng.permutation(len(image_ids))])
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


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
        import torch

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
