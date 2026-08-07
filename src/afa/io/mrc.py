"""Read cryo-EM ``.mrc`` micrographs, including the physical pixel size."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MrcImage:
    """A 2D micrograph plus its physical calibration.

    Attributes
    ----------
    data:
        2D float32 array (single micrograph). If the file holds a stack, the
        requested ``frame`` is returned.
    pixel_size_a:
        Pixel size in Angstrom/pixel, read from the MRC header when available.
    path:
        Source file path.
    """

    data: np.ndarray
    pixel_size_a: float
    path: Path

    @property
    def pixel_size_nm(self) -> float:
        return self.pixel_size_a / 10.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]


def load_mrc(
    path: str | Path,
    *,
    frame: int = 0,
    pixel_size_a: float | None = None,
) -> MrcImage:
    """Load a 2D micrograph from an ``.mrc`` file.

    Parameters
    ----------
    path:
        Path to the ``.mrc`` file.
    frame:
        If the file is a stack (3D), which z-slice / frame to return.
    pixel_size_a:
        Override the pixel size (Angstrom/pixel). If ``None``, it is read from
        the header (``voxel_size.x``); if the header is unset it falls back to
        ``1.0`` with a warning-friendly value the caller can detect.

    Returns
    -------
    MrcImage
    """
    import mrcfile

    path = Path(path)
    with mrcfile.open(path, permissive=True) as mrc:
        arr = np.asarray(mrc.data)
        header_px = float(mrc.voxel_size.x) if mrc.voxel_size.x else 0.0

    if arr.ndim == 3:
        if not (0 <= frame < arr.shape[0]):
            raise IndexError(f"frame {frame} out of range for stack of {arr.shape[0]}")
        arr = arr[frame]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported MRC ndim={arr.ndim} for {path}")

    px = pixel_size_a if pixel_size_a is not None else (header_px or 1.0)
    return MrcImage(data=arr.astype(np.float32, copy=False), pixel_size_a=px, path=path)


def normalize(image: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-clip and scale an image to [0, 1] for display/processing."""
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    out = (image - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
