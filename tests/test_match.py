"""Tests for matching a micrograph to the screenshot it was annotated on.

A wrong pairing silently corrupts the dataset, so the separation between a true
match and a near miss is what these pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from afa.io.match import block_average, carbon_mask, correlation, match_micrograph


def _fibril_field(seed: int, shape=(512, 720), n=12) -> np.ndarray:
    """Grey field with a few dark streaks, plus a dark carbon corner."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 0.6, dtype=np.float32)
    for _ in range(n):
        y0, x0 = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        dy, dx = rng.normal(size=2)
        dy, dx = (dy, dx) / np.hypot(dy, dx)
        for t in range(180):
            y, x = int(y0 + dy * t), int(x0 + dx * t)
            if 0 <= y < shape[0] - 2 and 0 <= x < shape[1] - 2:
                img[y:y + 3, x:x + 3] = 0.25
    # The carbon support: a large dark corner, identical in every field.
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img[(yy + xx) > (shape[0] + shape[1]) * 0.82] = 0.05
    return img


def test_block_average_reduces_shape_and_preserves_the_mean():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = block_average(img, 4)
    assert out.shape == (2, 2)
    assert out.mean() == pytest.approx(img.mean())


def test_block_average_trims_a_ragged_edge():
    assert block_average(np.zeros((10, 11), dtype=np.float32), 4).shape == (2, 2)


def test_block_average_is_a_noop_at_factor_one():
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert np.array_equal(block_average(img, 1), img)


def test_carbon_mask_excludes_the_dark_support():
    img = _fibril_field(0)
    mask = carbon_mask(img)
    # The support corner is excluded, the middle of the field is kept.
    assert not mask[-20, -20]
    assert mask[img.shape[0] // 2, img.shape[1] // 2]
    assert 0.05 < (~mask).mean() < 0.5


def test_correlation_of_an_image_with_itself_is_one():
    a = _fibril_field(1)
    assert correlation(a, a) == pytest.approx(1.0, abs=1e-5)


def test_correlation_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        correlation(np.zeros((4, 4)), np.zeros((4, 5)))


def test_a_true_pair_outscores_every_other_candidate():
    """The micrograph is a 4x-upsampled version of one screenshot."""
    from PIL import Image

    shots = {f"slide{i:02d}": _fibril_field(i) for i in range(6)}
    target = "slide03"
    big = np.asarray(
        Image.fromarray(shots[target]).resize(
            (shots[target].shape[1] * 4, shots[target].shape[0] * 4), Image.BILINEAR
        ),
        dtype=np.float32,
    )

    scores = match_micrograph(big, shots)

    assert scores[0][1] == target
    assert scores[0][0] > 0.85
    assert scores[0][0] - scores[1][0] > 0.2, f"not separated: {scores[:2]}"


def test_masking_the_carbon_is_what_separates_the_candidates():
    """The trap this module exists for, pinned in both directions.

    Two micrographs with entirely different fibrils correlate strongly while the
    shared support film is left in — it is a large, high-contrast feature common
    to every image of the same grid square. Masking it collapses the score to
    what the fibrils actually justify.
    """
    a, b = _fibril_field(1), _fibril_field(2)

    unmasked = correlation(a, b)
    masked = correlation(a, b, mask=carbon_mask(a))

    assert unmasked > 0.8, "the carbon alone makes unrelated images look identical"
    assert masked < 0.5
    assert unmasked - masked > 0.3
