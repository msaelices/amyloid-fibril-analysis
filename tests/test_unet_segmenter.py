"""Tests for the U-Net segmenter's inference path.

Skipped when the ``dl`` extra is not installed, so the deterministic core stays
testable without torch.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from afa.segment.torch_unet import UNet  # noqa: E402
from afa.segment.unet import UNetSegmenter  # noqa: E402


@pytest.fixture
def tiny_weights(tmp_path):
    """A randomly initialized checkpoint; enough to exercise the plumbing."""
    model = UNet(base=4, depth=2)
    path = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict(), "base": 4, "depth": 2}, path)
    return path


def test_predict_returns_probability_map_of_input_shape(tiny_weights):
    seg = UNetSegmenter(weights=tiny_weights, patch=64, overlap=16).load()
    image = np.random.default_rng(0).normal(0.5, 0.1, size=(150, 210)).astype(np.float32)

    prob = seg.predict(image)

    assert prob.shape == image.shape
    assert prob.dtype == np.float32
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_predict_accepts_float64_and_integer_images(tiny_weights):
    """Percentile normalization promotes to float64; the model is float32."""
    seg = UNetSegmenter(weights=tiny_weights, patch=64, overlap=16).load()

    as_double = np.random.default_rng(1).normal(0.5, 0.1, size=(100, 100)).astype(np.float64)
    as_uint8 = (np.random.default_rng(2).random((100, 100)) * 255).astype(np.uint8)

    assert seg.predict(as_double).shape == (100, 100)
    assert seg.predict(as_uint8).shape == (100, 100)


def test_predict_pads_images_smaller_than_one_patch(tiny_weights):
    seg = UNetSegmenter(weights=tiny_weights, patch=64, overlap=16).load()
    small = np.zeros((40, 50), dtype=np.float32)
    assert seg.predict(small).shape == (40, 50)


def test_load_requires_existing_weights(tmp_path):
    with pytest.raises(ValueError):
        UNetSegmenter().load()
    with pytest.raises(FileNotFoundError):
        UNetSegmenter(weights=tmp_path / "missing.pt").load()


def test_unet_forward_preserves_spatial_shape():
    model = UNet(base=4, depth=3)
    out = model(torch.zeros(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64)
