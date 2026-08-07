"""Tests for dataset splitting and label caching."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from afa.segment.dataset import load_labelled_images, split_images


def test_split_is_three_way_disjoint_and_exhaustive():
    ids = [f"img{i:02d}" for i in range(20)]

    train, val, test = split_images(ids, n_val=4, n_test=5, seed=0)

    assert len(val) == 4 and len(test) == 5 and len(train) == 11
    assert set(train) | set(val) | set(test) == set(ids)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_split_is_deterministic_and_seed_sensitive():
    ids = [f"img{i:02d}" for i in range(20)]

    assert split_images(ids, n_val=4, n_test=4, seed=0) == split_images(
        ids, n_val=4, n_test=4, seed=0
    )
    assert split_images(ids, n_val=4, n_test=4, seed=1) != split_images(
        ids, n_val=4, n_test=4, seed=0
    )


def test_split_rejects_asking_for_more_than_exists():
    with pytest.raises(ValueError):
        split_images([f"img{i}" for i in range(6)], n_val=3, n_test=3)


def _tiny_dataset(tmp_path, y):
    """One image with a dark horizontal line, plus a trace CSV following it."""
    images, traces = tmp_path / "images", tmp_path / "traces"
    images.mkdir(exist_ok=True)
    traces.mkdir(exist_ok=True)

    arr = np.full((80, 120), 200, dtype=np.uint8)
    arr[y - 1:y + 2, 10:110] = 40
    Image.fromarray(arr).save(images / "a.png")

    rows = ["filament_id,x,y"] + [f"f0,{x},{y}" for x in range(15, 105, 5)]
    (traces / "a.csv").write_text("\n".join(rows) + "\n")
    return images, traces


def test_cache_is_invalidated_when_the_traces_change(tmp_path):
    """Re-importing annotations must not silently serve the old masks."""
    cache = tmp_path / "cache"
    images, traces = _tiny_dataset(tmp_path, y=30)

    first = load_labelled_images(images, traces, cache_dir=cache, snap=False, width_px=3)[0]

    # Same parameters, completely different annotation.
    _tiny_dataset(tmp_path, y=60)
    second = load_labelled_images(images, traces, cache_dir=cache, snap=False, width_px=3)[0]

    assert not np.array_equal(first.mask, second.mask)
    assert second.centerlines[0][:, 1].mean() == pytest.approx(60, abs=2)


def test_cache_is_reused_when_nothing_changed(tmp_path):
    cache = tmp_path / "cache"
    images, traces = _tiny_dataset(tmp_path, y=30)

    first = load_labelled_images(images, traces, cache_dir=cache, snap=False, width_px=3)[0]
    second = load_labelled_images(images, traces, cache_dir=cache, snap=False, width_px=3)[0]

    assert np.array_equal(first.mask, second.mask)
    assert len(list(cache.iterdir())) == 1
