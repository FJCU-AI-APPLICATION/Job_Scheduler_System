"""Tests for the vendored Sutton & Barto tile-coding library."""

import pickle

import numpy as np
import pytest


def test_tiles_deterministic():
    """Same (IHT, features, ints) → same hash slots across calls."""
    from ai.optimizers.tiles3 import IHT, tiles

    iht = IHT(size=4096)
    a = tiles(iht, num_tilings=8, floats=[1.5, 2.5, 3.5], ints=[0])
    b = tiles(iht, num_tilings=8, floats=[1.5, 2.5, 3.5], ints=[0])
    assert a == b
    assert len(a) == 8


def test_tiles_iht_capped():
    """All returned indices are < iht_size, even after flooding."""
    from ai.optimizers.tiles3 import IHT, tiles

    iht = IHT(size=64)
    for i in range(1000):
        idx = tiles(iht, num_tilings=8, floats=[i * 0.7, i * 1.1], ints=[i % 9])
        assert all(0 <= j < 64 for j in idx)


def test_tiles_low_collision_rate():
    """On 10K distinct feature vectors at 10x scale + 8 tilings, collision rate < 1%."""
    from ai.optimizers.tiles3 import IHT, tiles

    iht = IHT(size=4096)
    rng = np.random.default_rng(0)
    seen: dict[tuple[int, ...], int] = {}
    collisions = 0
    for _ in range(10_000):
        feats = (rng.random(10) * 10.0).tolist()
        action = int(rng.integers(0, 9))
        key = tuple(tiles(iht, num_tilings=8, floats=feats, ints=[action]))
        if key in seen:
            collisions += 1
        seen[key] = 1
    assert collisions / 10_000 < 0.01


def test_iht_serializes():
    """iht.dictionary pickle-round-trips; new IHT from dict produces same hashes."""
    from ai.optimizers.tiles3 import IHT, tiles

    iht_a = IHT(size=4096)
    sample_features = [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]
    keys_a = [
        tuple(tiles(iht_a, num_tilings=8, floats=f, ints=[0])) for f in sample_features
    ]

    data = pickle.dumps(iht_a.dictionary)
    iht_b = IHT(size=4096)
    iht_b.dictionary = pickle.loads(data)
    keys_b = [
        tuple(tiles(iht_b, num_tilings=8, floats=f, ints=[0])) for f in sample_features
    ]
    assert keys_a == keys_b
