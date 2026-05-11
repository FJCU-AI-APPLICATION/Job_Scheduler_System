"""Tile coding for linear function approximation in RL.

Vendored from Richard Sutton's reference implementation:
  http://incompleteideas.net/tiles/tiles3.html
Original author: Richard S. Sutton. BSD-style permissive license.

Used by LAST-RL (issue #19) to map continuous state features + discrete
actions to a sparse binary index vector for SARSA(lambda).

Public API:
  IHT(size)                                      - bounded-capacity hash table
  tiles(iht, num_tilings, floats, ints=None)     - produces num_tilings hash slots
"""

from __future__ import annotations

from math import floor


basehash = hash


class IHT:
    """Index Hash Table - bounded-size hash table for tile-coding indices.

    Maintains a dict of seen (tuple) keys to integer slots in [0, size).
    On collision or overflow, falls back to plain hash modulo size.
    """

    def __init__(self, size: int):
        self.size = size
        self.overfull_count = 0
        self.dictionary: dict[tuple, int] = {}

    def __str__(self):
        return (
            f"IHT(size={self.size}, count={len(self.dictionary)}, "
            f"overfull_count={self.overfull_count})"
        )

    def count(self) -> int:
        return len(self.dictionary)

    def full(self) -> bool:
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only: bool = False) -> int:
        d = self.dictionary
        if obj in d:
            return d[obj]
        if read_only:
            return -1
        size = self.size
        count = len(d)
        if count >= size:
            if self.overfull_count == 0:
                print("IHT count exceeded size; using random fallback")
            self.overfull_count += 1
            return basehash(obj) % size
        d[obj] = count
        return count


def hashcoords(coordinates, m, read_only: bool = False) -> int:
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return basehash(tuple(coordinates)) % m
    if m is None:
        return coordinates
    raise TypeError(f"Unknown m type: {type(m)}")


def tiles(
    iht_or_size,
    num_tilings: int,
    floats: list[float],
    ints: list[int] | None = None,
    read_only: bool = False,
) -> list[int]:
    """Returns num_tilings hash slots for the given continuous + discrete features."""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    out: list[int] = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords: list[int] = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        out.append(hashcoords(coords, iht_or_size, read_only))
    return out
