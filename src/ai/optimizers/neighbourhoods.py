"""Neighborhood operators for the matheuristic — issue #18.

Each function: (schedule, sp, size_k, rng) -> (frozen, optimize).
The pair is a disjoint partition of range(num_shifts):
  frozen   — shift indices to be locked at their current employee
  optimize — shift indices to be re-optimized by the inner IP

Caller (MatheuristicOptimizer) is responsible for translating these index
sets into CP-SAT bound-locks + hints; see optimizers/matheuristic.py.
"""

from __future__ import annotations

import numpy as np

from ai.domain.problem import SchedulingProblem


def swap_day(
    schedule: list[int],
    sp: SchedulingProblem,
    size_k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Re-optimize size_k consecutive days. Frozen = all shifts outside."""
    day_start = int(rng.integers(0, sp.days - size_k + 1))
    shift_lo = day_start * sp.shifts_per_day
    shift_hi = shift_lo + size_k * sp.shifts_per_day
    optimize_set = set(range(shift_lo, shift_hi))
    optimize = sorted(optimize_set)
    frozen = [t for t in range(sp.num_shifts) if t not in optimize_set]
    return frozen, optimize


def swap_shift_block(
    schedule: list[int],
    sp: SchedulingProblem,
    size_k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Re-optimize a contiguous block of size_k * shifts_per_day shifts.

    The start is sampled uniformly over `range(num_shifts - block_len + 1)`,
    NOT snapped to a day boundary. This distinguishes the neighborhood from
    swap_day at k=1 and probes between-day structure the day-aligned GA
    crossover can't reach.
    """
    block_len = size_k * sp.shifts_per_day
    start = int(rng.integers(0, sp.num_shifts - block_len + 1))
    end = start + block_len
    optimize = list(range(start, end))
    frozen = [t for t in range(sp.num_shifts) if t < start or t >= end]
    return frozen, optimize


def swap_employee(
    schedule: list[int],
    sp: SchedulingProblem,
    size_k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Re-optimize all shifts currently assigned to size_k random employees.

    `optimize` collects every shift t where schedule[t] is one of the chosen
    employees. The inner IP then redistributes those shifts among ALL
    employees (it's not constrained to keep them with the chosen employees).
    """
    size_k = min(size_k, sp.num_employees)
    chosen = rng.choice(sp.num_employees, size=size_k, replace=False).tolist()
    chosen_set = set(chosen)
    optimize = [t for t, e in enumerate(schedule) if e in chosen_set]
    optimize_set = set(optimize)
    frozen = [t for t in range(sp.num_shifts) if t not in optimize_set]
    return frozen, optimize
