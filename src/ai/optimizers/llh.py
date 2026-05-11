"""Low-level heuristics (LLHs) for LAST-RL — issue #19.

Each LLH is a callable `(schedule, rng, **bound) -> new_schedule`. Returns a
NEW list, never mutates the input. Pure mutations (6) take sub-millisecond;
IP-backed slice operators (3, added in Task 5) take ~0.1-1s via CP-SAT.

LowLevelHeuristic Protocol requires a `name: str` attribute. Use
`partial_with_name(name, fn, **bound_kwargs)` (defined below) to wrap a
function into a Protocol-compatible callable.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

from ai.domain.problem import SchedulingProblem


class LowLevelHeuristic(Protocol):
    """A schedule-mutating operator the RL policy can choose to apply."""

    name: str

    def __call__(
        self,
        solution: list[int],
        rng: np.random.Generator,
    ) -> list[int]:
        ...


class _NamedLLH:
    """Callable adapter binding kwargs at problem-construction time.

    functools.partial doesn't carry a `name` attribute that the
    LowLevelHeuristic Protocol requires; this thin wrapper does.
    """

    def __init__(self, name: str, fn: Callable, **bound_kwargs):
        self.name = name
        self._fn = fn
        self._bound = bound_kwargs

    def __call__(
        self, solution: list[int], rng: np.random.Generator
    ) -> list[int]:
        return self._fn(solution, rng, **self._bound)


def partial_with_name(
    name: str, fn: Callable, **bound_kwargs: Any
) -> LowLevelHeuristic:
    return _NamedLLH(name, fn, **bound_kwargs)


# === Pure mutations ===


def single_move(
    sched: list[int],
    rng: np.random.Generator,
    *,
    num_employees: int,
) -> list[int]:
    """Pick a random shift; reassign to a uniformly-random different employee."""
    new = list(sched)
    t = int(rng.integers(0, len(new)))
    candidates = [e for e in range(num_employees) if e != new[t]]
    new[t] = int(rng.choice(candidates))
    return new


def random_swap(
    sched: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Swap the assignees of two random shifts."""
    new = list(sched)
    a, b = rng.choice(len(new), size=2, replace=False)
    new[a], new[b] = new[b], new[a]
    return new


def k_swap(
    sched: list[int],
    rng: np.random.Generator,
    *,
    k: int = 3,
) -> list[int]:
    """Cyclic permutation across k random shifts."""
    new = list(sched)
    if len(new) < k:
        return new
    idx = rng.choice(len(new), size=k, replace=False)
    values = [new[i] for i in idx]
    rotated = values[-1:] + values[:-1]
    for i, v in zip(idx, rotated):
        new[i] = v
    return new


def day_swap(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
) -> list[int]:
    """Swap the schedules of two random days entirely."""
    new = list(sched)
    if sp.days < 2:
        return new
    d1, d2 = rng.choice(sp.days, size=2, replace=False)
    s = sp.shifts_per_day
    a_start, b_start = int(d1) * s, int(d2) * s
    for k in range(s):
        new[a_start + k], new[b_start + k] = (
            new[b_start + k],
            new[a_start + k],
        )
    return new


def employee_swap(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
) -> list[int]:
    """Pick 2 random employees; relabel all their shifts mutually."""
    new = list(sched)
    if sp.num_employees < 2:
        return new
    e1, e2 = rng.choice(sp.num_employees, size=2, replace=False)
    for t in range(len(new)):
        if new[t] == e1:
            new[t] = int(e2)
        elif new[t] == e2:
            new[t] = int(e1)
    return new


def repair_unavailability(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
) -> list[int]:
    """For every shift where the assignee is unavailable on that day,
    reassign to a uniformly-random available employee. Idempotent."""
    new = list(sched)
    unavail_by_day: dict[int, set[int]] = {}
    for day, emp in sp.unavailability:
        unavail_by_day.setdefault(day, set()).add(emp)
    for t in range(len(new)):
        day = t // sp.shifts_per_day
        bad = unavail_by_day.get(day, set())
        if new[t] in bad:
            avail = [e for e in range(sp.num_employees) if e not in bad]
            if avail:
                new[t] = int(rng.choice(avail))
    return new


# === IP-backed slice operators ===
# Reuses matheuristic's _inner_ip_solve (#18). Failures (timeout/infeasibility)
# return the INPUT schedule unchanged — NOT None — because the outer loop is
# AM-acceptance and would propagate None into cost(). The "effective no-op"
# semantics are correct for AM: the RL just learns that this LLH was useless
# this step.

from ai.optimizers.matheuristic import _inner_ip_solve
from ai.optimizers.neighbourhoods import (
    swap_day as _nbh_swap_day,
    swap_employee as _nbh_swap_employee,
    swap_shift_block as _nbh_swap_shift_block,
)


def swap_day_ip(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
    time_budget_s: float,
    workers: int,
) -> list[int]:
    """Pick 1 random day; CP-SAT-resolve those shifts (lex b2b then fairness)."""
    frozen, optimize = _nbh_swap_day(sched, sp, size_k=1, rng=rng)
    result = _inner_ip_solve(
        sp=sp, incumbent=sched, frozen=frozen, optimize=optimize,
        time_budget_s=time_budget_s, workers=workers,
    )
    return result if result is not None else list(sched)


def swap_shift_block_ip(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
    time_budget_s: float,
    workers: int,
) -> list[int]:
    """Pick 1 random unaligned block (size = shifts_per_day); CP-SAT-resolve."""
    frozen, optimize = _nbh_swap_shift_block(sched, sp, size_k=1, rng=rng)
    result = _inner_ip_solve(
        sp=sp, incumbent=sched, frozen=frozen, optimize=optimize,
        time_budget_s=time_budget_s, workers=workers,
    )
    return result if result is not None else list(sched)


def swap_employee_ip(
    sched: list[int],
    rng: np.random.Generator,
    *,
    sp: SchedulingProblem,
    time_budget_s: float,
    workers: int,
) -> list[int]:
    """Pick 1 random employee's shifts; CP-SAT-resolve to redistribute."""
    frozen, optimize = _nbh_swap_employee(sched, sp, size_k=1, rng=rng)
    result = _inner_ip_solve(
        sp=sp, incumbent=sched, frozen=frozen, optimize=optimize,
        time_budget_s=time_budget_s, workers=workers,
    )
    return result if result is not None else list(sched)


# === Library builder ===


def build_llh_library(
    sp: SchedulingProblem,
    *,
    ip_time_budget_s: float = 2.0,
    ip_workers: int = 2,
) -> list[LowLevelHeuristic]:
    """The 9-LLH library used by RosteringLastRLProblem.

    Order matters — it's part of the action-index encoding stored in
    checkpoints. Do not reorder without bumping a checkpoint version.
    """
    return [
        partial_with_name("single_move",    single_move,            num_employees=sp.num_employees),
        partial_with_name("random_swap",    random_swap),
        partial_with_name("k_swap_3",       k_swap,                 k=3),
        partial_with_name("day_swap",       day_swap,               sp=sp),
        partial_with_name("employee_swap",  employee_swap,          sp=sp),
        partial_with_name("repair_unavail", repair_unavailability,  sp=sp),
        partial_with_name("swap_day_ip",    swap_day_ip,            sp=sp, time_budget_s=ip_time_budget_s, workers=ip_workers),
        partial_with_name("swap_block_ip",  swap_shift_block_ip,    sp=sp, time_budget_s=ip_time_budget_s, workers=ip_workers),
        partial_with_name("swap_emp_ip",    swap_employee_ip,       sp=sp, time_budget_s=ip_time_budget_s, workers=ip_workers),
    ]


# === Paper LLH library (6 LLHs, no IP) — sanity-check use only ===


def build_paper_llh_library(instance) -> list[LowLevelHeuristic]:
    """6 LLHs matching the LAST-RL paper's set on BCV-format instances.

    Differs from build_llh_library (our SchedulingProblem) in three ways:
      1. No IP-backed operators — paper is pure mutation
      2. `repair_unavailability` is replaced by `repair_coverage` (paper-specific)
      3. Operators use BCV's flat `num_days * num_shift_types` schedule shape

    The 6 LLHs are: 1-move, 2-swap, 3-swap, day-move, sequence-swap,
    repair-coverage.
    """
    nE = instance.num_employees

    def _bcv_single_move(sched, rng):
        return single_move(sched, rng, num_employees=nE)

    def _bcv_k_swap_2(sched, rng):
        return k_swap(sched, rng, k=2)

    def _bcv_k_swap_3(sched, rng):
        return k_swap(sched, rng, k=3)

    def _bcv_day_move(sched, rng):
        """Move a random day's worth of shifts to another day's slot."""
        new = list(sched)
        d1, d2 = rng.choice(instance.num_days, size=2, replace=False)
        s = instance.num_shift_types
        for k in range(s):
            new[int(d1) * s + k] = new[int(d2) * s + k]
        return new

    def _bcv_sequence_swap(sched, rng):
        """Pick a random consecutive 2-shift block; swap the assignments."""
        new = list(sched)
        if len(new) < 2:
            return new
        i = int(rng.integers(0, len(new) - 1))
        new[i], new[i + 1] = new[i + 1], new[i]
        return new

    def _bcv_repair_coverage(sched, rng):
        """For each (day, shift_type), if assigned employee is on days_off,
        reassign to a uniformly-random available employee."""
        new = list(sched)
        for t in range(len(new)):
            day = t // instance.num_shift_types
            if day in instance.days_off.get(new[t], set()):
                avail = [
                    e for e in range(nE)
                    if day not in instance.days_off.get(e, set())
                ]
                if avail:
                    new[t] = int(rng.choice(avail))
        return new

    return [
        partial_with_name("paper_1_move",        _bcv_single_move),
        partial_with_name("paper_2_swap",        _bcv_k_swap_2),
        partial_with_name("paper_3_swap",        _bcv_k_swap_3),
        partial_with_name("paper_day_move",      _bcv_day_move),
        partial_with_name("paper_seq_swap",      _bcv_sequence_swap),
        partial_with_name("paper_repair_cov",    _bcv_repair_coverage),
    ]
