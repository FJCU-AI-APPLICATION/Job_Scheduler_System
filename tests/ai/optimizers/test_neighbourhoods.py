"""Per-neighborhood invariants for matheuristic — issue #18."""

import numpy as np
import pytest

from ai.domain.problem import SchedulingProblem


@pytest.mark.parametrize("size_k", [1, 2, 3])
def test_swap_day_partition(tiny_problem: SchedulingProblem, size_k):
    """frozen ∪ optimize == range(num_shifts); disjoint."""
    from ai.optimizers.neighbourhoods import swap_day

    rng = np.random.default_rng(0)
    schedule = [0] * tiny_problem.num_shifts
    frozen, optimize = swap_day(schedule, tiny_problem, size_k=size_k, rng=rng)

    assert sorted(frozen + optimize) == list(range(tiny_problem.num_shifts))
    assert set(frozen).isdisjoint(set(optimize))
    assert len(optimize) > 0


@pytest.mark.parametrize("size_k", [1, 2, 3])
def test_swap_day_size(tiny_problem: SchedulingProblem, size_k):
    """optimize spans exactly size_k * shifts_per_day shifts."""
    from ai.optimizers.neighbourhoods import swap_day

    rng = np.random.default_rng(0)
    schedule = [0] * tiny_problem.num_shifts
    _, optimize = swap_day(schedule, tiny_problem, size_k=size_k, rng=rng)

    assert len(optimize) == size_k * tiny_problem.shifts_per_day


@pytest.mark.parametrize("size_k", [1, 2, 3])
def test_swap_shift_block_partition(tiny_problem: SchedulingProblem, size_k):
    """frozen ∪ optimize == range(num_shifts); disjoint."""
    from ai.optimizers.neighbourhoods import swap_shift_block

    rng = np.random.default_rng(0)
    schedule = [0] * tiny_problem.num_shifts
    frozen, optimize = swap_shift_block(schedule, tiny_problem, size_k=size_k, rng=rng)

    assert sorted(frozen + optimize) == list(range(tiny_problem.num_shifts))
    assert set(frozen).isdisjoint(set(optimize))
    assert len(optimize) > 0


def test_swap_shift_block_unaligned(tiny_problem: SchedulingProblem):
    """Over 100 seeds at k=1, the block's start is NOT always a day boundary —
    proves swap_shift_block distinguishes from swap_day."""
    from ai.optimizers.neighbourhoods import swap_shift_block

    starts = []
    for s in range(100):
        rng = np.random.default_rng(s)
        _, optimize = swap_shift_block(
            [0] * tiny_problem.num_shifts, tiny_problem, size_k=1, rng=rng
        )
        starts.append(optimize[0])

    unaligned = [s for s in starts if s % tiny_problem.shifts_per_day != 0]
    assert unaligned, (
        f"swap_shift_block@k=1 was always day-aligned over 100 seeds — "
        f"indistinguishable from swap_day"
    )


@pytest.mark.parametrize("size_k", [1, 2, 3])
def test_swap_employee_partition(tiny_problem: SchedulingProblem, size_k):
    """frozen ∪ optimize == range(num_shifts); disjoint."""
    from ai.optimizers.neighbourhoods import swap_employee

    rng = np.random.default_rng(0)
    # Construct a schedule that touches every employee at least once.
    schedule = [i % tiny_problem.num_employees for i in range(tiny_problem.num_shifts)]
    frozen, optimize = swap_employee(schedule, tiny_problem, size_k=size_k, rng=rng)

    assert sorted(frozen + optimize) == list(range(tiny_problem.num_shifts))
    assert set(frozen).isdisjoint(set(optimize))
    assert len(optimize) > 0


@pytest.mark.parametrize("size_k", [1, 2])
def test_swap_employee_size_employees(tiny_problem: SchedulingProblem, size_k):
    """`optimize` covers only shifts assigned to ≤ size_k distinct employees."""
    from ai.optimizers.neighbourhoods import swap_employee

    rng = np.random.default_rng(0)
    schedule = [i % tiny_problem.num_employees for i in range(tiny_problem.num_shifts)]
    _, optimize = swap_employee(schedule, tiny_problem, size_k=size_k, rng=rng)

    employees_in_optimize = {schedule[t] for t in optimize}
    assert len(employees_in_optimize) <= size_k


def test_neighborhoods_respect_seed(tiny_problem: SchedulingProblem):
    """Same rng seed → same (frozen, optimize) for every neighborhood."""
    from ai.optimizers.neighbourhoods import (
        swap_day,
        swap_employee,
        swap_shift_block,
    )

    schedule = [i % tiny_problem.num_employees for i in range(tiny_problem.num_shifts)]
    for nbh in (swap_day, swap_shift_block, swap_employee):
        a = nbh(schedule, tiny_problem, size_k=2, rng=np.random.default_rng(42))
        b = nbh(schedule, tiny_problem, size_k=2, rng=np.random.default_rng(42))
        assert a == b, f"{nbh.__name__} not seed-reproducible"


def test_neighborhoods_handle_k_max_equal_days(tiny_problem: SchedulingProblem):
    """k=days/k=num_employees is valid — entire schedule unfrozen."""
    from ai.optimizers.neighbourhoods import (
        swap_day,
        swap_employee,
        swap_shift_block,
    )

    rng = np.random.default_rng(0)
    schedule = [i % tiny_problem.num_employees for i in range(tiny_problem.num_shifts)]

    frozen, optimize = swap_day(schedule, tiny_problem, size_k=tiny_problem.days, rng=rng)
    assert len(optimize) == tiny_problem.num_shifts
    assert frozen == []

    frozen, optimize = swap_shift_block(
        schedule, tiny_problem, size_k=tiny_problem.days, rng=rng
    )
    assert len(optimize) == tiny_problem.num_shifts
    assert frozen == []

    frozen, optimize = swap_employee(
        schedule, tiny_problem, size_k=tiny_problem.num_employees, rng=rng
    )
    assert sorted(optimize) == list(range(tiny_problem.num_shifts))
    assert frozen == []
