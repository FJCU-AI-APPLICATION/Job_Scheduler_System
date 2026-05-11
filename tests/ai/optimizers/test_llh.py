"""Per-LLH invariant tests for LAST-RL — issue #19."""

import numpy as np
import pytest


@pytest.fixture
def tiny_schedule(tiny_problem):
    """A deterministic schedule on tiny_problem: 14 shifts, employees 0..2."""
    return [i % tiny_problem.num_employees for i in range(tiny_problem.num_shifts)]


def test_single_move_preserves_length_and_indices(tiny_problem, tiny_schedule):
    """single_move keeps length, returns valid indices, changes exactly one shift."""
    from ai.optimizers.llh import single_move

    for seed in range(100):
        rng = np.random.default_rng(seed)
        out = single_move(tiny_schedule, rng, num_employees=tiny_problem.num_employees)
        assert len(out) == len(tiny_schedule)
        assert all(0 <= e < tiny_problem.num_employees for e in out)
        diffs = [i for i in range(len(out)) if out[i] != tiny_schedule[i]]
        assert len(diffs) == 1, f"expected 1 differing shift, got {diffs}"


def test_random_swap_swaps_two_shifts(tiny_problem, tiny_schedule):
    """random_swap exchanges exactly two positions; others unchanged."""
    from ai.optimizers.llh import random_swap

    rng = np.random.default_rng(0)
    out = random_swap(tiny_schedule, rng)
    diffs = [i for i in range(len(out)) if out[i] != tiny_schedule[i]]
    assert len(diffs) in (0, 2)
    if len(diffs) == 2:
        a, b = diffs
        assert out[a] == tiny_schedule[b]
        assert out[b] == tiny_schedule[a]


def test_k_swap_rotates_k_positions(tiny_problem, tiny_schedule):
    """k_swap with k=3 cyclically rotates 3 positions; others unchanged."""
    from ai.optimizers.llh import k_swap

    rng = np.random.default_rng(0)
    out = k_swap(tiny_schedule, rng, k=3)
    diffs = [i for i in range(len(out)) if out[i] != tiny_schedule[i]]
    assert len(diffs) <= 3


def test_day_swap_swaps_full_days(tiny_problem, tiny_schedule):
    """day_swap exchanges two days' shifts; other days unchanged."""
    from ai.optimizers.llh import day_swap

    rng = np.random.default_rng(0)
    out = day_swap(tiny_schedule, rng, sp=tiny_problem)
    spd = tiny_problem.shifts_per_day
    differing_days = set()
    for t, (a, b) in enumerate(zip(out, tiny_schedule)):
        if a != b:
            differing_days.add(t // spd)
    assert len(differing_days) in (0, 2)


def test_employee_swap_relabels(tiny_problem, tiny_schedule):
    """employee_swap relabels two employees mutually; others unchanged."""
    from ai.optimizers.llh import employee_swap

    rng = np.random.default_rng(0)
    out = employee_swap(tiny_schedule, rng, sp=tiny_problem)
    mapping: dict[int, int] = {}
    for old, new in zip(tiny_schedule, out):
        if old in mapping:
            assert mapping[old] == new
        mapping[old] = new
    swapped = [k for k, v in mapping.items() if k != v]
    assert len(swapped) in (0, 2)
    if len(swapped) == 2:
        a, b = swapped
        assert mapping[a] == b
        assert mapping[b] == a


def test_repair_unavailability_fixes_violations():
    """repair_unavailability eliminates all unavail violations."""
    from ai.domain.problem import SchedulingProblem
    from ai.optimizers.llh import repair_unavailability

    sp = SchedulingProblem(
        num_employees=4, employee_types=("FT", "FT", "FT", "PT"),
        days=3, shifts_per_day=2, shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0), (1, 1)}),
    )
    sched = [0, 0, 1, 1, 0, 1]
    rng = np.random.default_rng(0)
    out = repair_unavailability(sched, rng, sp=sp)

    for t, emp in enumerate(out):
        day = t // sp.shifts_per_day
        assert (day, emp) not in sp.unavailability


def test_repair_unavailability_idempotent():
    """repair(repair(sched)) == repair(sched)."""
    from ai.domain.problem import SchedulingProblem
    from ai.optimizers.llh import repair_unavailability

    sp = SchedulingProblem(
        num_employees=4, employee_types=("FT", "FT", "FT", "PT"),
        days=3, shifts_per_day=2, shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0)}),
    )
    sched = [0, 1, 2, 0, 1, 2]
    out1 = repair_unavailability(sched, np.random.default_rng(0), sp=sp)
    out2 = repair_unavailability(out1, np.random.default_rng(1), sp=sp)
    assert out1 == out2


def test_repair_unavailability_noop_when_clean(tiny_problem, tiny_schedule):
    """If no violations, output == input (no-op)."""
    from ai.optimizers.llh import repair_unavailability

    rng = np.random.default_rng(0)
    out = repair_unavailability(tiny_schedule, rng, sp=tiny_problem)
    assert out == tiny_schedule


def test_swap_day_ip_returns_input_on_inner_ip_failure(tiny_problem, tiny_schedule, mocker):
    """When _inner_ip_solve returns None, swap_day_ip returns the input unchanged."""
    from ai.optimizers.llh import swap_day_ip

    mocker.patch("ai.optimizers.llh._inner_ip_solve", return_value=None)
    rng = np.random.default_rng(0)
    out = swap_day_ip(tiny_schedule, rng, sp=tiny_problem, time_budget_s=1.0, workers=1)
    assert out == tiny_schedule


def test_swap_shift_block_ip_returns_input_on_failure(tiny_problem, tiny_schedule, mocker):
    from ai.optimizers.llh import swap_shift_block_ip

    mocker.patch("ai.optimizers.llh._inner_ip_solve", return_value=None)
    rng = np.random.default_rng(0)
    out = swap_shift_block_ip(tiny_schedule, rng, sp=tiny_problem, time_budget_s=1.0, workers=1)
    assert out == tiny_schedule


def test_swap_employee_ip_returns_input_on_failure(tiny_problem, tiny_schedule, mocker):
    from ai.optimizers.llh import swap_employee_ip

    mocker.patch("ai.optimizers.llh._inner_ip_solve", return_value=None)
    rng = np.random.default_rng(0)
    out = swap_employee_ip(tiny_schedule, rng, sp=tiny_problem, time_budget_s=1.0, workers=1)
    assert out == tiny_schedule


def test_swap_day_ip_succeeds_on_real_solve(tiny_problem, tiny_schedule):
    """Real CP-SAT call returns a valid full schedule."""
    from ai.optimizers.llh import swap_day_ip

    rng = np.random.default_rng(0)
    out = swap_day_ip(tiny_schedule, rng, sp=tiny_problem, time_budget_s=10.0, workers=2)
    assert len(out) == len(tiny_schedule)
    assert all(0 <= e < tiny_problem.num_employees for e in out)


def test_build_llh_library_returns_9_named_heuristics(tiny_problem):
    """build_llh_library returns 9 LLHs with unique names matching the spec."""
    from ai.optimizers.llh import build_llh_library

    lib = build_llh_library(tiny_problem, ip_time_budget_s=1.0, ip_workers=1)
    names = [h.name for h in lib]
    assert names == [
        "single_move",
        "random_swap",
        "k_swap_3",
        "day_swap",
        "employee_swap",
        "repair_unavail",
        "swap_day_ip",
        "swap_block_ip",
        "swap_emp_ip",
    ]
    assert len(set(names)) == 9


def test_all_pure_llhs_seed_reproducible(tiny_problem, tiny_schedule):
    """Same rng seed → bit-identical output for all 6 pure LLHs."""
    from ai.optimizers.llh import build_llh_library

    lib = build_llh_library(tiny_problem, ip_time_budget_s=1.0, ip_workers=1)
    pure = lib[:6]
    for llh in pure:
        a = llh(tiny_schedule, np.random.default_rng(7))
        b = llh(tiny_schedule, np.random.default_rng(7))
        assert a == b, f"{llh.name} not seed-reproducible"
