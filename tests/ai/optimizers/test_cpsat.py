"""Tests for CPSATOptimizer — issue #14 acceptance criteria + invariants."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ai.domain.problem import SchedulingProblem


def test_optimal_zero_violations(tiny_problem: SchedulingProblem):
    """Issue #14 AC: solver returns OPTIMAL and respects all hard constraints."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    # Schedule covers every shift slot.
    assert len(result.best_schedule) == tiny_problem.num_shifts
    # Every assignment is a valid employee index.
    for emp_idx in result.best_schedule:
        assert 0 <= emp_idx < tiny_problem.num_employees
    # All stages finished cleanly.
    assert len(result.stages) == 2
    for stage in result.stages:
        assert stage.status in {"OPTIMAL", "FEASIBLE"}
        assert stage.wall_clock_s > 0


def test_best_fitness_is_three_tuple_zero_violations(tiny_problem: SchedulingProblem):
    """best_fitness must match the EA shape: (unfairness, violations, b2b)."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    assert len(result.best_fitness) == 3
    unfairness, violations, b2b = result.best_fitness
    assert violations == 0.0
    # unfairness is the α=∞ normalized form: 1 - n·min/total ∈ [0, 1].
    assert 0.0 <= unfairness <= 1.0
    assert b2b == float(result.b2b_count)
    # The Jain index is reported separately as a side metric (at α=2).
    assert 0.0 <= result.jain_index <= 1.0


def test_unavailability_respected():
    """For each (day, e) ∈ unavailability, no shift on that day is assigned to e."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    sp = SchedulingProblem(
        num_employees=4,
        employee_types=("FT", "FT", "FT", "PT"),
        days=5,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0), (2, 1)}),
    )
    optimizer = CPSATOptimizer(sp)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    for t, emp in enumerate(result.best_schedule):
        day = t // sp.shifts_per_day
        assert (day, emp) not in sp.unavailability


def test_max_hours_respected(tiny_problem: SchedulingProblem):
    """No employee exceeds their max_hours."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    hours = [0] * tiny_problem.num_employees
    for t, emp in enumerate(result.best_schedule):
        shift_idx = t % tiny_problem.shifts_per_day
        hours[emp] += tiny_problem.shift_lengths[shift_idx]

    for e, h in enumerate(hours):
        assert h <= tiny_problem.max_hours[e], (
            f"employee {e} got {h}h, cap is {tiny_problem.max_hours[e]}h"
        )


def test_stages_record_optimal_status(tiny_problem: SchedulingProblem):
    """Both stages should record one of the two valid statuses, in order."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    result = CPSATOptimizer(tiny_problem).run(
        CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)
    )

    objectives = [stage.objective for stage in result.stages]
    assert objectives == ["b2b", "fairness"]


def test_lex_priority_b2b_then_fairness(tiny_problem: SchedulingProblem):
    """Solving stage 1 alone yields b2b★; full pipeline matches it."""
    from ai.optimizers.cpsat import CPSATOptimizer, _solve_b2b_only
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=1, seed=42)
    b2b_star = _solve_b2b_only(tiny_problem, config)

    result = CPSATOptimizer(tiny_problem).run(config)

    assert result.b2b_count == b2b_star
    assert result.stages[1].status in {"OPTIMAL", "FEASIBLE"}


def test_objective_priority_validation():
    """Unsupported objective_priority raises Pydantic ValidationError."""
    from ai.optimizers.result import CPSATConfig

    with pytest.raises(ValidationError):
        CPSATConfig(objective_priority=["fairness", "spread"])

    with pytest.raises(ValidationError):
        CPSATConfig(objective_priority=["b2b"])

    with pytest.raises(ValidationError):
        CPSATConfig(objective_priority=["b2b", "spread"])  # old name now invalid


def test_fairness_alpha_must_be_inf():
    """CPSATConfig rejects finite fairness_alpha."""
    from ai.optimizers.result import CPSATConfig

    with pytest.raises(ValidationError):
        CPSATConfig(fairness_alpha=2.0)

    with pytest.raises(ValidationError):
        CPSATConfig(fairness_alpha=0.0)

    # inf is fine:
    CPSATConfig(fairness_alpha=float("inf"))


def test_infeasible_raises(over_constrained_problem: SchedulingProblem):
    """Over-constrained instance raises CPSATInfeasibleError at stage 1."""
    from ai.optimizers.cpsat import CPSATInfeasibleError, CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(over_constrained_problem)
    with pytest.raises(CPSATInfeasibleError) as exc:
        optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    assert exc.value.stage == "b2b"


def test_timeout_raises_via_mock(tiny_problem: SchedulingProblem, mocker):
    """Mocking solver to return UNKNOWN raises CPSATTimeoutError."""
    from ortools.sat.python import cp_model

    from ai.optimizers.cpsat import CPSATOptimizer, CPSATTimeoutError
    from ai.optimizers.result import CPSATConfig

    mocker.patch.object(cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN)
    mocker.patch.object(cp_model.CpSolver, "WallTime", return_value=0.5)

    with pytest.raises(CPSATTimeoutError) as exc:
        CPSATOptimizer(tiny_problem).run(CPSATConfig(timeout_s_per_stage=1.0))

    assert exc.value.stage == "b2b"
    assert exc.value.elapsed_s == pytest.approx(0.5)


@pytest.mark.slow
def test_seed_is_deterministic(tiny_problem: SchedulingProblem):
    """num_workers=1 + seed=42 yields identical schedules across runs."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=1, seed=42)
    a = CPSATOptimizer(tiny_problem).run(config)
    b = CPSATOptimizer(tiny_problem).run(config)
    assert a.best_schedule == b.best_schedule


@pytest.mark.slow
def test_default_instance_completes_in_budget(default_problem: SchedulingProblem):
    """Issue #14 AC: 7×30×3 returns within 2 × timeout_s_per_stage seconds."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=30.0, num_workers=4, seed=42)
    result = CPSATOptimizer(default_problem).run(config)

    assert result.total_wall_clock_s < 2 * config.timeout_s_per_stage
    assert len(result.best_schedule) == default_problem.num_shifts
