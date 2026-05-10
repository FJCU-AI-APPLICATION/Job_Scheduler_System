"""Numerical tests for RosteringProblem._evaluate_batch."""

import pytest
import torch

from ai.domain.problem import SchedulingProblem


def test_evaluate_batch_shape(tiny_problem: SchedulingProblem):
    """SolutionBatch evals tensor has shape (popsize, 3)."""
    from ai.optimizers.rostering_problem import RosteringProblem

    problem = RosteringProblem(tiny_problem, device="cpu")
    pop = problem.generate_batch(8)
    problem.evaluate(pop)

    assert pop.evals.shape == (8, 3)
    assert not torch.isnan(pop.evals).any()


def test_violations_count_correct(tiny_problem: SchedulingProblem):
    """Objective 1 (violations) = max-hours-overrun + 10 × unavailability hits."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # All shifts assigned to employee 0 — they go way over their 50h cap.
    schedule = [0] * tiny_problem.num_shifts
    new_problem = RosteringProblem(tiny_problem, device="cpu")
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    # Total assigned hours to employee 0 = num_shifts × shift_length.
    expected_overrun = tiny_problem.num_shifts * 8 - 50
    assert float(pop.evals[0, 1]) == pytest.approx(expected_overrun, rel=1e-5)


def test_b2b_count_correct(tiny_problem: SchedulingProblem):
    """Objective 2 (back-to-back) = count of consecutive same-employee shifts."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # All same employee → every adjacent pair is b2b → count = num_shifts - 1
    schedule = [0] * tiny_problem.num_shifts
    new_problem = RosteringProblem(tiny_problem, device="cpu")
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    assert int(pop.evals[0, 2]) == tiny_problem.num_shifts - 1


def test_unavailability_handling(tiny_problem: SchedulingProblem):
    """When an employee is unavailable but assigned, violations += 10 per hit."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # Inject an unavailability: employee 0 unavailable on day 0.
    sp = tiny_problem.model_copy(update={"unavailability": frozenset({(0, 0)})})
    new_problem = RosteringProblem(sp, device="cpu")
    # Schedule with employee 0 on day 0 (both shifts) — 2 hits.
    schedule = [0, 0] + [1] * (sp.num_shifts - 2)
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    # Violations include 10 × 2 = 20 from unavail, plus any max-hours overrun.
    # Employee 1 assigned 12 shifts × 8h = 96h > 50h cap → overrun 46h.
    # Total expected: 20 + 46 = 66.
    assert float(pop.evals[0, 1]) == pytest.approx(66.0, rel=1e-5)
