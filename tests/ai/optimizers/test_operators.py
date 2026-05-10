"""Unit tests for the three CopyingOperator subclasses."""

import pytest
import torch

from ai.domain.problem import SchedulingProblem
from ai.optimizers.rostering_problem import RosteringProblem


@pytest.fixture
def problem(tiny_problem: SchedulingProblem) -> RosteringProblem:
    torch.manual_seed(42)
    return RosteringProblem(tiny_problem, device="cpu")


def test_crossover_preserves_dimensions(problem: RosteringProblem):
    from ai.optimizers.operators import DayAlignedCrossOver

    op = DayAlignedCrossOver(problem, tournament_size=2, cross_over_rate=1.0)
    parents = problem.generate_batch(8)
    problem.evaluate(parents)
    children = op._do(parents)

    assert children.values.shape == parents.values.shape
    assert children.values.dtype == torch.int64
    # Values stay within the problem bounds.
    assert children.values.min() >= 0
    assert children.values.max() <= problem._sp.num_employees - 1


def test_crossover_only_at_day_boundaries(problem: RosteringProblem):
    """When cross_over_rate=1, the cut point should be a multiple of shifts_per_day."""
    from ai.optimizers.operators import DayAlignedCrossOver

    op = DayAlignedCrossOver(problem, tournament_size=2, cross_over_rate=1.0)
    sp = problem._sp

    # Build two parents with distinguishable patterns.
    parents = problem.generate_batch(2)
    parents._data[0] = torch.zeros(sp.num_shifts, dtype=torch.int64)
    parents._data[1] = torch.ones(sp.num_shifts, dtype=torch.int64) + 1  # all 2s
    problem.evaluate(parents)

    torch.manual_seed(0)
    children = op._do(parents)
    # Find positions where child[0] differs from parent[0] (=0).
    diff_positions = torch.where(children.values[0] != 0)[0]
    if diff_positions.numel() > 0:
        first_diff = int(diff_positions.min())
        # The cut must be a day boundary.
        assert first_diff % sp.shifts_per_day == 0


def test_mutation_respects_indpb_distribution(problem: RosteringProblem):
    """With mut_rate=1 and indpb=0.5 over a large batch, ~50% of genes change."""
    from ai.optimizers.operators import UniformIntMutation

    torch.manual_seed(123)
    op = UniformIntMutation(problem, indpb=0.5, mut_rate=1.0)

    # All-zero starting batch.
    n = 200
    batch = problem.generate_batch(n)
    batch._data[:] = 0
    problem.evaluate(batch)

    children = op._do(batch)
    # Fraction of non-zero entries in children should be ≈ 0.5 × (1 - 1/num_employees).
    # Because uniform replacement might pick 0 again with probability 1/num_employees.
    sp = problem._sp
    expected_nonzero_fraction = 0.5 * (1 - 1 / sp.num_employees)
    actual_nonzero_fraction = float((children.values != 0).float().mean())
    assert abs(actual_nonzero_fraction - expected_nonzero_fraction) < 0.05


def test_repair_eliminates_unavailability_violations(default_problem: SchedulingProblem):
    """After repair, no cell has an unavailable employee on the corresponding day."""
    from ai.optimizers.operators import RepairOperator

    # Inject some unavailability on the default problem.
    sp = default_problem.model_copy(
        update={"unavailability": frozenset({(0, 0), (5, 1), (10, 2)})}
    )
    rp = RosteringProblem(sp, device="cpu")
    op = RepairOperator(rp)

    torch.manual_seed(7)
    batch = rp.generate_batch(8)
    rp.evaluate(batch)
    repaired = op._do(batch)

    # No cell should have an unavailable assignment.
    n, T = repaired.values.shape
    day_per_shift = torch.arange(T) // sp.shifts_per_day
    unavail_hits = rp._unavail_mask[
        day_per_shift.unsqueeze(0).expand(n, -1), repaired.values
    ]
    assert unavail_hits.sum() == 0


def test_repair_idempotent(default_problem: SchedulingProblem):
    """Calling repair twice produces no further changes."""
    from ai.optimizers.operators import RepairOperator

    sp = default_problem.model_copy(update={"unavailability": frozenset({(0, 0), (5, 1)})})
    rp = RosteringProblem(sp, device="cpu")
    op = RepairOperator(rp)

    torch.manual_seed(0)
    batch = rp.generate_batch(4)
    rp.evaluate(batch)
    repaired_once = op._do(batch)
    repaired_twice = op._do(repaired_once)

    assert torch.equal(repaired_once.values, repaired_twice.values)
