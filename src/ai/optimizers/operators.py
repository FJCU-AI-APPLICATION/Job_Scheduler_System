"""Vectorized CopyingOperator subclasses for the rostering problem.

All three operators transform a SolutionBatch on the GPU/CPU device of the
underlying Problem. None contain Python-level per-individual loops.
"""

import torch
from evotorch import SolutionBatch
from evotorch.operators import CopyingOperator, CrossOver

from ai.optimizers.rostering_problem import RosteringProblem


class DayAlignedCrossOver(CrossOver):
    """Two-parent recombination cut at a day boundary.

    With probability `cross_over_rate`, the right half (from a randomly
    chosen day boundary) is swapped between the two parents.
    """

    def __init__(
        self,
        problem: RosteringProblem,
        *,
        tournament_size: int = 4,
        cross_over_rate: float = 0.7,
    ):
        super().__init__(problem, tournament_size=tournament_size, cross_over_rate=cross_over_rate)
        self._shifts_per_day = problem._sp.shifts_per_day
        self._num_shifts = problem._sp.num_shifts

    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        # parents1, parents2: (n_pairs, num_shifts) int64
        n_pairs = parents1.shape[0]
        d = parents1.device
        num_days = self._num_shifts // self._shifts_per_day

        # Sample a day boundary in [1, num_days) per pair.
        day_cuts = torch.randint(1, max(num_days, 2), (n_pairs,), device=d)
        shift_cuts = day_cuts * self._shifts_per_day  # (n_pairs,)

        # For each pair, swap the right half (>= shift_cut).
        idx = torch.arange(self._num_shifts, device=d).unsqueeze(0)  # (1, T)
        right_mask = idx >= shift_cuts.unsqueeze(1)  # (n_pairs, T)
        children1 = torch.where(right_mask, parents2, parents1)
        children2 = torch.where(right_mask, parents1, parents2)

        # Build a SolutionBatch from the offspring (n_pairs * 2 rows).
        out = self._make_children_batch(torch.cat([children1, children2], dim=0))
        return out


class UniformIntMutation(CopyingOperator):
    """Per-gene uniform integer mutation with both per-individual and per-gene gates."""

    def __init__(
        self,
        problem: RosteringProblem,
        *,
        indpb: float = 0.05,
        mut_rate: float = 0.2,
    ):
        super().__init__(problem)
        self._indpb = indpb
        self._mut_rate = mut_rate
        self._num_employees = problem._sp.num_employees

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = batch.clone()
        # Note: result.values returns a ReadOnlyTensor; use result._data for in-place mutation.
        # This pattern is used elsewhere in this codebase.
        values = result._data
        n = values.shape[0]
        ind_mask = (
            torch.rand(n, device=values.device) < self._mut_rate
        ).unsqueeze(1)
        gene_mask = torch.rand_like(values, dtype=torch.float32) < self._indpb
        new_values = torch.randint(
            0,
            self._num_employees,
            values.shape,
            device=values.device,
            dtype=values.dtype,
        )
        flip = ind_mask & gene_mask
        values[flip] = new_values[flip]
        return result


class RepairOperator(CopyingOperator):
    """Replace any cell where the assigned employee is unavailable with a random valid one."""

    def __init__(self, problem: RosteringProblem):
        super().__init__(problem)
        sp = problem._sp
        self._shifts_per_day = sp.shifts_per_day
        self._num_employees = sp.num_employees
        self._unavail_mask = problem._unavail_mask

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = batch.clone()
        values = result._data  # mutable; result.values is ReadOnly
        n, T = values.shape
        d = values.device

        day_per_shift = torch.arange(T, device=d) // self._shifts_per_day
        is_unavail = self._unavail_mask[
            day_per_shift.unsqueeze(0).expand(n, -1), values
        ]
        if not is_unavail.any():
            return result

        # availability[t, e] = True if employee e is available on day(t).
        availability = ~self._unavail_mask[day_per_shift]  # (T, num_employees)
        rand = torch.rand(n, T, self._num_employees, device=d)
        # Mask infeasible employees per shift.
        rand = rand * availability.unsqueeze(0).float() + (
            -1.0
        ) * (~availability).unsqueeze(0).float()
        replacement = rand.argmax(dim=-1)  # (N, T)

        values[is_unavail] = replacement[is_unavail]
        return result
