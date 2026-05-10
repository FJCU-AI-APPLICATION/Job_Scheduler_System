"""EvoTorch Problem adapter for shift scheduling.

Decision space: integer vector of length num_shifts, each gene in
[0, num_employees-1] = which employee is assigned to that shift.

Objectives (all minimized):
  0: imbalance       = 1 - Jain's fairness index
  1: violations      = max-hours overrun + 10 * unavailability hits
  2: back_to_back    = count of consecutive same-employee shifts
"""

import torch
from evotorch import Problem, SolutionBatch

from ai.domain.problem import SchedulingProblem


class RosteringProblem(Problem):
    def __init__(
        self,
        scheduling_problem: SchedulingProblem,
        device: torch.device | str = "cpu",
    ):
        self._sp = scheduling_problem
        super().__init__(
            objective_sense=["min", "min", "min"],
            solution_length=scheduling_problem.num_shifts,
            initial_bounds=(0, scheduling_problem.num_employees - 1),
            bounds=(0, scheduling_problem.num_employees - 1),
            dtype=torch.int64,
            device=device,
        )
        self._precompute_tensors()

    def _precompute_tensors(self) -> None:
        sp = self._sp
        d = self.device
        self._shift_lens = torch.tensor(
            sp.shift_lengths, dtype=torch.float64, device=d
        )
        self._shift_types = torch.arange(sp.num_shifts, device=d) % sp.shifts_per_day
        self._shift_hour_values = self._shift_lens[self._shift_types]
        self._max_hours_t = torch.tensor(sp.max_hours, dtype=torch.float64, device=d)

        unavail_mask = torch.zeros(
            sp.days, sp.num_employees, dtype=torch.bool, device=d
        )
        for day, emp in sp.unavailability:
            unavail_mask[day, emp] = True
        self._unavail_mask = unavail_mask

    def _evaluate_batch(self, solutions: SolutionBatch) -> None:
        pop = solutions.values.to(torch.long)
        n = pop.shape[0]
        sp = self._sp

        lens = self._shift_hour_values.unsqueeze(0).expand(n, -1)
        hours = torch.zeros(
            n, sp.num_employees, dtype=torch.float64, device=pop.device
        )
        hours.scatter_add_(1, pop, lens)

        sum_h = hours.sum(dim=1)
        sum_sq = hours.pow(2).sum(dim=1)
        jain = torch.where(
            sum_sq > 0,
            sum_h.pow(2) / (sp.num_employees * sum_sq),
            torch.ones(n, dtype=torch.float64, device=pop.device),
        )
        imbalance = 1.0 - jain

        exceed = (hours - self._max_hours_t).clamp(min=0).sum(dim=1)
        day_per_shift = (
            torch.arange(sp.num_shifts, device=pop.device) // sp.shifts_per_day
        )
        violations_per_cell = self._unavail_mask[
            day_per_shift.unsqueeze(0).expand(n, -1), pop
        ]
        unavail_count = violations_per_cell.sum(dim=1).to(torch.float64)
        violations = exceed + unavail_count * 10.0

        b2b = (pop[:, :-1] == pop[:, 1:]).sum(dim=1).to(torch.float64)

        fitnesses = torch.stack([imbalance, violations, b2b], dim=1)
        solutions.set_evals(fitnesses.to(torch.float32))
