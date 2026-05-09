"""Core scheduling problem representation and shared utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from ai.domain.schemas import SchedulingRequest, ShiftAssignment

if TYPE_CHECKING:
    from ai.agents.environment import EnvironmentConfig


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jain_fairness_index(values: torch.Tensor | np.ndarray | list) -> float:
    """Compute Jain's fairness index: (sum(x))^2 / (n * sum(x^2)).

    Returns 1.0 for perfect equality, 1/n for maximum inequality.
    Returns 1.0 if all values are zero or empty.
    """
    if not isinstance(values, torch.Tensor):
        t = torch.as_tensor(values, dtype=torch.float64, device=get_device())
    else:
        t = values
    n = t.shape[0]
    if n == 0:
        return 1.0
    sum_sq = t.pow(2).sum()
    if sum_sq == 0:
        return 1.0
    return float(t.sum().pow(2) / (n * sum_sq))


class SchedulingProblem(BaseModel):
    """Encapsulates a scheduling problem instance.

    Single source of truth for problem parameters, derived from either
    an EnvironmentConfig (for training) or a SchedulingRequest (for inference).
    """

    model_config = ConfigDict(frozen=True)

    num_employees: int
    employee_types: tuple[str, ...]
    days: int
    shifts_per_day: int
    shift_lengths: tuple[int, ...]
    max_hours: tuple[int, ...]
    unavailability: frozenset[tuple[int, int]] = frozenset()

    @property
    def num_shifts(self) -> int:
        return self.days * self.shifts_per_day

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> SchedulingProblem:
        max_hours = tuple(
            config.ft_max_hours if t == "FT" else config.pt_max_hours
            for t in config.employee_types
        )
        return cls(
            num_employees=config.num_employees,
            employee_types=tuple(config.employee_types),
            days=config.days,
            shifts_per_day=config.shifts_per_day,
            shift_lengths=tuple(config.shift_lengths),
            max_hours=max_hours,
            unavailability=frozenset(config.unavailability),
        )

    @classmethod
    def from_request(cls, request: SchedulingRequest) -> SchedulingProblem:
        """Build a SchedulingProblem from an API request.

        Maps external employee IDs to internal indices (0..n-1).
        """
        id_to_idx = {e.id: i for i, e in enumerate(request.employees)}
        unavail = frozenset(
            (u.day, id_to_idx[u.employee_id])
            for u in request.unavailability
            if u.employee_id in id_to_idx
        )

        return cls(
            num_employees=len(request.employees),
            employee_types=tuple(e.employee_type for e in request.employees),
            days=request.days,
            shifts_per_day=len(request.shifts),
            shift_lengths=tuple(s.length_hours for s in request.shifts),
            max_hours=tuple(e.max_hours for e in request.employees),
            unavailability=unavail,
        )

    def compute_hours(self, schedule: list[int] | torch.Tensor) -> torch.Tensor:
        """Compute total hours assigned per employee from a flat schedule."""
        device = get_device()
        shift_lens = torch.tensor(self.shift_lengths, dtype=torch.float64, device=device)
        sched = torch.as_tensor(schedule, dtype=torch.long, device=device)
        shift_types = torch.arange(sched.shape[0], device=device) % self.shifts_per_day
        hours = torch.zeros(self.num_employees, dtype=torch.float64, device=device)
        hours.scatter_add_(0, sched, shift_lens[shift_types])
        return hours

    def count_back_to_back(self, schedule: list[int] | torch.Tensor) -> int:
        """Count consecutive shifts assigned to the same employee."""
        sched = torch.as_tensor(schedule, dtype=torch.long, device=get_device())
        return int((sched[:-1] == sched[1:]).sum())


class ScheduleConverter:
    """Converts internal schedule representations to API response objects."""

    def __init__(self, problem: SchedulingProblem, request: SchedulingRequest):
        self._problem = problem
        self._request = request

    def to_assignments(
        self, schedule: list[int]
    ) -> tuple[list[ShiftAssignment], dict[int, int]]:
        """Convert a flat schedule to assignments and per-employee hours.

        Returns:
            assignments: List of ShiftAssignment with external employee IDs.
            hours_by_employee: Dict mapping external employee ID to total hours.
        """
        assignments: list[ShiftAssignment] = []
        hours_by_employee: dict[int, int] = {e.id: 0 for e in self._request.employees}

        for i, emp_idx in enumerate(schedule):
            day = i // self._problem.shifts_per_day
            shift_idx = i % self._problem.shifts_per_day
            emp = self._request.employees[int(emp_idx)]
            assignments.append(
                ShiftAssignment(day=day, shift_index=shift_idx, employee_id=emp.id)
            )
            hours_by_employee[emp.id] += self._problem.shift_lengths[shift_idx]

        return assignments, hours_by_employee
