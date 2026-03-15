from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvironmentConfig:
    """Configurable scheduling environment parameters."""
    num_employees: int = 7
    employee_types: list[str] = field(default_factory=lambda: ["FT", "FT", "FT", "FT", "PT", "PT", "PT"])
    days: int = 30
    shifts_per_day: int = 3
    shift_lengths: list[int] = field(default_factory=lambda: [9, 8, 7])
    ft_max_hours: int = 160
    pt_max_hours: int = 40


class SchedulingEnv:
    """
    A multi-shift-per-day scheduling environment.

    State = (shift_index, assigned_hours[0], ..., assigned_hours[num_employees-1])
    Action = which employee to assign to the current shift.
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.num_employees = self.config.num_employees
        self.num_shifts = self.config.days * self.config.shifts_per_day
        self.shift_lengths = self.config.shift_lengths
        self.max_hours_for_employee = [
            self.config.ft_max_hours if t == "FT" else self.config.pt_max_hours
            for t in self.config.employee_types
        ]

        self.state: tuple[int, ...] | None = None
        self.previous_employee: int | None = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = (0,) + tuple([0] * self.num_employees)
        self.previous_employee = None
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.state is not None
        shift_index = self.state[0]
        hours = list(self.state[1:])

        reward = 0.0

        # Penalty for consecutive shift with same employee
        if self.previous_employee is not None and action == self.previous_employee:
            reward -= 2.0

        # Penalize large differences in assigned hours among employees
        gap = max(hours) - min(hours)
        reward -= 0.01 * gap

        # Pick shift length by shift_index % shifts_per_day
        shift_type = shift_index % len(self.shift_lengths)
        shift_length = self.shift_lengths[shift_type]

        # Assign hours
        hours[action] += shift_length

        # Penalty for exceeding max hours
        if hours[action] > self.max_hours_for_employee[action]:
            reward -= 5.0

        # Base reward for valid assignment
        reward += 1.0

        shift_index += 1
        self.previous_employee = action

        done = shift_index >= self.num_shifts

        self.state = (shift_index,) + tuple(hours)
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> np.ndarray:
        assert self.state is not None
        return np.array(self.state, dtype=float)
