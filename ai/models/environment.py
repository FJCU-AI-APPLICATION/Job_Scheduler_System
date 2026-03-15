from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvironmentConfig:
    """Configurable scheduling environment parameters."""

    num_employees: int = 7
    employee_types: list[str] = field(
        default_factory=lambda: ["FT", "FT", "FT", "FT", "PT", "PT", "PT"]
    )
    days: int = 30
    shifts_per_day: int = 3
    shift_lengths: list[int] = field(default_factory=lambda: [9, 8, 7])
    ft_max_hours: int = 160
    pt_max_hours: int = 40
    unavailability: set[tuple[int, int]] = field(default_factory=set)


class SchedulingEnv(gym.Env):
    """
    Gymnasium-compliant multi-shift-per-day scheduling environment.

    State = (shift_index, assigned_hours[0], ..., assigned_hours[num_employees-1])
    Action = which employee to assign to the current shift.

    Supports action masking via action_masks() for sb3-contrib MaskablePPO.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig | None = None):
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.num_employees = self.config.num_employees
        self.num_shifts = self.config.days * self.config.shifts_per_day
        self.shift_lengths = self.config.shift_lengths
        self.max_hours_for_employee = [
            self.config.ft_max_hours if t == "FT" else self.config.pt_max_hours
            for t in self.config.employee_types
        ]
        self.unavailability = self.config.unavailability

        max_possible_hours = max(self.config.ft_max_hours, self.config.pt_max_hours)

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array(
                [float(self.num_shifts)]
                + [float(max_possible_hours)] * self.num_employees,
                dtype=np.float32,
            ),
            shape=(self.num_employees + 1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_employees)

        self._state: tuple[int, ...] | None = None
        self._previous_employee: int | None = None
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = (0,) + tuple([0] * self.num_employees)
        self._previous_employee = None
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None
        shift_index = self._state[0]
        hours = list(self._state[1:])

        reward = 0.0

        # Penalty for consecutive shift with same employee
        if self._previous_employee is not None and action == self._previous_employee:
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
        self._previous_employee = action

        terminated = shift_index >= self.num_shifts
        truncated = False

        self._state = (shift_index,) + tuple(hours)
        return self._get_observation(), reward, terminated, truncated, {}

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for the current state.

        Used by sb3-contrib MaskablePPO/MaskableDQN.
        Masks out employees who are unavailable on the current day.
        """
        assert self._state is not None
        mask = np.ones(self.num_employees, dtype=np.bool_)

        shift_index = self._state[0]
        if shift_index < self.num_shifts:
            current_day = shift_index // self.config.shifts_per_day

            for emp_idx in range(self.num_employees):
                if (current_day, emp_idx) in self.unavailability:
                    mask[emp_idx] = False

            # Ensure at least one action is valid (fallback: unmask all)
            if not mask.any():
                mask[:] = True

        return mask

    def _get_observation(self) -> np.ndarray:
        assert self._state is not None
        return np.array(self._state, dtype=np.float32)
