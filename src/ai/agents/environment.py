import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel

from ai.domain.fairness import aggregate_fairness


class EnvironmentConfig(BaseModel):
    """Configurable scheduling environment parameters."""

    num_employees: int = 7
    employee_types: list[str] = ["FT", "FT", "FT", "FT", "PT", "PT", "PT"]
    days: int = 30
    shifts_per_day: int = 3
    shift_lengths: list[int] = [9, 8, 7]
    ft_max_hours: int = 160
    pt_max_hours: int = 40
    unavailability: set[tuple[int, int]] = set()
    fairness_alpha: float = 2.0
    pareto_reference: list[tuple[float, float, float]] | None = None
    hv_reference_point: tuple[float, float, float] = (2.0, 1000.0, 100.0)


class SchedulingEnv(gym.Env):
    """Gymnasium-compliant multi-shift-per-day scheduling environment.

    Rich state representation (~30 features):
      - Normalized progress (shift_index / num_shifts)
      - Current day normalized (day / num_days)
      - Shift type one-hot (shifts_per_day dims)
      - Day-of-week one-hot (7 dims)
      - Normalized hours assigned per employee (num_employees dims)
      - Hours remaining to max per employee (num_employees dims)
      - Availability mask per employee (num_employees dims)

    Action = which employee to assign to the current shift (Discrete).
    Supports action masking via action_masks() for sb3-contrib MaskablePPO.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig | None = None):
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.num_employees = self.config.num_employees
        self.num_shifts = self.config.days * self.config.shifts_per_day
        self.shift_lengths = self.config.shift_lengths
        self.max_hours_for_employee = np.array(
            [
                self.config.ft_max_hours if t == "FT" else self.config.pt_max_hours
                for t in self.config.employee_types
            ],
            dtype=np.float32,
        )
        self.unavailability = self.config.unavailability

        self._obs_dim = (
            1
            + 1
            + self.config.shifts_per_day
            + 7
            + self.num_employees
            + self.num_employees
            + self.num_employees
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_employees)

        self._shift_index: int = 0
        self._hours: np.ndarray = np.zeros(self.num_employees, dtype=np.float32)
        self._previous_employee: int | None = None
        self._schedule: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._shift_index = 0
        self._hours = np.zeros(self.num_employees, dtype=np.float32)
        self._previous_employee = None
        self._schedule = []
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        shift_index = self._shift_index
        current_day = shift_index // self.config.shifts_per_day

        reward = 0.0

        if self._previous_employee is not None and action == self._previous_employee:
            reward -= 2.0

        if self._hours.sum() > 0:
            # Penalty is computed against the running total at each step.
            # At α=2 it's scale-invariant and matches 1 - jain exactly. At α=1
            # it can spike to >>1 on early states with few employees served —
            # intentional: α=1 incentivizes everyone-gets-something quickly.
            penalty = aggregate_fairness(
                self._hours, alpha=self.config.fairness_alpha, kind="unfairness"
            )
            reward -= 0.1 * penalty

        shift_type = shift_index % len(self.shift_lengths)
        shift_length = self.shift_lengths[shift_type]

        self._hours[action] += shift_length

        if self._hours[action] > self.max_hours_for_employee[action]:
            reward -= 10.0

        if (current_day, action) in self.unavailability:
            reward -= 10.0

        reward += 0.5

        self._shift_index += 1
        self._previous_employee = action
        self._schedule.append(int(action))

        terminated = self._shift_index >= self.num_shifts
        truncated = False

        info: dict = {}
        if terminated and self.config.pareto_reference is not None:
            delta_hv = self._compute_delta_hv()
            reward += delta_hv
            fit = self._episode_fitness()
            info = {
                "delta_hv": delta_hv,
                "episode_unfairness": float(fit[0]),
                "episode_violations": float(fit[1]),
                "episode_b2b": float(fit[2]),
            }

        return self._get_observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for the current state.

        Used by sb3-contrib MaskablePPO/MaskableDQN.
        Masks out employees who are unavailable on the current day.
        """
        mask = np.ones(self.num_employees, dtype=np.bool_)

        if self._shift_index < self.num_shifts:
            current_day = self._shift_index // self.config.shifts_per_day

            for emp_idx in range(self.num_employees):
                if (current_day, emp_idx) in self.unavailability:
                    mask[emp_idx] = False

            if not mask.any():
                mask[:] = True

        return mask

    def _compute_delta_hv(self) -> float:
        """Marginal HV contribution of the episode's final point against the
        reference front. Returns 0.0 if pareto_reference is empty/None."""
        if not self.config.pareto_reference:
            return 0.0
        from pymoo.indicators.hv import HV

        point = self._episode_fitness()
        ref = np.array(self.config.hv_reference_point, dtype=np.float64)
        front = np.array(self.config.pareto_reference, dtype=np.float64)
        hv = HV(ref_point=ref)
        return max(
            float(hv(np.vstack([front, point]))) - float(hv(front)),
            0.0,
        )

    def _episode_fitness(self) -> np.ndarray:
        """3-tuple (unfairness, violations, b2b) matching RosteringProblem._evaluate_batch."""
        unfairness = aggregate_fairness(
            self._hours, alpha=self.config.fairness_alpha, kind="unfairness"
        )
        exceed = float(
            np.clip(self._hours - self.max_hours_for_employee, 0.0, None).sum()
        )
        unavail_hits = sum(
            1
            for t, emp in enumerate(self._schedule)
            if (t // self.config.shifts_per_day, emp) in self.unavailability
        )
        violations = exceed + 10.0 * unavail_hits
        b2b = sum(
            1
            for i in range(1, len(self._schedule))
            if self._schedule[i] == self._schedule[i - 1]
        )
        return np.array([unfairness, violations, float(b2b)], dtype=np.float64)

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        idx = 0

        obs[idx] = self._shift_index / max(self.num_shifts, 1)
        idx += 1

        current_day = (
            self._shift_index // self.config.shifts_per_day
            if self._shift_index < self.num_shifts
            else self.config.days - 1
        )
        obs[idx] = current_day / max(self.config.days - 1, 1)
        idx += 1

        if self._shift_index < self.num_shifts:
            shift_type = self._shift_index % self.config.shifts_per_day
            obs[idx + shift_type] = 1.0
        idx += self.config.shifts_per_day

        dow = current_day % 7
        obs[idx + dow] = 1.0
        idx += 7

        max_h = self.max_hours_for_employee.copy()
        max_h[max_h == 0] = 1.0
        obs[idx : idx + self.num_employees] = self._hours / max_h
        idx += self.num_employees

        remaining = np.maximum(self.max_hours_for_employee - self._hours, 0.0)
        obs[idx : idx + self.num_employees] = remaining / max_h
        idx += self.num_employees

        if self._shift_index < self.num_shifts:
            for emp_idx in range(self.num_employees):
                obs[idx + emp_idx] = (
                    0.0 if (current_day, emp_idx) in self.unavailability else 1.0
                )
        else:
            obs[idx : idx + self.num_employees] = 1.0
        idx += self.num_employees

        return obs
