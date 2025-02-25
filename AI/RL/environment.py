# environment.py

import numpy as np
from typing import Tuple

# Example: we have employees with different max hours (FT=160, PT=40)
EMPLOYEE_TYPES = ["FT", "FT", "FT","FT","PT", "PT", "PT"]
NUM_EMPLOYEES = len(EMPLOYEE_TYPES)

# 30 days, 3 shifts/day => 90 total shift assignments
DAYS = 30
SHIFTS_PER_DAY = 3
NUM_SHIFTS = DAYS * SHIFTS_PER_DAY
SHIFT_LENGTHS = [9, 8, 7]  # midnight=9h, day=8h, afternoon=7h

def get_employee_max_hours(emp_type: str) -> int:
    return 160 if emp_type == "FT" else 40

class SchedulingEnv:
    """
    A 3-shift-per-day scheduling environment.
    State = (shift_index, assigned_hours[0], ..., assigned_hours[NUM_EMPLOYEES-1])
    Action = which employee to assign to the current shift.
    """
    def __init__(self):
        self.num_employees = NUM_EMPLOYEES
        self.num_shifts = NUM_SHIFTS
        self.shift_lengths = SHIFT_LENGTHS
        self.max_hours_for_employee = [get_employee_max_hours(t) for t in EMPLOYEE_TYPES]

        self.state = None
        self.previous_employee = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = (0,) + tuple([0]*self.num_employees)
        self.previous_employee = None
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        shift_index = self.state[0]
        hours = list(self.state[1:])

        reward = 0.0
        done = False

        # # penalty if consecutive shift with same employee
        # if self.previous_employee is not None and action == self.previous_employee:
        #     reward -= 2.0

         # We penalize large differences in assigned hours among employees.
        gap = max(hours) - min(hours)
        reward -= 0.01 * gap

        # pick shift length by shift_index % 3
        shift_type = shift_index % 3
        shift_length = self.shift_lengths[shift_type]

        # assign hours
        hours[action] += shift_length

        # penalty if exceeding max hours for that employee
        # if hours[action] > self.max_hours_for_employee[action]:
        #     reward -= 5.0

        # small positive reward for a valid assignment
        reward += 1.0

        shift_index += 1
        self.previous_employee = action

        done =  shift_index >= self.num_shifts

        self.state = (shift_index,) + tuple(hours)
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> np.ndarray:
        return np.array(self.state, dtype=float)
