import numpy as np
import torch

from models.dqn import DQN
from models.environment import EnvironmentConfig, SchedulingEnv
from server.schemas import (
    ScheduleMetrics,
    SchedulingRequest,
    SchedulingResponse,
    ShiftAssignment,
)
from server.services.model_registry import get_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_rl_inference(
    request: SchedulingRequest,
    checkpoint: str = "model.pth",
) -> SchedulingResponse:
    """Run DQN inference to generate an optimized schedule."""
    num_employees = len(request.employees)
    shifts_per_day = len(request.shifts)
    shift_lengths = [s.length_hours for s in request.shifts]
    employee_types = [e.employee_type for e in request.employees]

    config = EnvironmentConfig(
        num_employees=num_employees,
        employee_types=employee_types,
        days=request.days,
        shifts_per_day=shifts_per_day,
        shift_lengths=shift_lengths,
        ft_max_hours=max((e.max_hours for e in request.employees if e.employee_type == "FT"), default=160),
        pt_max_hours=max((e.max_hours for e in request.employees if e.employee_type == "PT"), default=40),
    )

    env = SchedulingEnv(config)
    state_dim = num_employees + 1
    action_dim = num_employees

    # Load model
    registry = get_registry()
    try:
        model = registry.load_dqn(checkpoint, state_dim, action_dim)
    except FileNotFoundError:
        # Fallback: random assignment if no trained model
        return _random_schedule(request, config)

    # Build unavailability lookup
    unavail_set: set[tuple[int, int]] = set()
    for u in request.unavailability:
        emp_idx = next(
            (i for i, e in enumerate(request.employees) if e.id == u.employee_id),
            None,
        )
        if emp_idx is not None:
            unavail_set.add((u.day, emp_idx))

    # Run inference
    schedule_actions: list[int] = []
    state = env.reset()
    done = False

    while not done:
        shift_index = int(state[0])
        day = shift_index // shifts_per_day
        shift_in_day = shift_index % shifts_per_day

        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_v).squeeze(0)

        # Mask unavailable employees
        for emp_idx in range(num_employees):
            if (day, emp_idx) in unavail_set:
                q_values[emp_idx] = float("-inf")

        action = int(torch.argmax(q_values).item())
        schedule_actions.append(action)
        state, _, done, _ = env.step(action)

    # Convert to response
    assignments: list[ShiftAssignment] = []
    hours_by_employee: dict[int, int] = {e.id: 0 for e in request.employees}

    for i, emp_idx in enumerate(schedule_actions):
        day = i // shifts_per_day
        shift_idx = i % shifts_per_day
        emp = request.employees[emp_idx]
        assignments.append(
            ShiftAssignment(day=day, shift_index=shift_idx, employee_id=emp.id)
        )
        hours_by_employee[emp.id] += shift_lengths[shift_idx]

    hours_values = list(hours_by_employee.values())
    fairness = 1.0 - (max(hours_values) - min(hours_values)) / max(max(hours_values), 1)

    return SchedulingResponse(
        schedule=assignments,
        metrics=ScheduleMetrics(
            fairness_score=round(fairness, 4),
            total_hours_by_employee=hours_by_employee,
        ),
    )


def _random_schedule(
    request: SchedulingRequest,
    config: EnvironmentConfig,
) -> SchedulingResponse:
    """Fallback random schedule when no model is available."""
    assignments: list[ShiftAssignment] = []
    hours_by_employee: dict[int, int] = {e.id: 0 for e in request.employees}
    num_employees = len(request.employees)
    shifts_per_day = len(request.shifts)

    for day in range(request.days):
        for shift_idx in range(shifts_per_day):
            emp_idx = (day * shifts_per_day + shift_idx) % num_employees
            emp = request.employees[emp_idx]
            assignments.append(
                ShiftAssignment(day=day, shift_index=shift_idx, employee_id=emp.id)
            )
            hours_by_employee[emp.id] += request.shifts[shift_idx].length_hours

    hours_values = list(hours_by_employee.values())
    fairness = 1.0 - (max(hours_values) - min(hours_values)) / max(max(hours_values), 1)

    return SchedulingResponse(
        schedule=assignments,
        metrics=ScheduleMetrics(
            fairness_score=round(fairness, 4),
            total_hours_by_employee=hours_by_employee,
        ),
    )
