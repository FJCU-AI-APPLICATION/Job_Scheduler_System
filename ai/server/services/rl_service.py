"""RL inference service using SB3 models with action masking."""

import torch

from models.environment import EnvironmentConfig, SchedulingEnv
from models.problem import ScheduleConverter, SchedulingProblem

from server.schemas import SchedulingRequest, SchedulingResponse
from server.services.metrics import compute_metrics
from server.services.model_registry import get_registry


def run_rl_inference(
    request: SchedulingRequest,
    checkpoint: str = "best_model.zip",
) -> SchedulingResponse:
    """Run SB3 model inference to generate an optimized schedule."""
    problem = SchedulingProblem.from_request(request)

    config = EnvironmentConfig(
        num_employees=problem.num_employees,
        employee_types=list(problem.employee_types),
        days=problem.days,
        shifts_per_day=problem.shifts_per_day,
        shift_lengths=list(problem.shift_lengths),
        ft_max_hours=max(
            (e.max_hours for e in request.employees if e.employee_type == "FT"),
            default=160,
        ),
        pt_max_hours=max(
            (e.max_hours for e in request.employees if e.employee_type == "PT"),
            default=40,
        ),
        unavailability=set(problem.unavailability),
    )

    env = SchedulingEnv(config)

    # Load SB3 model
    registry = get_registry()
    try:
        model = registry.load_model(checkpoint)
    except FileNotFoundError:
        return _random_schedule(request, problem)

    # Run inference
    schedule_actions: list[int] = []
    obs, _ = env.reset()
    done = False

    while not done:
        action_masks = env.action_masks()

        try:
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
        except TypeError:
            action, _ = model.predict(obs, deterministic=True)
            if not action_masks[int(action)]:
                valid = torch.where(torch.as_tensor(action_masks))[0]
                action = valid[torch.randint(len(valid), (1,))].item() if len(valid) > 0 else action

        action = int(action)
        schedule_actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(schedule_actions)
    metrics = compute_metrics(assignments, request, hours_by_employee)

    return SchedulingResponse(schedule=assignments, metrics=metrics)


def _random_schedule(
    request: SchedulingRequest,
    problem: SchedulingProblem,
) -> SchedulingResponse:
    """Fallback random schedule when no model is available."""
    num_shifts = problem.num_shifts
    schedule = [i % problem.num_employees for i in range(num_shifts)]

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(schedule)
    metrics = compute_metrics(assignments, request, hours_by_employee)

    return SchedulingResponse(schedule=assignments, metrics=metrics)
