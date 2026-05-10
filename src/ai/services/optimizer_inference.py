"""Single inference service that dispatches to any registered optimizer."""

from fastapi import HTTPException

from ai.domain.problem import ScheduleConverter, SchedulingProblem
from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.optimizers.result import CCMOResult
from ai.services.metrics import compute_metrics


def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
    device: str = "cpu",
) -> SchedulingResponse:
    """Dispatch to the named optimizer; convert its best schedule to the API response."""
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)

    config = optimizer.config_class(
        generations=generations,
        pop_size=pop_size,
        device=device,
    )
    result = optimizer.run(config)

    if isinstance(result, CCMOResult) and result.fell_back_to_auxiliary:
        raise HTTPException(
            status_code=422,
            detail="No feasible schedule found; instance is over-constrained or budget too tight",
        )

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(result.best_schedule)
    metrics = compute_metrics(assignments, request, hours_by_employee)
    return SchedulingResponse(schedule=assignments, metrics=metrics)
