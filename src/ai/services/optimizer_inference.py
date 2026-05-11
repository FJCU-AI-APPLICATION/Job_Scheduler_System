"""Single inference service that dispatches to any registered optimizer.

The dispatch generalizes over optimizer families: the route handler builds
a dict of validated query params; the dispatch passes them straight to the
optimizer's config_class. Pydantic config models reject unknown keys, so
mis-routed knobs surface at config-build time, not at solve time.
"""

from typing import Any

from fastapi import HTTPException

from ai.domain.problem import ScheduleConverter, SchedulingProblem
from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.optimizers.cpsat import CPSATInfeasibleError, CPSATTimeoutError
from ai.optimizers.matheuristic import MatheuristicError
from ai.optimizers.result import CCMOResult
from ai.services.metrics import compute_metrics


def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    config_overrides: dict[str, Any] | None = None,
) -> SchedulingResponse:
    """Dispatch to the named optimizer; convert its best schedule to the API response."""
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)
    config = optimizer.config_class(**(config_overrides or {}))

    try:
        result = optimizer.run(config)
    except CPSATInfeasibleError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Instance is infeasible (stage={e.stage}, status={e.status_name})",
        )
    except CPSATTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Solver budget exhausted (stage={e.stage}, elapsed={e.elapsed_s:.1f}s)",
        )
    except MatheuristicError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if isinstance(result, CCMOResult) and result.fell_back_to_auxiliary:
        raise HTTPException(
            status_code=422,
            detail="No feasible schedule found; instance is over-constrained or budget too tight",
        )

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(result.best_schedule)
    metrics = compute_metrics(
        assignments,
        request,
        hours_by_employee,
        fairness_alpha=getattr(config, "fairness_alpha", 2.0),
    )
    return SchedulingResponse(schedule=assignments, metrics=metrics)
