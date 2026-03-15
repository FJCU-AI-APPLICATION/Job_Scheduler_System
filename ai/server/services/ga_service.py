"""GA inference service using the shared GAOptimizer."""

from models.ga_optimizer import GAConfig, GAOptimizer
from models.problem import ScheduleConverter, SchedulingProblem
from server.schemas import SchedulingRequest, SchedulingResponse
from server.services.metrics import compute_metrics


def run_ga_inference(
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
) -> SchedulingResponse:
    """Run NSGA-II Genetic Algorithm optimization for scheduling."""
    problem = SchedulingProblem.from_request(request)
    optimizer = GAOptimizer(problem)

    config = GAConfig(generations=generations, pop_size=pop_size)
    result = optimizer.run(config)

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(result.best_schedule)
    metrics = compute_metrics(assignments, request, hours_by_employee)

    return SchedulingResponse(schedule=assignments, metrics=metrics)
