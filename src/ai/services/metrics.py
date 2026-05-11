"""Schedule quality metrics. Each scorer is independent; compute_metrics composes them."""

from ai.domain.fairness import alpha_fairness
from ai.domain.schemas import (
    ConstraintViolations,
    ScheduleMetrics,
    SchedulingRequest,
    ShiftAssignment,
)


def compute_fairness(
    hours_by_employee: dict[int, int], alpha: float = 2.0
) -> tuple[float, float]:
    """Compute fairness scores from per-employee hour totals.

    Returns:
        (fairness_score, fairness_metric) where:
        - fairness_score: 1 - (max-min)/max, range [0, 1] (α-agnostic, easy to read)
        - fairness_metric: alpha_fairness(values, α), the configurable welfare value
    """
    hours_values = list(hours_by_employee.values())
    max_h = max(hours_values) if hours_values else 0
    min_h = min(hours_values) if hours_values else 0
    fairness_score = 1.0 - (max_h - min_h) / max(max_h, 1)
    fairness_metric = alpha_fairness(hours_values, alpha)
    return fairness_score, fairness_metric


def compute_violations(
    assignments: list[ShiftAssignment],
    request: SchedulingRequest,
    hours_by_employee: dict[int, int],
) -> ConstraintViolations:
    """Count constraint violations: unavailability and max hours."""
    unavail_set: set[tuple[int, int]] = {
        (u.day, u.employee_id) for u in request.unavailability
    }

    unavail_violations = sum(
        1 for a in assignments if (a.day, a.employee_id) in unavail_set
    )
    max_hours_violations = sum(
        1
        for emp in request.employees
        if hours_by_employee.get(emp.id, 0) > emp.max_hours
    )

    return ConstraintViolations(
        unavailability_violations=unavail_violations,
        max_hours_violations=max_hours_violations,
        total_violations=unavail_violations + max_hours_violations,
    )


def compute_back_to_back_rate(
    assignments: list[ShiftAssignment],
    shifts_per_day: int,
) -> float:
    """Rate of back-to-back same-employee shifts in [0, 1]."""
    if len(assignments) <= 1:
        return 0.0

    sorted_assignments = sorted(assignments, key=lambda a: (a.day, a.shift_index))
    back_to_back = 0
    for i in range(len(sorted_assignments) - 1):
        curr = sorted_assignments[i]
        nxt = sorted_assignments[i + 1]
        curr_global = curr.day * shifts_per_day + curr.shift_index
        nxt_global = nxt.day * shifts_per_day + nxt.shift_index
        if nxt_global == curr_global + 1 and curr.employee_id == nxt.employee_id:
            back_to_back += 1

    return back_to_back / (len(assignments) - 1)


def compute_coverage_rate(
    assignments: list[ShiftAssignment],
    total_shifts: int,
) -> float:
    """Fraction of shift slots filled, in [0, 1]."""
    if total_shifts <= 0:
        return 0.0
    return len(assignments) / total_shifts


def compute_shift_type_distribution(
    assignments: list[ShiftAssignment],
    employee_ids: list[int],
) -> dict[int, dict[int, int]]:
    """{employee_id: {shift_index: count}}"""
    dist: dict[int, dict[int, int]] = {eid: {} for eid in employee_ids}
    for a in assignments:
        emp_dist = dist[a.employee_id]
        emp_dist[a.shift_index] = emp_dist.get(a.shift_index, 0) + 1
    return dist


def compute_metrics(
    assignments: list[ShiftAssignment],
    request: SchedulingRequest,
    hours_by_employee: dict[int, int],
    fairness_alpha: float = 2.0,
) -> ScheduleMetrics:
    """Compute all schedule quality metrics."""
    shifts_per_day = len(request.shifts)
    total_shifts = request.days * shifts_per_day

    fairness_score, fairness_metric = compute_fairness(hours_by_employee, alpha=fairness_alpha)
    violations = compute_violations(assignments, request, hours_by_employee)
    b2b_rate = compute_back_to_back_rate(assignments, shifts_per_day)
    coverage = compute_coverage_rate(assignments, total_shifts)
    shift_dist = compute_shift_type_distribution(
        assignments, [e.id for e in request.employees]
    )

    return ScheduleMetrics(
        fairness_score=round(fairness_score, 4),
        fairness_metric=round(fairness_metric, 4),
        fairness_alpha=fairness_alpha,
        total_hours_by_employee=hours_by_employee,
        constraint_violations=violations,
        back_to_back_rate=round(b2b_rate, 4),
        coverage_rate=round(coverage, 4),
        shift_type_distribution=shift_dist,
    )
