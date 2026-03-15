"""Shared schedule quality metrics computation."""

from models.problem import jain_fairness_index
from server.schemas import (
    ConstraintViolations,
    ScheduleMetrics,
    SchedulingRequest,
    ShiftAssignment,
)


def compute_metrics(
    assignments: list[ShiftAssignment],
    request: SchedulingRequest,
    hours_by_employee: dict[int, int],
) -> ScheduleMetrics:
    """Compute comprehensive schedule quality metrics."""
    shifts_per_day = len(request.shifts)
    num_shifts = request.days * shifts_per_day

    # Build unavailability lookup by employee_id
    unavail_set: set[tuple[int, int]] = set()
    for u in request.unavailability:
        unavail_set.add((u.day, u.employee_id))

    hours_values = list(hours_by_employee.values())

    # Fairness scores
    max_h = max(hours_values) if hours_values else 0
    min_h = min(hours_values) if hours_values else 0
    fairness_score = 1.0 - (max_h - min_h) / max(max_h, 1)
    jain_idx = jain_fairness_index(hours_values)

    # Constraint violations
    unavail_violations = sum(
        1 for a in assignments if (a.day, a.employee_id) in unavail_set
    )
    max_hours_violations = sum(
        1 for emp in request.employees if hours_by_employee.get(emp.id, 0) > emp.max_hours
    )

    violations = ConstraintViolations(
        unavailability_violations=unavail_violations,
        max_hours_violations=max_hours_violations,
        total_violations=unavail_violations + max_hours_violations,
    )

    # Back-to-back rate
    back_to_back = 0
    sorted_assignments = sorted(assignments, key=lambda a: (a.day, a.shift_index))
    for i in range(len(sorted_assignments) - 1):
        curr = sorted_assignments[i]
        nxt = sorted_assignments[i + 1]
        curr_global = curr.day * shifts_per_day + curr.shift_index
        nxt_global = nxt.day * shifts_per_day + nxt.shift_index
        if nxt_global == curr_global + 1 and curr.employee_id == nxt.employee_id:
            back_to_back += 1

    back_to_back_rate = back_to_back / max(len(assignments) - 1, 1)

    # Coverage rate
    coverage_rate = len(assignments) / max(num_shifts, 1)

    # Shift type distribution: employee_id -> {shift_index: count}
    shift_type_dist: dict[int, dict[int, int]] = {e.id: {} for e in request.employees}
    for a in assignments:
        dist = shift_type_dist[a.employee_id]
        dist[a.shift_index] = dist.get(a.shift_index, 0) + 1

    return ScheduleMetrics(
        fairness_score=round(fairness_score, 4),
        jain_fairness_index=round(jain_idx, 4),
        total_hours_by_employee=hours_by_employee,
        constraint_violations=violations,
        back_to_back_rate=round(back_to_back_rate, 4),
        coverage_rate=round(coverage_rate, 4),
        shift_type_distribution=shift_type_dist,
    )
