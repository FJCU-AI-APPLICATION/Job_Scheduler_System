"""Tests for schedule quality metrics."""

import pytest

from ai.domain.schemas import (
    EmployeeInfo,
    ShiftInfo,
    SchedulingRequest,
)


def _tiny_request_and_assignments():
    """Synthesize a tiny request + a corresponding assignments list."""
    from ai.domain.schemas import ShiftAssignment
    request = SchedulingRequest(
        employees=[
            EmployeeInfo(id=10, employee_type="FT", max_hours=50),
            EmployeeInfo(id=11, employee_type="FT", max_hours=50),
        ],
        days=2,
        shifts=[
            ShiftInfo(start_time="06:00:00", end_time="14:00:00", length_hours=8),
            ShiftInfo(start_time="14:00:00", end_time="22:00:00", length_hours=8),
        ],
        unavailability=[],
    )
    assignments = [
        ShiftAssignment(day=0, shift_index=0, employee_id=10),
        ShiftAssignment(day=0, shift_index=1, employee_id=11),
        ShiftAssignment(day=1, shift_index=0, employee_id=10),
        ShiftAssignment(day=1, shift_index=1, employee_id=11),
    ]
    hours_by_employee = {10: 16, 11: 16}
    return request, assignments, hours_by_employee


def test_default_alpha_metric_equals_legacy_jain():
    """At default α=2, fairness_metric ≡ jain_fairness_index within rounding (4 decimals)."""
    from ai.domain.problem import jain_fairness_index
    from ai.services.metrics import compute_metrics

    request, assignments, hours_by_employee = _tiny_request_and_assignments()
    metrics = compute_metrics(assignments, request, hours_by_employee)

    expected = jain_fairness_index(list(hours_by_employee.values()))
    assert metrics.fairness_metric == pytest.approx(expected, abs=1e-4)
    assert metrics.fairness_alpha == 2.0


def test_alpha_inf_metric_is_min_hours():
    """At α=∞, fairness_metric = min(hours)."""
    from ai.services.metrics import compute_metrics

    request, assignments, hours_by_employee = _tiny_request_and_assignments()
    metrics = compute_metrics(
        assignments, request, hours_by_employee, fairness_alpha=float("inf")
    )

    assert metrics.fairness_metric == pytest.approx(min(hours_by_employee.values()))
    assert metrics.fairness_alpha == float("inf")


def test_legacy_jain_alias_still_parses():
    """Old JSON with 'jain_fairness_index' field still deserializes."""
    from ai.domain.schemas import ScheduleMetrics

    legacy_json = {
        "fairness_score": 0.9,
        "jain_fairness_index": 0.95,         # legacy name
        "total_hours_by_employee": {10: 16, 11: 16},
        "constraint_violations": {
            "unavailability_violations": 0,
            "max_hours_violations": 0,
            "total_violations": 0,
        },
        "back_to_back_rate": 0.0,
        "coverage_rate": 1.0,
        "shift_type_distribution": {},
    }
    metrics = ScheduleMetrics.model_validate(legacy_json)
    assert metrics.fairness_metric == 0.95
    assert metrics.fairness_alpha == 2.0   # default fills in
