"""End-to-end tests for run_optimizer_inference dispatching to CP-SAT."""

import pytest
from fastapi import HTTPException

from ai.domain.schemas import EmployeeInfo, SchedulingRequest, ShiftInfo


def _tiny_request() -> SchedulingRequest:
    return SchedulingRequest(
        employees=[
            EmployeeInfo(id=10, employee_type="FT", max_hours=50),
            EmployeeInfo(id=11, employee_type="FT", max_hours=50),
            EmployeeInfo(id=12, employee_type="PT", max_hours=20),
        ],
        days=7,
        shifts=[
            ShiftInfo(start_time="06:00:00", end_time="14:00:00", length_hours=8),
            ShiftInfo(start_time="14:00:00", end_time="22:00:00", length_hours=8),
        ],
        unavailability=[],
    )


def _over_constrained_request() -> SchedulingRequest:
    return SchedulingRequest(
        employees=[
            EmployeeInfo(id=10, employee_type="PT", max_hours=20),
            EmployeeInfo(id=11, employee_type="PT", max_hours=20),
            EmployeeInfo(id=12, employee_type="PT", max_hours=20),
        ],
        days=7,
        shifts=[
            ShiftInfo(start_time="06:00:00", end_time="14:00:00", length_hours=8),
            ShiftInfo(start_time="14:00:00", end_time="22:00:00", length_hours=8),
            ShiftInfo(start_time="22:00:00", end_time="06:00:00", length_hours=8),
        ],
        unavailability=[],
    )


def test_round_trip_through_dispatch():
    """run_optimizer_inference("cpsat", request, …) returns a valid SchedulingResponse."""
    from ai.services.optimizer_inference import run_optimizer_inference

    response = run_optimizer_inference(
        "cpsat",
        _tiny_request(),
        config_overrides={"timeout_s_per_stage": 10.0, "num_workers": 2, "seed": 42},
    )

    assert len(response.schedule) == 7 * 2
    for assignment in response.schedule:
        assert assignment.employee_id in {10, 11, 12}


def test_infeasible_returns_422():
    """CPSATInfeasibleError surfaces as HTTPException(422)."""
    from ai.services.optimizer_inference import run_optimizer_inference

    with pytest.raises(HTTPException) as exc:
        run_optimizer_inference(
            "cpsat",
            _over_constrained_request(),
            config_overrides={"timeout_s_per_stage": 10.0, "num_workers": 2, "seed": 42},
        )
    assert exc.value.status_code == 422
    assert "infeasible" in exc.value.detail.lower()


def test_timeout_returns_504_via_mock(mocker):
    """Mocked CPSATTimeoutError surfaces as HTTPException(504)."""
    from ortools.sat.python import cp_model

    from ai.services.optimizer_inference import run_optimizer_inference

    mocker.patch.object(cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN)
    mocker.patch.object(cp_model.CpSolver, "WallTime", return_value=0.1)

    with pytest.raises(HTTPException) as exc:
        run_optimizer_inference(
            "cpsat",
            _tiny_request(),
            config_overrides={"timeout_s_per_stage": 1.0, "num_workers": 1},
        )
    assert exc.value.status_code == 504


def test_evolutionary_dispatch_still_works():
    """Regression guard: nsga2 route through the new dict-shape dispatch."""
    from ai.services.optimizer_inference import run_optimizer_inference

    response = run_optimizer_inference(
        "nsga2",
        _tiny_request(),
        config_overrides={"generations": 5, "pop_size": 20, "device": "cpu"},
    )

    assert len(response.schedule) == 7 * 2
