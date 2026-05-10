"""Shared pytest fixtures for the AI service test suite."""

import pytest

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem


@pytest.fixture
def tiny_problem() -> SchedulingProblem:
    """3 employees × 7 days × 2 shifts/day. Solves in <1s."""
    return SchedulingProblem(
        num_employees=3,
        employee_types=("FT", "FT", "PT"),
        days=7,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 20),
        unavailability=frozenset(),
    )


@pytest.fixture
def default_problem() -> SchedulingProblem:
    """The canonical 7×30×3 instance from EnvironmentConfig defaults."""
    return SchedulingProblem.from_config(EnvironmentConfig())


@pytest.fixture
def over_constrained_problem() -> SchedulingProblem:
    """An instance where total demand exceeds total cap. For infeasibility tests.

    3 employees, all PT (max 20h each = 60h cap total), 7 days × 3 shifts × 8h = 168h demand.
    """
    return SchedulingProblem(
        num_employees=3,
        employee_types=("PT", "PT", "PT"),
        days=7,
        shifts_per_day=3,
        shift_lengths=(8, 8, 8),
        max_hours=(20, 20, 20),
        unavailability=frozenset(),
    )
