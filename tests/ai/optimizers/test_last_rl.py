"""Tests for LastRLOptimizer — issue #19 acceptance criteria."""

import pytest
from pydantic import ValidationError


def test_alpha_inf_validator():
    """LastRLConfig(fairness_alpha=2.0) raises ValidationError."""
    from ai.optimizers.result import LastRLConfig

    with pytest.raises(ValidationError):
        LastRLConfig(fairness_alpha=2.0)
    with pytest.raises(ValidationError):
        LastRLConfig(fairness_alpha=0.0)
    LastRLConfig(fairness_alpha=float("inf"))


def test_epsilon_bounds_validator():
    """epsilon_start and epsilon_end must each be in [0, 1]."""
    from ai.optimizers.result import LastRLConfig

    with pytest.raises(ValidationError):
        LastRLConfig(epsilon_start=1.5)
    with pytest.raises(ValidationError):
        LastRLConfig(epsilon_end=-0.1)


def test_epsilon_schedule_validator():
    """epsilon_start must be >= epsilon_end (decay schedule, not inflation)."""
    from ai.optimizers.result import LastRLConfig

    with pytest.raises(ValidationError):
        LastRLConfig(epsilon_start=0.05, epsilon_end=0.5)
    # Equal is fine (constant schedule)
    LastRLConfig(epsilon_start=0.5, epsilon_end=0.5)


def test_config_defaults():
    """LastRLConfig() has the documented defaults."""
    from ai.optimizers.result import LastRLConfig

    c = LastRLConfig()
    assert c.num_episodes == 200
    assert c.episode_length == 500
    assert c.wall_clock_budget_s is None
    assert c.alpha == 0.1
    assert c.gamma == 0.99
    assert c.lam == 0.9
    assert c.epsilon_start == 0.5
    assert c.epsilon_end == 0.05
    assert c.iht_size == 4096
    assert c.num_tilings == 8
    assert c.ip_time_budget_s == 2.0
    assert c.ip_workers == 2
    assert c.checkpoint_path is None
    assert c.fairness_alpha == float("inf")


def test_step_status_construct():
    """LastRLStepStatus constructs with all fields."""
    from ai.optimizers.result import LastRLStepStatus

    s = LastRLStepStatus(
        step=0, llh_name="single_move", action=0,
        reward=1.5, current_cost=10.0, best_cost=10.0, stagnation_count=0,
    )
    assert s.step == 0
    assert s.llh_name == "single_move"


def test_episode_status_construct():
    """LastRLEpisodeStatus constructs with all fields."""
    from ai.optimizers.result import LastRLEpisodeStatus

    e = LastRLEpisodeStatus(
        episode=0, epsilon=0.5, initial_cost=100.0, final_cost=50.0,
        best_cost_in_episode=45.0,
        neighborhood_usage={"single_move": 200, "random_swap": 300},
        wall_clock_s=12.3, total_reward=50.0, mean_step_reward=0.1,
        fraction_improving_steps=0.4,
    )
    assert e.fraction_improving_steps == 0.4


def test_result_schemas_import():
    """All schemas importable; LastRLResult inherits OptimizerResult."""
    from ai.optimizers.result import LastRLResult, OptimizerResult

    assert issubclass(LastRLResult, OptimizerResult)


def test_checkpoint_schemas_import():
    """LastRLConfigSnapshot + LastRLTrainResult are importable without cycles."""
    from ai.domain.schemas import LastRLConfigSnapshot, LastRLTrainResult

    snap = LastRLConfigSnapshot(
        num_employees=7, employee_types=["FT"] * 7,
        days=30, shifts_per_day=3, shift_lengths=[9, 8, 7],
        num_episodes=10, episode_length=100, wall_clock_budget_s=None,
        alpha=0.1, gamma=0.99, lam=0.9,
        epsilon_start=0.5, epsilon_end=0.05,
        iht_size=4096, num_tilings=8,
        ip_time_budget_s=2.0, ip_workers=2,
        fairness_alpha=float("inf"),
    )
    assert snap.num_employees == 7
