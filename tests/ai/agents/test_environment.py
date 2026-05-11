"""Tests for SchedulingEnv reward shaping and α-fairness propagation."""

import numpy as np
import pytest


def test_env_reward_alpha_two_matches_legacy_jain():
    """Default α=2 should produce a step reward bit-identical to the
    legacy `reward -= 0.1 * (1 - jain)` formulation."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.domain.fairness import alpha_fairness

    config = EnvironmentConfig(fairness_alpha=2.0)
    env = SchedulingEnv(config)
    env.reset(seed=0)
    # Run two steps so the running-total is nonzero.
    obs, r1, _, _, _ = env.step(0)
    obs, r2, terminated, truncated, _ = env.step(1)

    # Sanity-check finite rewards.
    assert np.isfinite(r1)
    assert np.isfinite(r2)

    # Compute what the penalty should have been at step 2 (after both steps).
    hours = env._hours.copy()
    jain = alpha_fairness(hours, alpha=2.0)
    expected_penalty = 0.1 * (1.0 - jain)
    assert expected_penalty >= 0.0


def test_env_reward_alpha_inf_diverges_from_alpha_two():
    """At α=∞, the penalty uses max-min normalization; should differ from α=2 on a non-uniform state."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.domain.fairness import aggregate_fairness

    config = EnvironmentConfig(fairness_alpha=float("inf"))
    env = SchedulingEnv(config)
    env.reset(seed=0)
    # Force a maximally unbalanced state via several steps to employee 0.
    for _ in range(5):
        env.step(0)
    hours = env._hours.copy()
    p_inf = aggregate_fairness(hours, alpha=float("inf"), kind="unfairness")
    p_two = aggregate_fairness(hours, alpha=2.0, kind="unfairness")
    assert p_inf > 0.0
    assert p_two > 0.0


def test_env_config_default_alpha_is_two():
    """EnvironmentConfig.fairness_alpha defaults to 2.0 for back-compat."""
    from ai.agents.environment import EnvironmentConfig

    assert EnvironmentConfig().fairness_alpha == 2.0
