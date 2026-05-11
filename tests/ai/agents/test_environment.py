"""Tests for SchedulingEnv reward shaping and α-fairness propagation."""

import math

import pytest


def test_env_reward_alpha_two_matches_legacy_jain():
    """At α=2.0, step() reward matches the legacy `reward -= 0.1 * (1 − jain)`
    formulation bit-for-bit.

    Verified by reconstructing the reward formula for a known action sequence:
    step 1 sees `_hours.sum() == 0` (no fairness penalty applied; reward = +0.5
    completion bonus). Step 2 sees `_hours = [9, 0, 0, …, 0]` before the action's
    increment, so the fairness penalty is `0.1 * (1 − jain([9, 0, …, 0]))`.
    """
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.domain.problem import jain_fairness_index

    config = EnvironmentConfig(fairness_alpha=2.0)
    env = SchedulingEnv(config)
    env.reset(seed=0)

    # Step 1: hours.sum() == 0, no fairness penalty; action=0 is the first action so
    # there's no back-to-back penalty either. Expected reward = +0.5 (completion bonus).
    _, r1, _, _, _ = env.step(0)
    assert r1 == pytest.approx(0.5)

    # Step 2: capture hours BEFORE the increment (the fairness penalty reads them at
    # that point), then act on a different employee so there's no back-to-back penalty.
    hours_before_step2 = env._hours.copy()
    _, r2, _, _, _ = env.step(1)
    expected_penalty = 0.1 * (1.0 - jain_fairness_index(hours_before_step2))
    assert r2 == pytest.approx(0.5 - expected_penalty)


def test_env_reward_alpha_inf_diverges_from_alpha_two():
    """At α=∞ vs α=2, the unfairness penalty on a maximally skewed state differs.

    Forces all 5 first shifts onto employee 0 so the per-employee hours vector is
    [45, 0, 0, …, 0] (approx, depending on shift_lengths). Both α=∞ (max-min) and
    α=2 (Jain) penalties are positive, but the values should differ — α=∞ saturates
    earlier than α=2 on this state.
    """
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.domain.fairness import aggregate_fairness

    config = EnvironmentConfig(fairness_alpha=float("inf"))
    env = SchedulingEnv(config)
    env.reset(seed=0)
    for _ in range(5):
        env.step(0)
    hours = env._hours.copy()

    p_inf = aggregate_fairness(hours, alpha=float("inf"), kind="unfairness")
    p_two = aggregate_fairness(hours, alpha=2.0, kind="unfairness")

    assert p_inf > 0.0
    assert p_two > 0.0
    assert not math.isclose(p_inf, p_two, rel_tol=1e-6), (
        f"p_inf={p_inf} and p_two={p_two} should differ on a maximally skewed state"
    )


def test_env_config_default_alpha_is_two():
    """EnvironmentConfig.fairness_alpha defaults to 2.0 for back-compat."""
    from ai.agents.environment import EnvironmentConfig

    assert EnvironmentConfig().fairness_alpha == 2.0
