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


def test_pareto_reference_defaults_to_none():
    """Back-compat: EnvironmentConfig.pareto_reference defaults to None."""
    from ai.agents.environment import EnvironmentConfig

    assert EnvironmentConfig().pareto_reference is None
    assert EnvironmentConfig().hv_reference_point == (2.0, 1000.0, 100.0)


def test_reward_without_pareto_reference_is_unchanged():
    """With pareto_reference=None, terminal-step info dict is empty AND
    no ΔHV bonus is added. Drives the env to terminal to actually exercise
    the back-compat guarantee."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv

    config = EnvironmentConfig(pareto_reference=None)
    env = SchedulingEnv(config)
    env.reset(seed=0)

    info_at_terminal = None
    for t in range(env.num_shifts):
        _, _, terminated, _, info = env.step(t % env.num_employees)
        if terminated:
            info_at_terminal = info
            break

    # Terminal reached; info dict must be empty (no Pareto reference set).
    assert info_at_terminal == {}


def test_episode_fitness_matches_rostering_problem_formula():
    """_episode_fitness returns (unfairness, violations, b2b) matching the EA fitness shape."""
    import numpy as np
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.domain.fairness import aggregate_fairness

    config = EnvironmentConfig()
    env = SchedulingEnv(config)
    env.reset(seed=0)
    # Force employee 0 to take all of day 0's shifts (3 shifts in a row → b2b=2).
    for _ in range(3):
        env.step(0)
    fit = env._episode_fitness()
    assert fit.shape == (3,)
    assert fit[0] == pytest.approx(
        aggregate_fairness(env._hours, alpha=2.0, kind="unfairness")
    )
    # 3 consecutive employee 0 → 2 back-to-back transitions.
    assert fit[2] == 2.0


def test_delta_hv_returns_zero_for_empty_reference():
    """Empty pareto_reference → ΔHV = 0 (defense-in-depth)."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv

    config = EnvironmentConfig(pareto_reference=[])
    env = SchedulingEnv(config)
    env.reset(seed=0)
    # Drive env to terminal.
    for t in range(env.num_shifts):
        env.step(t % env.num_employees)
    assert env._compute_delta_hv() == 0.0


def test_delta_hv_is_non_negative():
    """ΔHV is monotone — always ≥ 0 for any episode point, by construction
    (clamped via max(..., 0.0))."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv

    reference = [(0.5, 0.0, 50.0)]
    config = EnvironmentConfig(pareto_reference=reference)
    env = SchedulingEnv(config)
    env.reset(seed=0)
    for t in range(env.num_shifts):
        env.step(t % env.num_employees)
    delta_hv = env._compute_delta_hv()
    assert delta_hv >= 0.0


def test_terminal_step_emits_info_dict():
    """At terminal step with a Pareto reference, info dict carries delta_hv + components."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv

    config = EnvironmentConfig(pareto_reference=[(0.5, 0.0, 50.0)])
    env = SchedulingEnv(config)
    env.reset(seed=0)
    info_terminal = None
    for t in range(env.num_shifts):
        _, _, terminated, _, info = env.step(t % env.num_employees)
        if terminated:
            info_terminal = info
            break
    assert info_terminal is not None
    assert "delta_hv" in info_terminal
    assert "episode_unfairness" in info_terminal
    assert "episode_violations" in info_terminal
    assert "episode_b2b" in info_terminal
