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


import numpy as np


def test_sarsa_policy_q_zero_at_init():
    """Fresh policy returns Q=0 for any (features, action)."""
    from ai.optimizers.last_rl import SARSALambdaPolicy

    p = SARSALambdaPolicy(iht_size=4096, num_tilings=8, num_actions=9)
    f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert p.q(f, 0) == 0.0
    assert p.q_all(f).tolist() == [0.0] * 9


def test_sarsa_policy_update_with_zero_delta_doesnt_change_weights():
    """If reward==0 and Q(s',a')==Q(s,a), update leaves weights unchanged."""
    from ai.optimizers.last_rl import SARSALambdaPolicy

    p = SARSALambdaPolicy(iht_size=4096, num_tilings=8, num_actions=9)
    f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.update(f, 0, 0.0, f, 0, alpha=0.1, gamma=0.99, lam=0.9)
    assert np.allclose(p.w, 0.0)


def test_sarsa_policy_learns_constant_reward_problem():
    """On a 2-action problem where action 0 → r=+1, action 1 → r=-1:
    after enough updates, argmax(Q(s, ·)) == 0 for any s."""
    from ai.optimizers.last_rl import SARSALambdaPolicy

    p = SARSALambdaPolicy(iht_size=512, num_tilings=8, num_actions=2)
    f = np.array([0.5, 0.5])

    for step in range(500):
        action = step % 2
        reward = 1.0 if action == 0 else -1.0
        next_action = (step + 1) % 2
        p.update(f, action, reward, f, next_action, alpha=0.1, gamma=0.0, lam=0.0)

    q = p.q_all(f)
    assert q[0] > q[1], f"expected action 0 to win, got Q={q}"


def test_sarsa_replacing_trace_resets_per_update():
    """After visiting (s, a=0), then (s, a=1), trace for action 0 should decay."""
    from ai.optimizers.last_rl import SARSALambdaPolicy

    p = SARSALambdaPolicy(iht_size=256, num_tilings=8, num_actions=2)
    f = np.array([0.3, 0.7])
    p.update(f, 0, 1.0, f, 0, alpha=0.1, gamma=0.9, lam=0.9)
    e_after_first = p.e.copy()
    assert e_after_first[:, 1].max() == 0.0
    assert e_after_first[:, 0].max() == 1.0

    p.update(f, 1, 1.0, f, 1, alpha=0.1, gamma=0.9, lam=0.9)
    assert e_after_first[:, 0].max() > p.e[:, 0].max()
    assert p.e[:, 1].max() == 1.0


def test_search_history_construct():
    """SearchHistory is a dataclass with default-initializable fields."""
    from ai.optimizers.last_rl_problem import SearchHistory

    h = SearchHistory(
        iteration=0, max_iterations=100,
        current_cost=10.0, initial_cost=10.0, best_cost=10.0,
        stagnation_count=0, last_llh_idx=-1, last_reward=0.0,
        last_5_llh_indices=[], last_5_rewards=[],
    )
    assert h.iteration == 0


def test_select_action_epsilon_one_random():
    """At ε=1.0, all 1000 calls return an integer in [0, num_actions)."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, select_action

    p = SARSALambdaPolicy(iht_size=256, num_tilings=8, num_actions=4)
    f = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    actions = [select_action(p, f, 1.0, rng, 4) for _ in range(1000)]
    assert all(0 <= a < 4 for a in actions)
    assert len(set(actions)) == 4


def test_select_action_epsilon_zero_greedy():
    """At ε=0.0, action == argmax(q_all)."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, select_action

    p = SARSALambdaPolicy(iht_size=256, num_tilings=8, num_actions=3)
    f = np.array([0.5, 0.5])
    p.update(f, 1, 1.0, f, 1, alpha=0.5, gamma=0.0, lam=0.0)
    rng = np.random.default_rng(0)
    actions = [select_action(p, f, 0.0, rng, 3) for _ in range(20)]
    assert actions == [1] * 20


def test_run_episode_step_history_full(tiny_problem):
    """After run_episode, step_history length == episode_length and usage sums up."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, run_episode
    from ai.optimizers.last_rl_problem import RosteringLastRLProblem
    from ai.optimizers.result import LastRLConfig

    config = LastRLConfig(episode_length=20, ip_time_budget_s=0.5, ip_workers=1, seed=42)
    problem = RosteringLastRLProblem(tiny_problem, config)
    policy = SARSALambdaPolicy(iht_size=512, num_tilings=8, num_actions=problem.num_actions)
    rng = np.random.default_rng(42)
    result = run_episode(problem, policy, config, epsilon=0.3, rng=rng, learning=True)

    assert len(result.step_history) == 20
    assert sum(result.neighborhood_usage.values()) == 20


def test_run_episode_tracks_best(tiny_problem):
    """best_cost in step_history is monotone non-increasing."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, run_episode
    from ai.optimizers.last_rl_problem import RosteringLastRLProblem
    from ai.optimizers.result import LastRLConfig

    config = LastRLConfig(episode_length=30, ip_time_budget_s=0.5, ip_workers=1, seed=1)
    problem = RosteringLastRLProblem(tiny_problem, config)
    policy = SARSALambdaPolicy(iht_size=512, num_tilings=8, num_actions=problem.num_actions)
    rng = np.random.default_rng(1)
    result = run_episode(problem, policy, config, epsilon=0.3, rng=rng, learning=True)

    prev = float("inf")
    for s in result.step_history:
        assert s.best_cost <= prev + 1e-9
        prev = s.best_cost


def test_run_episode_all_moves_accepts_unconditionally(tiny_problem, mocker):
    """Mock LLH that always worsens cost; current_cost still changes each step."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, run_episode
    from ai.optimizers.last_rl_problem import RosteringLastRLProblem
    from ai.optimizers.result import LastRLConfig

    config = LastRLConfig(episode_length=10, ip_time_budget_s=0.5, ip_workers=1, seed=2)
    problem = RosteringLastRLProblem(tiny_problem, config)

    bad_sched = [0] * tiny_problem.num_shifts
    original_lib = problem.llh_library()
    wrapped = []
    for llh in original_lib:
        from ai.optimizers.llh import partial_with_name
        wrapped.append(partial_with_name(llh.name, lambda s, r, _b=bad_sched: list(_b)))
    mocker.patch.object(problem, "llh_library", return_value=wrapped)

    policy = SARSALambdaPolicy(iht_size=256, num_tilings=8, num_actions=problem.num_actions)
    rng = np.random.default_rng(2)
    result = run_episode(problem, policy, config, epsilon=0.0, rng=rng, learning=False)

    bad_cost = problem.cost(bad_sched)
    assert all(abs(s.current_cost - bad_cost) < 1e-9 for s in result.step_history)


def test_train_returns_episodes(tiny_problem):
    """train() returns one EpisodeResult per episode."""
    from ai.optimizers.last_rl import SARSALambdaPolicy, train
    from ai.optimizers.last_rl_problem import RosteringLastRLProblem
    from ai.optimizers.result import LastRLConfig

    config = LastRLConfig(
        num_episodes=3, episode_length=10,
        ip_time_budget_s=0.5, ip_workers=1, seed=0,
    )
    problem = RosteringLastRLProblem(tiny_problem, config)
    policy = SARSALambdaPolicy(iht_size=256, num_tilings=8, num_actions=problem.num_actions)
    rng = np.random.default_rng(0)
    episodes = train(problem, policy, config, rng)

    assert len(episodes) == 3
    for ep in episodes:
        assert len(ep.step_history) == 10


def test_inference_requires_checkpoint(tiny_problem):
    """LastRLOptimizer.run() without checkpoint_path raises ValueError."""
    from ai.optimizers.last_rl import LastRLOptimizer
    from ai.optimizers.result import LastRLConfig

    optimizer = LastRLOptimizer(tiny_problem)
    with pytest.raises(ValueError) as exc:
        optimizer.run(LastRLConfig())
    assert "checkpoint_path" in str(exc.value)


def test_runs_via_create(tiny_problem):
    """Optimizer.create('last_rl', sp) returns a LastRLOptimizer."""
    from ai.optimizers.base import Optimizer
    from ai.optimizers.last_rl import LastRLOptimizer

    optimizer = Optimizer.create("last_rl", tiny_problem)
    assert isinstance(optimizer, LastRLOptimizer)
