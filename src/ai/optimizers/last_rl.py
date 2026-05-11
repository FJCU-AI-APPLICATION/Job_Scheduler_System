"""LAST-RL hyper-heuristic optimizer — issue #19.

Outer loop: SARSA(λ) over tile-coded state selects from a library of
low-level heuristics. Acceptance: All Moves (apply unconditionally,
reward = -Δcost). Best-seen schedule tracked separately.

Reference:
  Kletzander L., Musliu N. (2023). Large-State Reinforcement Learning for
  Hyper-Heuristics. AAAI.
  https://ojs.aaai.org/index.php/AAAI/article/view/26466

  Sutton R., Barto A. (2018). Reinforcement Learning: An Introduction
  (2nd ed.). Chapter 9 (tile coding).
"""

from __future__ import annotations

import numpy as np

from ai.optimizers.tiles3 import IHT, tiles


# === SARSA(λ) policy ===


class SARSALambdaPolicy:
    """Linear-FA SARSA(λ) over tile-coded (state, action) features.

    Replacing traces, alpha-divided-by-num_tilings normalization. One weight
    per (tile_idx, action) pair — stored as a (iht_size, num_actions) array.
    """

    def __init__(self, iht_size: int, num_tilings: int, num_actions: int):
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.w = np.zeros((iht_size, num_actions), dtype=np.float64)
        self.e = np.zeros((iht_size, num_actions), dtype=np.float64)

    def _tile_indices(self, features: np.ndarray) -> list[int]:
        """Hash continuous features to num_tilings hash slots in [0, iht_size)."""
        return tiles(
            self.iht,
            self.num_tilings,
            list((features * 10.0).astype(float)),
        )

    def q(self, features: np.ndarray, action: int) -> float:
        """Q(s, a) = sum of weights at the visited (tile_idx, a) pairs."""
        idx = self._tile_indices(features)
        return float(self.w[idx, action].sum())

    def q_all(self, features: np.ndarray) -> np.ndarray:
        """Q(s, ·) — vector over all actions; used for argmax."""
        idx = self._tile_indices(features)
        return self.w[idx, :].sum(axis=0)

    def update(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        next_action: int,
        alpha: float,
        gamma: float,
        lam: float,
    ) -> None:
        """One SARSA(λ) step with replacing traces."""
        idx = self._tile_indices(features)
        next_idx = self._tile_indices(next_features)
        q_sa = self.w[idx, action].sum()
        q_sa_next = self.w[next_idx, next_action].sum()
        delta = reward + gamma * q_sa_next - q_sa

        self.e *= gamma * lam
        self.e[idx, action] = 1.0
        self.w += (alpha / self.num_tilings) * delta * self.e

    def reset_traces(self) -> None:
        """Clear eligibility traces at the start of an episode."""
        self.e.fill(0.0)


import time
from dataclasses import dataclass

from ai.optimizers.last_rl_problem import LastRLProblem, SearchHistory
from ai.optimizers.result import LastRLConfig, LastRLStepStatus


@dataclass
class EpisodeResult:
    """One episode's run result — returned by run_episode."""
    best_sched: list[int]
    best_cost: float
    initial_cost: float
    final_cost: float
    step_history: list[LastRLStepStatus]
    neighborhood_usage: dict[str, int]
    wall_clock_s: float


def select_action(
    policy,
    features: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    num_actions: int,
) -> int:
    """ε-greedy action selection. Ties broken uniformly at random."""
    if rng.random() < epsilon:
        return int(rng.integers(0, num_actions))
    q_values = policy.q_all(features)
    max_q = q_values.max()
    argmax = np.flatnonzero(q_values == max_q)
    return int(rng.choice(argmax))


def run_episode(
    problem: LastRLProblem,
    policy: SARSALambdaPolicy,
    config: LastRLConfig,
    epsilon: float,
    rng: np.random.Generator,
    learning: bool = True,
) -> EpisodeResult:
    """Single SARSA(λ) episode.

    Training mode (learning=True): updates policy after each step.
    Inference mode (learning=False): no updates; usually ε=0.
    """
    sched = problem.initial_solution(rng)
    initial_cost = problem.cost(sched)
    best_sched = list(sched)
    best_cost = initial_cost

    llh_lib = problem.llh_library()
    history = SearchHistory(
        iteration=0,
        max_iterations=config.episode_length,
        current_cost=initial_cost,
        initial_cost=initial_cost,
        best_cost=initial_cost,
        stagnation_count=0,
        last_llh_idx=-1,
        last_reward=0.0,
    )
    step_history: list[LastRLStepStatus] = []
    usage: dict[str, int] = {h.name: 0 for h in llh_lib}
    policy.reset_traces()

    t0 = time.perf_counter()
    features = problem.features(sched, history)
    action = select_action(policy, features, epsilon, rng, problem.num_actions)

    for step in range(config.episode_length):
        if (
            config.wall_clock_budget_s is not None
            and time.perf_counter() - t0 >= config.wall_clock_budget_s
        ):
            break
        old_cost = problem.cost(sched)
        new_sched = llh_lib[action](sched, rng)
        new_cost = problem.cost(new_sched)
        reward = old_cost - new_cost
        sched = new_sched
        usage[llh_lib[action].name] += 1

        if new_cost < best_cost:
            best_cost = new_cost
            best_sched = list(new_sched)
            history.stagnation_count = 0
        else:
            history.stagnation_count += 1

        history.iteration = step + 1
        history.current_cost = new_cost
        history.best_cost = best_cost
        history.last_llh_idx = action
        history.last_reward = reward
        history.last_5_llh_indices = (history.last_5_llh_indices + [action])[-5:]
        history.last_5_rewards = (history.last_5_rewards + [reward])[-5:]

        next_features = problem.features(sched, history)
        next_action = select_action(
            policy, next_features, epsilon, rng, problem.num_actions
        )

        if learning:
            policy.update(
                features, action, reward, next_features, next_action,
                alpha=config.alpha, gamma=config.gamma, lam=config.lam,
            )

        step_history.append(LastRLStepStatus(
            step=step,
            llh_name=llh_lib[action].name,
            action=action,
            reward=reward,
            current_cost=new_cost,
            best_cost=best_cost,
            stagnation_count=history.stagnation_count,
        ))
        features, action = next_features, next_action

    return EpisodeResult(
        best_sched=best_sched,
        best_cost=best_cost,
        initial_cost=initial_cost,
        final_cost=history.current_cost,
        step_history=step_history,
        neighborhood_usage=usage,
        wall_clock_s=time.perf_counter() - t0,
    )


from typing import ClassVar

from ai.optimizers.base import Optimizer
from ai.optimizers.result import (
    LastRLResult,
    OptimizerConfig,
    OptimizerResult,
)


def train(
    problem: LastRLProblem,
    policy: SARSALambdaPolicy,
    config: LastRLConfig,
    rng: np.random.Generator,
) -> list[EpisodeResult]:
    """Run config.num_episodes of SARSA(λ) training; return per-episode results.

    Epsilon decays linearly from epsilon_start to epsilon_end over the run.
    """
    if config.num_episodes <= 0:
        return []
    epsilon_schedule = np.linspace(
        config.epsilon_start, config.epsilon_end, config.num_episodes
    )
    episodes: list[EpisodeResult] = []
    for ep_idx in range(config.num_episodes):
        eps = float(epsilon_schedule[ep_idx])
        result = run_episode(problem, policy, config, eps, rng, learning=True)
        episodes.append(result)
    return episodes


class LastRLOptimizer(Optimizer):
    """LAST-RL hyper-heuristic (inference-time).

    Loads a trained checkpoint and runs one greedy episode to produce a
    schedule. Training is via `python -m ai.training.last_rl`.
    """

    name: ClassVar[str] = "last_rl"
    config_class: ClassVar[type[OptimizerConfig]] = LastRLConfig
    result_class: ClassVar[type[OptimizerResult]] = LastRLResult

    def run(
        self,
        config: LastRLConfig | None = None,
        verbose: bool = False,
    ) -> LastRLResult:
        config = config or LastRLConfig()
        if config.checkpoint_path is None:
            raise ValueError(
                "last_rl requires config.checkpoint_path. "
                "Train via `python -m ai.training.last_rl train-ours` first."
            )
        raise NotImplementedError("Checkpoint loading + inference: Task 10 (Phase 4).")
