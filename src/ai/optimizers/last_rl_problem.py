"""LastRLProblem Protocol + concrete implementations — issue #19.

A LastRLProblem is everything the SARSA(λ) loop needs from a problem
instance: an initial solution, a scalar cost, an LLH library, and a
state-feature extractor. The algorithm in last_rl.py consumes only this
Protocol — concrete classes (RosteringLastRLProblem, PaperBenchmarkProblem)
plug in without changing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ai.domain.problem import SchedulingProblem
from ai.optimizers.llh import LowLevelHeuristic, build_llh_library


@dataclass
class SearchHistory:
    """Per-episode mutable state passed into features() by the outer loop."""

    iteration: int
    max_iterations: int
    current_cost: float
    initial_cost: float
    best_cost: float
    stagnation_count: int
    last_llh_idx: int
    last_reward: float
    last_5_llh_indices: list[int] = field(default_factory=list)
    last_5_rewards: list[float] = field(default_factory=list)


class LastRLProblem(Protocol):
    """The interface LAST-RL learns over."""

    name: str
    num_actions: int

    def initial_solution(self, rng: np.random.Generator) -> list[int]: ...

    def cost(self, solution: list[int]) -> float: ...

    def llh_library(self) -> list[LowLevelHeuristic]: ...

    def features(
        self,
        solution: list[int],
        history: SearchHistory,
    ) -> np.ndarray: ...


class RosteringLastRLProblem:
    """LastRLProblem implementation for our SchedulingProblem.

    Cost is lex-weighted scalar matching matheuristic: 1000 * violations +
    100 * b2b + fairness_gap. Features are 10-dim (7 history + 3 problem-state).
    """

    num_actions: int = 9

    def __init__(self, sp: SchedulingProblem, config):
        self._sp = sp
        self.name = f"rostering:{sp.num_employees}x{sp.days}x{sp.shifts_per_day}"
        self._llh_lib = build_llh_library(
            sp,
            ip_time_budget_s=config.ip_time_budget_s,
            ip_workers=config.ip_workers,
        )

    def initial_solution(self, rng: np.random.Generator) -> list[int]:
        from ai.optimizers.matheuristic import _init_random_feasible
        return _init_random_feasible(self._sp, rng)

    def cost(self, solution: list[int]) -> float:
        sp = self._sp
        hours = [0] * sp.num_employees
        for t, emp in enumerate(solution):
            hours[emp] += sp.shift_lengths[t % sp.shifts_per_day]

        max_hours_overrun = sum(
            max(0, hours[e] - sp.max_hours[e]) for e in range(sp.num_employees)
        )
        unavail_hits = sum(
            1 for t, emp in enumerate(solution)
            if (t // sp.shifts_per_day, emp) in sp.unavailability
        )
        violations = max_hours_overrun + 10 * unavail_hits
        b2b = sum(
            1 for i in range(len(solution) - 1) if solution[i] == solution[i + 1]
        )
        fair_gap = max(hours) - min(hours) if hours else 0
        return 1000.0 * violations + 100.0 * b2b + fair_gap

    def llh_library(self) -> list[LowLevelHeuristic]:
        return self._llh_lib

    def features(
        self, solution: list[int], history: SearchHistory
    ) -> np.ndarray:
        sp = self._sp
        hours = [0] * sp.num_employees
        for t, emp in enumerate(solution):
            hours[emp] += sp.shift_lengths[t % sp.shifts_per_day]
        b2b = sum(
            1 for i in range(len(solution) - 1) if solution[i] == solution[i + 1]
        )
        fair_gap = max(hours) - min(hours) if hours else 0
        max_hours_overrun = sum(
            max(0, hours[e] - sp.max_hours[e]) for e in range(sp.num_employees)
        )
        unavail_hits = sum(
            1 for t, emp in enumerate(solution)
            if (t // sp.shifts_per_day, emp) in sp.unavailability
        )
        violations = max_hours_overrun + 10 * unavail_hits
        total_hours = sum(hours)

        scale = max(history.initial_cost, 1.0)
        last_llh_norm = (
            (history.last_llh_idx + 1) / max(self.num_actions, 1)
            if history.last_llh_idx >= 0 else 0.0
        )
        mean_last_5 = (
            float(np.mean(history.last_5_rewards))
            if history.last_5_rewards else 0.0
        )

        return np.array([
            history.iteration / max(history.max_iterations, 1),
            history.current_cost / scale,
            (history.current_cost - history.best_cost) / scale,
            history.stagnation_count / max(history.max_iterations, 1),
            float(np.clip(history.last_reward / scale, -1.0, 1.0)),
            last_llh_norm,
            float(np.clip(mean_last_5 / scale, -1.0, 1.0)),
            b2b / max(sp.num_shifts, 1),
            fair_gap / max(total_hours, 1),
            violations / max(sp.num_shifts * 10, 1),
        ], dtype=np.float64)


# === Concrete: PaperBenchmarkProblem (sanity-check only) ===


class PaperBenchmarkProblem:
    """LastRLProblem wrapping a PaperInstance for the sanity-check tests.

    Used by `python -m ai.training.last_rl train-paper` and the slow benchmark
    test that asserts within-10pct reproduction. Not used in our normal
    inference path.
    """

    num_actions: int = 6

    def __init__(self, instance):
        self._instance = instance
        self.name = f"paper:{instance.name}"
        from ai.optimizers.llh import build_paper_llh_library
        self._llh_lib = build_paper_llh_library(instance)

    def initial_solution(self, rng: np.random.Generator) -> list[int]:
        """Random feasible: sample uniformly over employees per shift,
        respecting days_off. Standard "random feasible" paper init."""
        n = self._instance.num_days * self._instance.num_shift_types
        sched: list[int] = []
        for t in range(n):
            day = t // self._instance.num_shift_types
            available = [
                e for e in range(self._instance.num_employees)
                if day not in self._instance.days_off.get(e, set())
            ]
            if not available:
                available = list(range(self._instance.num_employees))
            sched.append(int(rng.choice(available)))
        return sched

    def cost(self, solution: list[int]) -> float:
        from ai.data.last_rl_benchmark import paper_cost
        return paper_cost(self._instance, solution)

    def llh_library(self) -> list[LowLevelHeuristic]:
        return self._llh_lib

    def features(
        self, solution: list[int], history: SearchHistory
    ) -> np.ndarray:
        """Paper §3.2 feature set: history features 0-6 (same as ours) plus
        paper-specific instance-state features 7-9."""
        scale = max(history.initial_cost, 1.0)
        last_llh_norm = (
            (history.last_llh_idx + 1) / max(self.num_actions, 1)
            if history.last_llh_idx >= 0 else 0.0
        )
        mean_last_5 = (
            float(np.mean(history.last_5_rewards))
            if history.last_5_rewards else 0.0
        )
        f7 = float(np.clip(history.current_cost / scale, 0.0, 2.0))
        f8 = history.stagnation_count / max(history.max_iterations, 1)
        f9 = float(np.clip(history.last_reward / scale, -1.0, 1.0))
        return np.array([
            history.iteration / max(history.max_iterations, 1),
            history.current_cost / scale,
            (history.current_cost - history.best_cost) / scale,
            history.stagnation_count / max(history.max_iterations, 1),
            float(np.clip(history.last_reward / scale, -1.0, 1.0)),
            last_llh_norm,
            float(np.clip(mean_last_5 / scale, -1.0, 1.0)),
            f7, f8, f9,
        ], dtype=np.float64)
