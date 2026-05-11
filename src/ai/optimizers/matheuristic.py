"""Hybrid IP + VNS / SA matheuristic for shift scheduling — issue #18.

Outer loop: classical GVNS (Burke et al. EJOR 2017) cycling through three
neighborhoods at expanding size k=1..k_max. On improvement, restart at k=1.

Inner re-optimization: restricted CP-SAT lex two-stage solve on the
`optimize` slice, with `frozen` shifts bound-locked at their incumbent
values and AddHint() on free vars for warm-start.

Acceptance:
  - VNS: strict lex (cand.b2b, cand.fairness_gap) < (inc.b2b, inc.fairness_gap)
  - SA:  exp(−Δ/T) on Δ = w·(cand.b2b − inc.b2b) + (cand.fairness_gap − inc.fairness_gap)

`best` tracked separately so SA uphills never lose the best-so-far.
"""

from __future__ import annotations

import math
import time
from typing import ClassVar

import numpy as np

from ai.domain.fairness import aggregate_fairness, alpha_fairness
from ai.domain.problem import SchedulingProblem
from ai.optimizers.base import Optimizer
# Reuses optimizers/cpsat.py's semi-public model-building contract
# (_build_model, _solve_stage, CPSAT errors). Third consumer of this contract
# alongside CPSATOptimizer itself and agents/warm_start.enumerate_cpsat_optima
# (#17). Any refactor of cpsat.py must keep these symbols compatible.
from ai.optimizers.cpsat import (
    CPSATInfeasibleError,
    CPSATTimeoutError,
    _build_model,
    _solve_stage,
)
from ai.optimizers.neighbourhoods import swap_day, swap_employee, swap_shift_block
from ai.optimizers.result import (
    CPSATConfig,
    MatheuristicConfig,
    MatheuristicResult,
    MatheuristicStepStatus,
    OptimizerConfig,
    OptimizerResult,
)


class MatheuristicError(RuntimeError):
    """Raised when the matheuristic cannot construct a feasible initial solution."""


def _init_random_feasible(
    sp: SchedulingProblem,
    rng: np.random.Generator,
) -> list[int]:
    """Build a random schedule that respects unavailability hard constraints.

    For each shift, sample uniformly over the employees available on that
    day. Max-hours overruns are NOT enforced at init — the inner IP repairs
    them on the next slice (CP-SAT enforces max_hours as a hard constraint).
    If every neighborhood slice is also CP-SAT-infeasible (rare on well-posed
    instances), `best` stays the random init, which may violate max_hours.

    Raises MatheuristicError if any day has zero available employees
    (instance is unavailability-infeasible).
    """
    unavail_by_day: list[set[int]] = [set() for _ in range(sp.days)]
    for day, emp in sp.unavailability:
        unavail_by_day[day].add(emp)
    available_by_day: list[list[int]] = [
        [e for e in range(sp.num_employees) if e not in unavail_by_day[day]]
        for day in range(sp.days)
    ]
    for day, avail in enumerate(available_by_day):
        if not avail:
            raise MatheuristicError(
                f"Day {day} has zero available employees — instance is "
                f"infeasible under unavailability."
            )

    schedule = [0] * sp.num_shifts
    for t in range(sp.num_shifts):
        day = t // sp.shifts_per_day
        schedule[t] = int(rng.choice(available_by_day[day]))
    return schedule


def _inner_ip_solve(
    sp: SchedulingProblem,
    incumbent: list[int],
    frozen: list[int],
    optimize: list[int],
    time_budget_s: float,
    workers: int,
) -> list[int] | None:
    """Solve the lex two-stage IP on `optimize` with `frozen` bound-locked.

    Stage 1: lock frozen vars to their incumbent values; AddHint on free
    vars; minimize b2b_total. Stage 2: same locks + hints; minimize
    fairness_gap under b2b_total ≤ b2b★. Returns the full schedule, or
    None on infeasibility/timeout (outer loop treats this as no candidate).

    Time budget is split 50/50 across the two CP-SAT solves. Tighter splits
    (e.g. 70/30 favoring stage 2) would require a new config field; if you
    need this, surface it on MatheuristicConfig.
    """
    half_budget = time_budget_s / 2.0
    cp_config = CPSATConfig(
        timeout_s_per_stage=half_budget,
        num_workers=workers,
        objective_priority=["b2b", "fairness"],
    )

    bundle_1 = _build_model(sp)
    for t in frozen:
        bundle_1.model.Add(bundle_1.x[t][incumbent[t]] == 1)
    for t in optimize:
        bundle_1.model.AddHint(bundle_1.x[t][incumbent[t]], 1)
    bundle_1.model.Minimize(bundle_1.b2b_total)
    try:
        _, _, b2b_star, _ = _solve_stage(bundle_1, cp_config, stage="b2b")
    except (CPSATInfeasibleError, CPSATTimeoutError):
        return None

    bundle_2 = _build_model(sp)
    for t in frozen:
        bundle_2.model.Add(bundle_2.x[t][incumbent[t]] == 1)
    for t in optimize:
        bundle_2.model.AddHint(bundle_2.x[t][incumbent[t]], 1)
    bundle_2.model.Add(bundle_2.b2b_total <= b2b_star)
    bundle_2.model.Minimize(bundle_2.fairness_gap)
    try:
        solver_2, _, _, _ = _solve_stage(bundle_2, cp_config, stage="fairness")
    except (CPSATInfeasibleError, CPSATTimeoutError):
        return None

    return _extract_schedule(solver_2, bundle_2, sp)


def _extract_schedule(solver, bundle, sp: SchedulingProblem) -> list[int]:
    """Read the solved x[t][e] vars back into a flat schedule list."""
    schedule: list[int] = []
    for t in range(sp.num_shifts):
        assigned = next(
            (e for e in range(sp.num_employees) if solver.Value(bundle.x[t][e]) == 1),
            0,
        )
        schedule.append(assigned)
    return schedule


_NEIGHBORHOODS = (
    ("swap_day", swap_day),
    ("swap_shift_block", swap_shift_block),
    ("swap_employee", swap_employee),
)


# Plain-Python fast paths for the outer loop. SchedulingProblem.compute_hours
# and SchedulingProblem.count_back_to_back exist (domain/problem.py) but return
# torch tensors and pay get_device() overhead per call — wasteful inside a
# tight outer loop that processes one candidate at a time on CPU.


def _compute_b2b(schedule: list[int]) -> int:
    return sum(1 for i in range(len(schedule) - 1) if schedule[i] == schedule[i + 1])


def _compute_hours(schedule: list[int], sp: SchedulingProblem) -> list[int]:
    hours = [0] * sp.num_employees
    for t, emp in enumerate(schedule):
        hours[emp] += sp.shift_lengths[t % sp.shifts_per_day]
    return hours


def _compute_fairness_gap(schedule: list[int], sp: SchedulingProblem) -> int:
    hours = _compute_hours(schedule, sp)
    return max(hours) - min(hours)


def _dominates_lex(cand: tuple[int, int], inc: tuple[int, int]) -> bool:
    """Strict lex comparison: cand < inc."""
    return cand < inc


def _accept_vns(
    cand_b2b: int,
    cand_fair: int,
    inc_b2b: int,
    inc_fair: int,
    temperature: float,
    weight: float,
    rng: np.random.Generator,
) -> bool:
    return _dominates_lex((cand_b2b, cand_fair), (inc_b2b, inc_fair))


def _accept_sa(
    cand_b2b: int,
    cand_fair: int,
    inc_b2b: int,
    inc_fair: int,
    temperature: float,
    weight: float,
    rng: np.random.Generator,
) -> bool:
    """SA acceptance with lex-scalarized score.

    Δ = weight * (cand_b2b - inc_b2b) + (cand_fair - inc_fair)
    Accept iff Δ < 0 OR uniform(0,1) < exp(-Δ/T).

    The large `weight` (default 1000) enforces lex priority: any b2b
    improvement dominates any fairness regression at typical T.
    """
    delta = weight * (cand_b2b - inc_b2b) + (cand_fair - inc_fair)
    if delta < 0:
        return True
    if temperature <= 0.0:
        return False
    threshold = math.exp(-delta / temperature)
    return float(rng.random()) < threshold


class MatheuristicOptimizer(Optimizer):
    """Hybrid IP + VNS / SA matheuristic."""

    name: ClassVar[str] = "matheuristic"
    config_class: ClassVar[type[OptimizerConfig]] = MatheuristicConfig
    result_class: ClassVar[type[OptimizerResult]] = MatheuristicResult

    def run(
        self,
        config: MatheuristicConfig | None = None,
        verbose: bool = False,
    ) -> MatheuristicResult:
        config = config or MatheuristicConfig()
        rng = np.random.default_rng(config.seed)

        incumbent = _init_random_feasible(self._sp, rng)
        inc_b2b = _compute_b2b(incumbent)
        inc_fair = _compute_fairness_gap(incumbent, self._sp)

        best = incumbent
        best_b2b = inc_b2b
        best_fair = inc_fair

        history: list[MatheuristicStepStatus] = []
        usage: dict[str, int] = {name: 0 for name, _ in _NEIGHBORHOODS}
        total_ip_calls = 0
        total_ip_failures = 0
        total_accepted = 0
        stagnation = 0
        T = config.sa_initial_temperature
        accept_fn = _accept_vns if config.acceptance == "vns" else _accept_sa

        t0 = time.perf_counter()
        termination_reason: str | None = None
        iteration = 0

        while iteration < config.max_iterations:
            wall_now = time.perf_counter() - t0
            if wall_now >= config.time_budget_s:
                termination_reason = "time_budget"
                break
            if stagnation >= config.stagnation_limit:
                termination_reason = "stagnation"
                break

            improved_best = False
            broke = False
            for k in range(1, config.k_max + 1):
                for nbh_name, nbh_fn in _NEIGHBORHOODS:
                    frozen, optimize = nbh_fn(incumbent, self._sp, size_k=k, rng=rng)

                    ip_t0 = time.perf_counter()
                    candidate = _inner_ip_solve(
                        sp=self._sp,
                        incumbent=incumbent,
                        frozen=frozen,
                        optimize=optimize,
                        time_budget_s=config.inner_ip_time_budget_s,
                        workers=config.inner_ip_workers,
                    )
                    ip_wall = time.perf_counter() - ip_t0
                    total_ip_calls += 1
                    usage[nbh_name] += 1

                    cand_b2b = None
                    cand_fair = None
                    accepted = False

                    if candidate is None:
                        total_ip_failures += 1
                    else:
                        cand_b2b = _compute_b2b(candidate)
                        cand_fair = _compute_fairness_gap(candidate, self._sp)
                        accepted = accept_fn(
                            cand_b2b, cand_fair, inc_b2b, inc_fair, T,
                            config.sa_lex_weight_b2b, rng,
                        )
                        if accepted:
                            incumbent = candidate
                            inc_b2b = cand_b2b
                            inc_fair = cand_fair
                            total_accepted += 1
                            if _dominates_lex((cand_b2b, cand_fair), (best_b2b, best_fair)):
                                best = candidate
                                best_b2b = cand_b2b
                                best_fair = cand_fair
                                improved_best = True

                    history.append(MatheuristicStepStatus(
                        iteration=iteration,
                        neighborhood=nbh_name,
                        size_k=k,
                        accepted=accepted,
                        candidate_b2b=cand_b2b,
                        candidate_fairness_gap=cand_fair,
                        incumbent_b2b=inc_b2b,
                        incumbent_fairness_gap=inc_fair,
                        best_b2b=best_b2b,
                        best_fairness_gap=best_fair,
                        temperature=T,
                        inner_ip_wall_clock_s=ip_wall,
                        cumulative_wall_clock_s=time.perf_counter() - t0,
                    ))

                    if accepted:
                        broke = True
                        break

                    if time.perf_counter() - t0 >= config.time_budget_s:
                        broke = True
                        break

                if broke:
                    break

            if improved_best:
                stagnation = 0
            else:
                stagnation += 1
            T *= config.sa_cooling_rate
            iteration += 1

            if verbose and iteration % 5 == 0:
                print(
                    f"[matheuristic] iter={iteration} best=({best_b2b},{best_fair}) "
                    f"inc=({inc_b2b},{inc_fair}) T={T:.2f} stagn={stagnation}"
                )

        if termination_reason is None:
            termination_reason = "max_iterations"

        hours = _compute_hours(best, self._sp)
        unfairness = aggregate_fairness(hours, alpha=float("inf"), kind="unfairness")
        fairness_metric = alpha_fairness(hours, alpha=float("inf"))
        jain = alpha_fairness(hours, alpha=2.0)

        return MatheuristicResult(
            best_schedule=best,
            best_fitness=(float(unfairness), 0.0, float(best_b2b)),
            b2b_count=best_b2b,
            fairness_gap=best_fair,
            fairness_metric=float(fairness_metric),
            fairness_alpha=float("inf"),
            jain_index=jain,
            step_history=history,
            total_iterations=iteration,
            total_accepted=total_accepted,
            total_inner_ip_calls=total_ip_calls,
            total_inner_ip_failures=total_ip_failures,
            neighborhood_usage=usage,
            final_temperature=T,
            termination_reason=termination_reason,
            total_wall_clock_s=time.perf_counter() - t0,
        )
