"""CP-SAT exact baseline optimizer.

Closes issue #14. Uses OR-Tools CP-SAT with an all-hard core (one-per-shift,
unavailability, max-hours) and a two-stage lexicographic soft objective:
minimize back-to-back assignments, then minimize max-min hour spread under
the b2b optimum from stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ortools.sat.python import cp_model

from ai.domain.problem import SchedulingProblem
from ai.optimizers.base import Optimizer
from ai.optimizers.result import (
    CPSATConfig,
    CPSATResult,
    CPSATStageResult,
    OptimizerConfig,
    OptimizerResult,
)


# === Exceptions ===


class CPSATError(RuntimeError):
    """Base class for CP-SAT solver-related failures."""


class CPSATInfeasibleError(CPSATError):
    """The CP-SAT model has no feasible solution at the named stage."""

    def __init__(self, stage: str, status_name: str):
        super().__init__(
            f"CP-SAT stage {stage!r} returned {status_name}; instance is infeasible"
        )
        self.stage = stage
        self.status_name = status_name


class CPSATTimeoutError(CPSATError):
    """The CP-SAT solver did not find a solution within the time budget."""

    def __init__(self, stage: str, elapsed_s: float):
        super().__init__(
            f"CP-SAT stage {stage!r} returned UNKNOWN after {elapsed_s:.2f}s"
        )
        self.stage = stage
        self.elapsed_s = elapsed_s


_STATUS_NAME = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


# === Model bundle ===


@dataclass
class _ModelBundle:
    """A built CP-SAT model and the variable handles we'll need post-solve."""

    model: cp_model.CpModel
    x: list[list[cp_model.IntVar]]              # x[t][e]
    hours: list[cp_model.IntVar]                # hours[e]
    b2b_total: cp_model.IntVar
    fairness_gap: cp_model.IntVar               # was: spread
    h_max: cp_model.IntVar
    h_min: cp_model.IntVar


# Consumed by ai.agents.warm_start.enumerate_cpsat_optima (#17): treats _build_model,
# _make_solver, _solve_stage, and _ModelBundle.x as a semi-public contract for
# rebuilding the model with custom constraints during optimum enumeration.
def _build_model(sp: SchedulingProblem) -> _ModelBundle:
    """Build the shared CP-SAT model from a SchedulingProblem.

    Returns the model plus handles to the decision and aux variables. The
    returned model has no objective set — callers add Minimize() and any
    extra constraints (e.g. b2b <= b2b*) before solving.
    """
    model = cp_model.CpModel()
    T = sp.num_shifts
    E = sp.num_employees

    x = [[model.NewBoolVar(f"x_{t}_{e}") for e in range(E)] for t in range(T)]

    # Hard 1: exactly one employee per shift.
    for t in range(T):
        model.AddExactlyOne(x[t])

    # Hard 2: unavailability.
    for day, emp in sp.unavailability:
        for shift_in_day in range(sp.shifts_per_day):
            t = day * sp.shifts_per_day + shift_in_day
            model.Add(x[t][emp] == 0)

    # Hours and hard 3: max-hours per employee.
    hours: list[cp_model.IntVar] = []
    for e in range(E):
        h_e = model.NewIntVar(0, sp.max_hours[e], f"hours_{e}")
        model.Add(
            h_e
            == sum(
                sp.shift_lengths[t % sp.shifts_per_day] * x[t][e] for t in range(T)
            )
        )
        hours.append(h_e)

    # Back-to-back: b2b_te[t,e] = x[t,e] AND x[t+1,e] using reified booleans.
    b2b_terms: list[cp_model.IntVar] = []
    for t in range(T - 1):
        for e in range(E):
            b2b_te = model.NewBoolVar(f"b2b_{t}_{e}")
            model.AddBoolAnd([x[t][e], x[t + 1][e]]).OnlyEnforceIf(b2b_te)
            model.AddBoolOr([x[t][e].Not(), x[t + 1][e].Not()]).OnlyEnforceIf(b2b_te.Not())
            b2b_terms.append(b2b_te)

    b2b_total = model.NewIntVar(0, T - 1, "b2b_total")
    model.Add(b2b_total == sum(b2b_terms))

    # Spread = h_max - h_min.
    hours_ub = max(sp.max_hours) if sp.max_hours else 0
    h_max = model.NewIntVar(0, hours_ub, "h_max")
    h_min = model.NewIntVar(0, hours_ub, "h_min")
    model.AddMaxEquality(h_max, hours)
    model.AddMinEquality(h_min, hours)
    fairness_gap = model.NewIntVar(0, hours_ub, "fairness_gap")
    model.Add(fairness_gap == h_max - h_min)

    return _ModelBundle(
        model=model,
        x=x,
        hours=hours,
        b2b_total=b2b_total,
        fairness_gap=fairness_gap,
        h_max=h_max,
        h_min=h_min,
    )


def _make_solver(config: CPSATConfig) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.timeout_s_per_stage
    solver.parameters.num_search_workers = config.num_workers
    if config.seed is not None:
        solver.parameters.random_seed = int(config.seed)
    return solver


def _solve_stage(
    bundle: _ModelBundle,
    config: CPSATConfig,
    stage: str,
) -> tuple[cp_model.CpSolver, int, int, float]:
    """Solve a model. Returns (solver, status, objective_value, wall_clock_s).

    Raises CPSATInfeasibleError or CPSATTimeoutError on bad statuses.
    """
    solver = _make_solver(config)
    status = solver.Solve(bundle.model)
    elapsed = solver.WallTime()

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver, status, int(solver.ObjectiveValue()), elapsed
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        raise CPSATInfeasibleError(stage, _STATUS_NAME.get(status, str(status)))
    raise CPSATTimeoutError(stage, elapsed)


def _solve_b2b_only(sp: SchedulingProblem, config: CPSATConfig) -> int:
    """Test helper: solve stage 1 in isolation and return b2b*."""
    bundle = _build_model(sp)
    bundle.model.Minimize(bundle.b2b_total)
    _, _, b2b_star, _ = _solve_stage(bundle, config, stage="b2b")
    return b2b_star


# === Optimizer ===


class CPSATOptimizer(Optimizer):
    """Exact-baseline optimizer via CP-SAT lexicographic two-stage search."""

    name: ClassVar[str] = "cpsat"
    config_class: ClassVar[type[OptimizerConfig]] = CPSATConfig
    result_class: ClassVar[type[OptimizerResult]] = CPSATResult

    def run(
        self,
        config: CPSATConfig | None = None,
        verbose: bool = False,
    ) -> CPSATResult:
        config = config or CPSATConfig()
        priority = config.objective_priority
        first_obj, second_obj = priority[0], priority[1]

        # Stage 1
        bundle_1 = _build_model(self._sp)
        bundle_1.model.Minimize(self._objective_var(bundle_1, first_obj))
        solver_1, status_1, first_star, t1 = _solve_stage(bundle_1, config, stage=first_obj)

        # Stage 2 — rebuild model so the new bound is added cleanly.
        bundle_2 = _build_model(self._sp)
        bundle_2.model.Add(self._objective_var(bundle_2, first_obj) <= first_star)
        bundle_2.model.Minimize(self._objective_var(bundle_2, second_obj))
        solver_2, status_2, second_star, t2 = _solve_stage(
            bundle_2, config, stage=second_obj
        )

        schedule, hours_per_emp = self._extract(solver_2, bundle_2)

        # Map first/second back to b2b/fairness for the result fields.
        b2b_count    = first_star if first_obj == "b2b"      else second_star
        fairness_gap = first_star if first_obj == "fairness" else second_star

        from ai.domain.fairness import alpha_fairness, aggregate_fairness

        unfairness      = aggregate_fairness(hours_per_emp, alpha=float("inf"), kind="unfairness")
        fairness_metric = alpha_fairness(hours_per_emp, alpha=float("inf"))   # = min(hours)
        jain            = alpha_fairness(hours_per_emp, alpha=2.0)             # side metric

        if verbose:
            print(f"[cpsat] stage 1 ({first_obj}): {first_star}, {t1:.2f}s")
            print(f"[cpsat] stage 2 ({second_obj}): {second_star}, {t2:.2f}s")

        return CPSATResult(
            best_schedule=schedule,
            best_fitness=(unfairness, 0.0, float(b2b_count)),
            b2b_count=int(b2b_count),
            fairness_gap=int(fairness_gap),
            fairness_metric=float(fairness_metric),
            fairness_alpha=float("inf"),
            jain_index=jain,
            stages=[
                CPSATStageResult(
                    objective=first_obj,
                    status=_STATUS_NAME[status_1],
                    objective_value=first_star,
                    wall_clock_s=t1,
                ),
                CPSATStageResult(
                    objective=second_obj,
                    status=_STATUS_NAME[status_2],
                    objective_value=second_star,
                    wall_clock_s=t2,
                ),
            ],
            total_wall_clock_s=t1 + t2,
        )

    @staticmethod
    def _objective_var(bundle: _ModelBundle, name: str) -> cp_model.IntVar:
        if name == "b2b":
            return bundle.b2b_total
        if name == "fairness":
            return bundle.fairness_gap
        raise AssertionError(f"unreachable: validator should have rejected {name!r}")

    def _extract(
        self,
        solver: cp_model.CpSolver,
        bundle: _ModelBundle,
    ) -> tuple[list[int], list[int]]:
        T = self._sp.num_shifts
        E = self._sp.num_employees
        schedule: list[int] = []
        for t in range(T):
            assigned = next(
                (e for e in range(E) if solver.Value(bundle.x[t][e]) == 1),
                0,
            )
            schedule.append(assigned)
        hours_per_emp = [int(solver.Value(bundle.hours[e])) for e in range(E)]
        return schedule, hours_per_emp
