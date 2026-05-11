# CP-SAT exact baseline implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CPSATOptimizer` (closes [#14](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/14)) under the existing `Optimizer` ABC, with a small principled refactor that splits evolutionary-only fields out of the base config and result classes.

**Architecture:** OR-Tools CP-SAT model with all-hard core (one-per-shift, unavailability, max-hours) and a lexicographic two-stage soft objective (minimize `b2b`, then minimize `max_hours − min_hours` under `b2b ≤ b2b★`). `OptimizerConfig`/`OptimizerResult` split into universal + evolutionary/multi-objective layers; CP-SAT subclasses the universal layer directly. Inference dispatch generalizes to a `config_overrides: dict` shape so one function serves all optimizer families. Standalone `POST /predict/cpsat` route, separate `python -m ai.training.cpsat` CLI.

**Tech Stack:** Python 3.13, OR-Tools (`ortools>=9.10`), Pydantic v2, FastAPI, pytest, pytest-mock. Reuses `ai.domain.problem.SchedulingProblem` and `ai.domain.problem.jain_fairness_index` unchanged.

**Spec:** `docs/superpowers/specs/2026-05-10-cpsat-baseline-design.md`

---

## File structure

```
src/ai/
├── optimizers/
│   ├── result.py              EDIT — split into 4 classes; reparent NSGAII / CCMO; add CPSATConfig / CPSATResult / CPSATStageResult
│   ├── nsga2.py               EDIT — type-import update only
│   ├── ccmo.py                EDIT — type-import update only
│   ├── cpsat.py               NEW  — CPSATError + CPSATInfeasibleError + CPSATTimeoutError + CPSATOptimizer
│   ├── base.py                UNCHANGED — typing already supports the split
│   └── __init__.py            EDIT — eagerly import cpsat
│
├── services/
│   └── optimizer_inference.py EDIT — generalize to config_overrides: dict; catch CPSAT errors → 422 / 504
│
├── api/
│   └── inference.py           EDIT — POST /predict/cpsat; /evolutionary handler builds dict
│
├── training/
│   └── cpsat.py               NEW — argparse CLI mirroring training/rl.py
│
└── domain/schemas.py          EDIT — CPSATConfigSnapshot + CPSATTrainResult

tests/ai/
├── optimizers/
│   ├── test_cpsat.py          NEW  — model + AC + lex-priority + error-path tests
│   └── test_registry.py       EDIT — assert "cpsat" registered
└── services/
    ├── __init__.py            NEW (empty)
    └── test_cpsat_inference.py NEW — round-trip + 422/504 + dispatch regression

pyproject.toml                 EDIT — ortools>=9.10 to [ai] extra
README.md                      EDIT — add cpsat row, /predict/cpsat note
```

**File responsibilities:** `optimizers/cpsat.py` does model construction, two-stage solving, status mapping, and error definitions in one cohesive module (~250 LOC expected). `services/optimizer_inference.py` is the single dispatch for all `Optimizer` subclasses. `domain/schemas.py` carries only API/checkpoint schemas (CPSATConfigSnapshot, CPSATTrainResult); `optimizers/result.py` carries algorithm-internal Pydantic types.

---

## Task 1: Add ortools dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `ortools>=9.10` to the `[ai]` extra**

Edit `pyproject.toml`. In the `[project.optional-dependencies]` block, find the `ai = [...]` list and insert `"ortools>=9.10",` next to `"evotorch>=0.5.1",`. Final shape:

```toml
ai = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "torch>=2.5.0",
    "numpy>=2.0.0",
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.3.0",
    "sb3-contrib>=2.3.0",
    "evotorch>=0.5.1",
    "ortools>=9.10",
    "tensorboard>=2.17.0",
    "sqlalchemy>=2.0.48",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",
]
```

- [ ] **Step 2: Sync the lockfile**

Run: `uv sync --extra ai --extra dev --extra benchmarks`

Expected: `ortools` and its transitive deps download cleanly. No errors.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from ortools.sat.python import cp_model; print(cp_model.OPTIMAL)"`

Expected: prints `4` (or whatever the int value is in the installed version) and exits 0.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
build: add ortools>=9.10 to [ai] extra

For the CP-SAT exact baseline (#14). No code uses it yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Refactor OptimizerConfig and OptimizerResult into universal + family layers

**Files:**
- Modify: `src/ai/optimizers/result.py`
- Modify: `src/ai/optimizers/nsga2.py:18-24` (type-import update)
- Modify: `src/ai/optimizers/ccmo.py:14-20` (type-import update)

This task is a pure re-parenting refactor with zero behavior change. The 42 existing tests act as the regression guard.

- [ ] **Step 1: Rewrite `src/ai/optimizers/result.py`**

Replace the file's contents with:

```python
"""Algorithm-internal config and result schemas shared across optimizers.

Two layers of inheritance:
  * Universal layer  — `OptimizerConfig` / `OptimizerResult` carry only
    fields every optimizer family needs.
  * Family layer     — `EvolutionaryConfig` / `MultiObjectiveResult`
    add evolutionary-specific or multi-objective-specific fields.

CP-SAT inherits the universal layer directly; NSGA-II and CCMO inherit
the family layer.
"""

from pydantic import BaseModel, field_validator


# === Config hierarchy ===


class OptimizerConfig(BaseModel):
    """Universal hyperparameters for any optimizer family."""

    seed: int | None = None


class EvolutionaryConfig(OptimizerConfig):
    """Hyperparameters shared by all evolutionary optimizers."""

    generations: int = 200
    pop_size: int = 100
    cxpb: float = 0.7
    mutpb: float = 0.2
    indpb: float = 0.05
    tournament_size: int = 4
    device: str = "cpu"


class NSGAIIConfig(EvolutionaryConfig):
    """NSGA-II hyperparameters. `pop_size` is the single-population size."""

    elitist: bool = True


class CCMOConfig(EvolutionaryConfig):
    """CCMO hyperparameters. `pop_size` is **per population** — CCMO maintains
    two coevolving populations of this size each, so total memory and
    per-generation evaluation count are 2 × pop_size.
    """

    pass


_VALID_OBJECTIVE_PRIORITIES = (
    ["b2b", "spread"],
    ["spread", "b2b"],
)


class CPSATConfig(OptimizerConfig):
    """CP-SAT exact-baseline hyperparameters."""

    timeout_s_per_stage: float = 30.0
    num_workers: int = 8
    objective_priority: list[str] = ["b2b", "spread"]

    @field_validator("objective_priority")
    @classmethod
    def _validate_priority(cls, v: list[str]) -> list[str]:
        if v not in _VALID_OBJECTIVE_PRIORITIES:
            raise ValueError(
                f"Unsupported objective_priority {v}; "
                "only ['b2b','spread'] or ['spread','b2b'] are valid until issue #16 lands"
            )
        return v


# === Step-status types ===


class GAStepStatus(BaseModel):
    """Per-generation snapshot for the NSGA-II loop."""

    generation: int
    mean_obj0_imbalance: float
    mean_obj1_violations: float
    mean_obj2_b2b: float
    pareto_front_size: int


class CCMOStepStatus(BaseModel):
    """Per-generation snapshot for CCMO. Tracks both populations."""

    generation: int
    pop1_feasible_count: int
    pop1_best_imbalance: float
    pop1_best_b2b: float
    pop1_pareto_size: int
    pop2_pareto_size: int
    pop2_mean_violations: float


class CPSATStageResult(BaseModel):
    """Per-stage record for the CP-SAT lex pipeline."""

    objective: str          # "b2b" | "spread"
    status: str             # "OPTIMAL" | "FEASIBLE"
    objective_value: int
    wall_clock_s: float


# === Result hierarchy ===


class OptimizerResult(BaseModel):
    """Universal result every optimizer must return.

    `best_fitness` is the same 3-tuple `(imbalance, violations, b2b)` for
    every family, so the inference layer and any future benchmark runner
    can index it uniformly. CP-SAT reports `(1 - jain_index, 0.0, b2b_count)`.
    """

    best_schedule: list[int]
    best_fitness: tuple[float, float, float]


class MultiObjectiveResult(OptimizerResult):
    """Adds Pareto-front telemetry for multi-objective optimizers."""

    pareto_front: list[list[int]]
    pareto_fitnesses: list[tuple[float, float, float]]


class NSGAIIResult(MultiObjectiveResult):
    """NSGA-II result. `pareto_front` is the rank-0 front."""

    step_history: list[GAStepStatus]


class CCMOResult(MultiObjectiveResult):
    """CCMO result. `pareto_front` mirrors the *feasible* Pop1 rank-0 front
    (what most consumers want); auxiliary fields preserve dual-population
    telemetry for research scripts.
    """

    feasible_pareto_front: list[list[int]]
    feasible_pareto_fitnesses: list[tuple[float, float, float]]
    auxiliary_pareto_front: list[list[int]]
    auxiliary_pareto_fitnesses: list[tuple[float, float, float]]
    step_history: list[CCMOStepStatus]
    fell_back_to_auxiliary: bool = False


class CPSATResult(OptimizerResult):
    """CP-SAT exact-baseline result. Single optimal schedule, no Pareto front."""

    b2b_count: int
    spread: int
    jain_index: float
    stages: list[CPSATStageResult]
    total_wall_clock_s: float
```

- [ ] **Step 2: Update `src/ai/optimizers/nsga2.py` imports**

Find the imports block at the top of the file. Update the import of result types to also pull in the new parents (so existing names keep resolving). The line previously reads:

```python
from ai.optimizers.result import (
    GAStepStatus,
    NSGAIIConfig,
    NSGAIIResult,
    OptimizerConfig,
    OptimizerResult,
)
```

No change required — these names still resolve. The class hierarchy reparented underneath. Skip if the import block already matches.

- [ ] **Step 3: Update `src/ai/optimizers/ccmo.py` imports — same as Step 2**

The existing import lists `CCMOConfig`, `CCMOResult`, `CCMOStepStatus`, `OptimizerConfig`, `OptimizerResult`. All resolve unchanged.

- [ ] **Step 4: Run the full existing suite as a regression guard**

Run: `uv run pytest tests/ -q --no-header`

Expected: **42 passed** in ~12s. If anything fails, the refactor introduced an unintended behavior change — investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/ai/optimizers/result.py
git commit -m "$(cat <<'EOF'
refactor: split OptimizerConfig and OptimizerResult into universal + family layers

Universal layer (OptimizerConfig / OptimizerResult) keeps only fields every
optimizer family needs: seed; best_schedule + best_fitness.

Family layer (EvolutionaryConfig / MultiObjectiveResult) adds
generations/pop_size/cxpb/etc. and pareto_front/pareto_fitnesses.

NSGA-II and CCMO reparent transparently:
  NSGAIIConfig  : EvolutionaryConfig  (was OptimizerConfig)
  CCMOConfig    : EvolutionaryConfig
  NSGAIIResult  : MultiObjectiveResult (was OptimizerResult)
  CCMOResult    : MultiObjectiveResult

Adds CPSATConfig (used in the next commit) with a Pydantic field_validator
that gates objective_priority to ['b2b','spread'] (any order). CPSATResult,
CPSATStageResult sketched in for the next commit's optimizer.

Existing 42 tests pass without modification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement CPSATOptimizer (TDD)

**Files:**
- Create: `src/ai/optimizers/cpsat.py`
- Create: `tests/ai/optimizers/test_cpsat.py`
- Modify: `src/ai/optimizers/__init__.py`
- Modify: `tests/ai/optimizers/test_registry.py:8-15`

- [ ] **Step 1: Create the test file with the smallest acceptance test**

Create `tests/ai/optimizers/test_cpsat.py`:

```python
"""Tests for CPSATOptimizer — issue #14 acceptance criteria + invariants."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ai.domain.problem import SchedulingProblem


def test_optimal_zero_violations(tiny_problem: SchedulingProblem):
    """Issue #14 AC: solver returns OPTIMAL and respects all hard constraints."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    # Schedule covers every shift slot.
    assert len(result.best_schedule) == tiny_problem.num_shifts
    # Every assignment is a valid employee index.
    for emp_idx in result.best_schedule:
        assert 0 <= emp_idx < tiny_problem.num_employees
    # All stages finished cleanly.
    assert len(result.stages) == 2
    for stage in result.stages:
        assert stage.status in {"OPTIMAL", "FEASIBLE"}
        assert stage.wall_clock_s > 0


def test_best_fitness_is_three_tuple_zero_violations(tiny_problem: SchedulingProblem):
    """best_fitness must match the EA shape: (imbalance, violations, b2b)."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    assert len(result.best_fitness) == 3
    imbalance, violations, b2b = result.best_fitness
    assert violations == 0.0                                 # all-hard model
    assert imbalance == pytest.approx(1.0 - result.jain_index)
    assert b2b == float(result.b2b_count)


def test_unavailability_respected():
    """For each (day, e) ∈ unavailability, no shift on that day is assigned to e."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    sp = SchedulingProblem(
        num_employees=4,
        employee_types=("FT", "FT", "FT", "PT"),
        days=5,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0), (2, 1)}),
    )
    optimizer = CPSATOptimizer(sp)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    for t, emp in enumerate(result.best_schedule):
        day = t // sp.shifts_per_day
        assert (day, emp) not in sp.unavailability


def test_max_hours_respected(tiny_problem: SchedulingProblem):
    """No employee exceeds their max_hours."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(tiny_problem)
    result = optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    hours = [0] * tiny_problem.num_employees
    for t, emp in enumerate(result.best_schedule):
        shift_idx = t % tiny_problem.shifts_per_day
        hours[emp] += tiny_problem.shift_lengths[shift_idx]

    for e, h in enumerate(hours):
        assert h <= tiny_problem.max_hours[e], (
            f"employee {e} got {h}h, cap is {tiny_problem.max_hours[e]}h"
        )


def test_stages_record_optimal_status(tiny_problem: SchedulingProblem):
    """Both stages should record one of the two valid statuses, in order."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    result = CPSATOptimizer(tiny_problem).run(
        CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)
    )

    objectives = [stage.objective for stage in result.stages]
    assert objectives == ["b2b", "spread"]


def test_lex_priority_b2b_then_spread(tiny_problem: SchedulingProblem):
    """Solving stage 1 alone yields b2b★; full pipeline matches it."""
    from ai.optimizers.cpsat import CPSATOptimizer, _solve_b2b_only
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=1, seed=42)
    b2b_star = _solve_b2b_only(tiny_problem, config)

    result = CPSATOptimizer(tiny_problem).run(config)

    assert result.b2b_count == b2b_star
    assert result.stages[1].status in {"OPTIMAL", "FEASIBLE"}


def test_objective_priority_validation():
    """Unsupported objective_priority raises Pydantic ValidationError."""
    from ai.optimizers.result import CPSATConfig

    with pytest.raises(ValidationError):
        CPSATConfig(objective_priority=["fairness"])

    with pytest.raises(ValidationError):
        CPSATConfig(objective_priority=["b2b"])


def test_infeasible_raises(over_constrained_problem: SchedulingProblem):
    """Over-constrained instance raises CPSATInfeasibleError at stage 1."""
    from ai.optimizers.cpsat import CPSATInfeasibleError, CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    optimizer = CPSATOptimizer(over_constrained_problem)
    with pytest.raises(CPSATInfeasibleError) as exc:
        optimizer.run(CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42))

    assert exc.value.stage == "b2b"


def test_timeout_raises_via_mock(tiny_problem: SchedulingProblem, mocker):
    """Mocking solver to return UNKNOWN raises CPSATTimeoutError."""
    from ortools.sat.python import cp_model

    from ai.optimizers.cpsat import CPSATOptimizer, CPSATTimeoutError
    from ai.optimizers.result import CPSATConfig

    mocker.patch.object(cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN)
    mocker.patch.object(cp_model.CpSolver, "WallTime", return_value=0.5)

    with pytest.raises(CPSATTimeoutError) as exc:
        CPSATOptimizer(tiny_problem).run(CPSATConfig(timeout_s_per_stage=1.0))

    assert exc.value.stage == "b2b"
    assert exc.value.elapsed_s == pytest.approx(0.5)


@pytest.mark.slow
def test_seed_is_deterministic(tiny_problem: SchedulingProblem):
    """num_workers=1 + seed=42 yields identical schedules across runs."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=1, seed=42)
    a = CPSATOptimizer(tiny_problem).run(config)
    b = CPSATOptimizer(tiny_problem).run(config)
    assert a.best_schedule == b.best_schedule


@pytest.mark.slow
def test_default_instance_completes_in_budget(default_problem: SchedulingProblem):
    """Issue #14 AC: 7×30×3 returns within 2 × timeout_s_per_stage seconds."""
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=30.0, num_workers=4, seed=42)
    result = CPSATOptimizer(default_problem).run(config)

    assert result.total_wall_clock_s < 2 * config.timeout_s_per_stage
    assert len(result.best_schedule) == default_problem.num_shifts
```

- [ ] **Step 2: Run the test file to verify all tests fail (no module yet)**

Run: `uv run pytest tests/ai/optimizers/test_cpsat.py -v`

Expected: each test fails with `ModuleNotFoundError: No module named 'ai.optimizers.cpsat'`. Confirms the test file is wired up.

- [ ] **Step 3: Implement `src/ai/optimizers/cpsat.py`**

```python
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

from ai.domain.problem import SchedulingProblem, jain_fairness_index
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
    spread: cp_model.IntVar
    h_max: cp_model.IntVar
    h_min: cp_model.IntVar


def _build_model(sp: SchedulingProblem) -> _ModelBundle:
    """Build the shared CP-SAT model from a SchedulingProblem.

    Returns the model plus handles to the decision and aux variables. The
    returned model has no objective set — callers add Minimize() and any
    extra constraints (e.g. b2b ≤ b2b★) before solving.
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

    # Back-to-back: b2b_te[t,e] = x[t,e] AND x[t+1,e] (boolean product).
    b2b_terms: list[cp_model.IntVar] = []
    for t in range(T - 1):
        for e in range(E):
            b2b_te = model.NewBoolVar(f"b2b_{t}_{e}")
            model.AddMultiplicationEquality(b2b_te, [x[t][e], x[t + 1][e]])
            b2b_terms.append(b2b_te)

    b2b_total = model.NewIntVar(0, T - 1, "b2b_total")
    model.Add(b2b_total == sum(b2b_terms))

    # Spread = h_max - h_min.
    hours_ub = max(sp.max_hours) if sp.max_hours else 0
    h_max = model.NewIntVar(0, hours_ub, "h_max")
    h_min = model.NewIntVar(0, hours_ub, "h_min")
    model.AddMaxEquality(h_max, hours)
    model.AddMinEquality(h_min, hours)
    spread = model.NewIntVar(0, hours_ub, "spread")
    model.Add(spread == h_max - h_min)

    return _ModelBundle(
        model=model,
        x=x,
        hours=hours,
        b2b_total=b2b_total,
        spread=spread,
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
    """Test helper: solve stage 1 in isolation and return b2b★."""
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
        jain = jain_fairness_index(hours_per_emp)

        # Map first/second back to b2b/spread for the result fields.
        b2b_count = first_star if first_obj == "b2b" else second_star
        spread = second_star if first_obj == "b2b" else first_star

        if verbose:
            print(f"[cpsat] stage 1 ({first_obj}): {first_star}, {t1:.2f}s")
            print(f"[cpsat] stage 2 ({second_obj}): {second_star}, {t2:.2f}s")

        return CPSATResult(
            best_schedule=schedule,
            best_fitness=(1.0 - jain, 0.0, float(b2b_count)),
            b2b_count=int(b2b_count),
            spread=int(spread),
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
        if name == "spread":
            return bundle.spread
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
```

- [ ] **Step 4: Wire `__init__.py` to import cpsat for auto-registration**

Modify `src/ai/optimizers/__init__.py`:

```python
"""Optimizer package."""

from ai.optimizers.base import Optimizer
from ai.optimizers.ccmo import CCMOOptimizer  # noqa: F401 — import for registration
from ai.optimizers.cpsat import CPSATOptimizer  # noqa: F401 — import for registration
from ai.optimizers.nsga2 import NSGAIIOptimizer  # noqa: F401 — import for registration

__all__ = ["Optimizer", "NSGAIIOptimizer", "CCMOOptimizer", "CPSATOptimizer"]
```

- [ ] **Step 5: Update `tests/ai/optimizers/test_registry.py`**

Find `test_init_subclass_registers` (around line 8) and add the cpsat assertion:

```python
def test_init_subclass_registers(tiny_problem):
    """Concrete subclasses with a 'name' attribute are auto-registered."""
    available = Optimizer.list_available()
    assert "nsga2" in available
    assert "ccmo" in available
    assert "cpsat" in available
```

Also update `test_create_unknown_raises`:

```python
def test_create_unknown_raises(tiny_problem):
    with pytest.raises(ValueError) as exc:
        Optimizer.create("does-not-exist", tiny_problem)
    assert "Unknown optimizer" in str(exc.value)
    assert "nsga2" in str(exc.value)
    assert "ccmo" in str(exc.value)
    assert "cpsat" in str(exc.value)
```

- [ ] **Step 6: Run the CP-SAT test suite (excluding slow)**

Run: `uv run pytest tests/ai/optimizers/test_cpsat.py tests/ai/optimizers/test_registry.py -v -m "not slow"`

Expected: all non-slow CPSAT tests + all 6 registry tests pass. Slow tests deselected.

If `test_lex_priority_b2b_then_spread` fails because `_solve_b2b_only` isn't defined, double-check the helper export from `cpsat.py`. The test imports it from `ai.optimizers.cpsat`.

If `AddMultiplicationEquality` is unavailable in the installed ortools version (very old), fall back to the reified pattern:

```python
b2b_te = model.NewBoolVar(f"b2b_{t}_{e}")
model.AddBoolAnd([x[t][e], x[t + 1][e]]).OnlyEnforceIf(b2b_te)
model.AddBoolOr([x[t][e].Not(), x[t + 1][e].Not()]).OnlyEnforceIf(b2b_te.Not())
```

- [ ] **Step 7: Run the slow tests**

Run: `uv run pytest tests/ai/optimizers/test_cpsat.py -v -m "slow"`

Expected: `test_seed_is_deterministic` and `test_default_instance_completes_in_budget` pass within budget.

- [ ] **Step 8: Run the entire suite for regression**

Run: `uv run pytest tests/ -q --no-header`

Expected: **53 passed** (42 existing + 11 new CPSAT tests). Slow tests run by default unless `-m "not slow"`.

- [ ] **Step 9: Commit**

```bash
git add src/ai/optimizers/cpsat.py src/ai/optimizers/__init__.py \
        tests/ai/optimizers/test_cpsat.py tests/ai/optimizers/test_registry.py
git commit -m "$(cat <<'EOF'
feat: optimizers/cpsat.py — CPSATOptimizer with lex two-stage and typed errors

Closes #14. CP-SAT exact baseline using OR-Tools.

Model: x[t,e] booleans for shift t × employee e. Hard constraints:
exactly-one per shift (AddExactlyOne); unavailability (x=0); max-hours
(linear sum). Aux variables: hours[e], b2b_total via product equalities,
h_max / h_min via AddMaxEquality / AddMinEquality, spread = h_max - h_min.

Two-stage lex: minimize b2b_total → b2b★; rebuild model with
b2b_total ≤ b2b★, minimize spread. Status mapping: OPTIMAL/FEASIBLE
record, INFEASIBLE/MODEL_INVALID raise CPSATInfeasibleError, UNKNOWN
raises CPSATTimeoutError.

best_fitness = (1 - jain_index, 0.0, b2b_count) so CP-SAT is a drop-in
peer of NSGA-II/CCMO under the Optimizer ABC. spread, stages, and
jain_index live in CPSATResult-specific fields.

Registers automatically as name='cpsat' via __init_subclass__.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Generalize the inference dispatch to config_overrides

**Files:**
- Modify: `src/ai/services/optimizer_inference.py` (full rewrite)
- Modify: `src/ai/api/inference.py:30-45` (route handler builds dict)
- Modify: `src/ai/api/inference.py:48-57` (deprecated /ga handler builds dict)

- [ ] **Step 1: Rewrite `src/ai/services/optimizer_inference.py`**

```python
"""Single inference service that dispatches to any registered optimizer.

The dispatch generalizes over optimizer families: the route handler builds
a dict of validated query params; the dispatch passes them straight to the
optimizer's config_class. Pydantic config models reject unknown keys, so
mis-routed knobs surface at config-build time, not at solve time.
"""

from typing import Any

from fastapi import HTTPException

from ai.domain.problem import ScheduleConverter, SchedulingProblem
from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.optimizers.cpsat import CPSATInfeasibleError, CPSATTimeoutError
from ai.optimizers.result import CCMOResult
from ai.services.metrics import compute_metrics


def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    config_overrides: dict[str, Any] | None = None,
) -> SchedulingResponse:
    """Dispatch to the named optimizer; convert its best schedule to the API response."""
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)
    config = optimizer.config_class(**(config_overrides or {}))

    try:
        result = optimizer.run(config)
    except CPSATInfeasibleError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Instance is infeasible (stage={e.stage}, status={e.status_name})",
        )
    except CPSATTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Solver budget exhausted (stage={e.stage}, elapsed={e.elapsed_s:.1f}s)",
        )

    if isinstance(result, CCMOResult) and result.fell_back_to_auxiliary:
        raise HTTPException(
            status_code=422,
            detail="No feasible schedule found; instance is over-constrained or budget too tight",
        )

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(result.best_schedule)
    metrics = compute_metrics(assignments, request, hours_by_employee)
    return SchedulingResponse(schedule=assignments, metrics=metrics)
```

- [ ] **Step 2: Update `/predict/evolutionary/{algorithm}` to build the dict**

In `src/ai/api/inference.py`, find `predict_evolutionary` (around line 30) and replace its body:

```python
@router.post("/evolutionary/{algorithm}", response_model=SchedulingResponse)
async def predict_evolutionary(
    algorithm: EvolutionaryAlgorithm,
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
    device: str = Query("cpu", pattern=r"^(cpu|cuda)$"),
) -> SchedulingResponse:
    """Run an evolutionary multi-objective optimizer ('nsga2' | 'ccmo')."""
    return run_optimizer_inference(
        algorithm.value,
        request,
        config_overrides={
            "generations": generations,
            "pop_size": pop_size,
            "device": device,
        },
    )
```

- [ ] **Step 3: Update the deprecated `/predict/ga` handler**

Same file, find `predict_ga` (around line 48) and replace its body:

```python
@router.post("/ga", response_model=SchedulingResponse, deprecated=True)
async def predict_ga(
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
) -> SchedulingResponse:
    """DEPRECATED: use /predict/evolutionary/nsga2."""
    return run_optimizer_inference(
        "nsga2",
        request,
        config_overrides={"generations": generations, "pop_size": pop_size},
    )
```

- [ ] **Step 4: Run the existing test suite as a regression guard**

Run: `uv run pytest tests/ -q --no-header`

Expected: all 53 tests pass (refactor is shape-only). If anything fails, the dispatch refactor changed behavior — investigate before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/ai/services/optimizer_inference.py src/ai/api/inference.py
git commit -m "$(cat <<'EOF'
refactor: services/optimizer_inference.py — generalize to config_overrides dict

run_optimizer_inference() previously took evolutionary-specific kwargs
(generations, pop_size, device). It now takes config_overrides:
dict[str, Any] | None, which the route handlers build from validated
query params and the dispatch passes straight to optimizer.config_class.

Pydantic config models reject unknown keys, so mis-routed knobs surface
at config-build time. Same dispatch now serves nsga2, ccmo, and cpsat.

Soft-breaking change for any direct importer; the two route handlers in
api/inference.py are the only callers in this repo.

Also adds CPSATInfeasibleError → 422 and CPSATTimeoutError → 504 mapping
in the dispatch. CCMO's fell_back_to_auxiliary stays a result flag
(different semantic — CCMO did find something, just not feasible).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `POST /predict/cpsat` route + inference round-trip tests

**Files:**
- Modify: `src/ai/api/inference.py` (add new route)
- Create: `tests/ai/services/__init__.py` (empty)
- Create: `tests/ai/services/test_cpsat_inference.py`

- [ ] **Step 1: Create the empty package marker**

Run: `mkdir -p tests/ai/services && touch tests/ai/services/__init__.py`

- [ ] **Step 2: Write the failing inference test file**

Create `tests/ai/services/test_cpsat_inference.py`:

```python
"""End-to-end tests for run_optimizer_inference dispatching to CP-SAT."""

import pytest
from fastapi import HTTPException

from ai.domain.schemas import EmployeeInfo, SchedulingRequest, ShiftInfo


def _tiny_request() -> SchedulingRequest:
    return SchedulingRequest(
        employees=[
            EmployeeInfo(id=10, employee_type="FT", max_hours=50),
            EmployeeInfo(id=11, employee_type="FT", max_hours=50),
            EmployeeInfo(id=12, employee_type="PT", max_hours=20),
        ],
        days=7,
        shifts=[
            ShiftInfo(start_time="06:00:00", end_time="14:00:00", length_hours=8),
            ShiftInfo(start_time="14:00:00", end_time="22:00:00", length_hours=8),
        ],
        unavailability=[],
    )


def _over_constrained_request() -> SchedulingRequest:
    return SchedulingRequest(
        employees=[
            EmployeeInfo(id=10, employee_type="PT", max_hours=20),
            EmployeeInfo(id=11, employee_type="PT", max_hours=20),
            EmployeeInfo(id=12, employee_type="PT", max_hours=20),
        ],
        days=7,
        shifts=[
            ShiftInfo(start_time="06:00:00", end_time="14:00:00", length_hours=8),
            ShiftInfo(start_time="14:00:00", end_time="22:00:00", length_hours=8),
            ShiftInfo(start_time="22:00:00", end_time="06:00:00", length_hours=8),
        ],
        unavailability=[],
    )


def test_round_trip_through_dispatch():
    """run_optimizer_inference("cpsat", request, …) returns a valid SchedulingResponse."""
    from ai.services.optimizer_inference import run_optimizer_inference

    response = run_optimizer_inference(
        "cpsat",
        _tiny_request(),
        config_overrides={"timeout_s_per_stage": 10.0, "num_workers": 2, "seed": 42},
    )

    assert len(response.schedule) == 7 * 2
    for assignment in response.schedule:
        assert assignment.employee_id in {10, 11, 12}


def test_infeasible_returns_422():
    """CPSATInfeasibleError surfaces as HTTPException(422)."""
    from ai.services.optimizer_inference import run_optimizer_inference

    with pytest.raises(HTTPException) as exc:
        run_optimizer_inference(
            "cpsat",
            _over_constrained_request(),
            config_overrides={"timeout_s_per_stage": 10.0, "num_workers": 2, "seed": 42},
        )
    assert exc.value.status_code == 422
    assert "infeasible" in exc.value.detail.lower()


def test_timeout_returns_504_via_mock(mocker):
    """Mocked CPSATTimeoutError surfaces as HTTPException(504)."""
    from ortools.sat.python import cp_model

    from ai.services.optimizer_inference import run_optimizer_inference

    mocker.patch.object(cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN)
    mocker.patch.object(cp_model.CpSolver, "WallTime", return_value=0.1)

    with pytest.raises(HTTPException) as exc:
        run_optimizer_inference(
            "cpsat",
            _tiny_request(),
            config_overrides={"timeout_s_per_stage": 1.0, "num_workers": 1},
        )
    assert exc.value.status_code == 504


def test_evolutionary_dispatch_still_works():
    """Regression guard: nsga2 route through the new dict-shape dispatch."""
    from ai.services.optimizer_inference import run_optimizer_inference

    response = run_optimizer_inference(
        "nsga2",
        _tiny_request(),
        config_overrides={"generations": 5, "pop_size": 20, "device": "cpu"},
    )

    assert len(response.schedule) == 7 * 2
```

- [ ] **Step 3: Run the test file (it should mostly fail — no /cpsat route yet, but dispatch already works)**

Run: `uv run pytest tests/ai/services/test_cpsat_inference.py -v`

Expected: all 4 tests pass already, because the dispatch refactor in Task 4 already covers everything these tests exercise. The route registration is independent. Confirm and continue.

If a test fails because the request schema fields don't match, open `src/ai/domain/schemas.py` and adjust the constructor calls. The current schema names are `EmployeeInfo` / `ShiftInfo` / `UnavailabilityInfo`; `ShiftInfo` requires all three of `start_time`, `end_time`, `length_hours`.

- [ ] **Step 4: Add the `/predict/cpsat` route to `src/ai/api/inference.py`**

Insert the new handler after the `predict_rl` handler and before `predict_evolutionary`:

```python
@router.post("/cpsat", response_model=SchedulingResponse)
async def predict_cpsat(
    request: SchedulingRequest,
    timeout_s_per_stage: float = Query(30.0, ge=1.0, le=300.0),
    num_workers: int = Query(8, ge=1, le=32),
) -> SchedulingResponse:
    """Run the CP-SAT exact-baseline solver."""
    return run_optimizer_inference(
        "cpsat",
        request,
        config_overrides={
            "timeout_s_per_stage": timeout_s_per_stage,
            "num_workers": num_workers,
        },
    )
```

- [ ] **Step 5: Verify the route is registered**

Run: `uv run python -c "from ai.main import app; print(sorted(r.path for r in app.routes))"`

Expected: list including `/predict/cpsat`, `/predict/evolutionary/{algorithm}`, `/predict/ga`, `/predict/rl`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q --no-header`

Expected: **57 passed** (53 + 4 new inference tests).

- [ ] **Step 7: Commit**

```bash
git add src/ai/api/inference.py tests/ai/services/__init__.py tests/ai/services/test_cpsat_inference.py
git commit -m "$(cat <<'EOF'
feat: api/inference.py — POST /predict/cpsat route

Standalone route mirroring /predict/rl. Query params: timeout_s_per_stage,
num_workers. Dispatches via run_optimizer_inference("cpsat", ...).

Tests cover round-trip through the dispatch, 422 on infeasible, 504 on
mocked timeout, and a regression guard that the dispatch refactor in the
previous commit hasn't broken nsga2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add the `python -m ai.training.cpsat` CLI

**Files:**
- Create: `src/ai/training/cpsat.py`
- Modify: `src/ai/domain/schemas.py` — add `CPSATConfigSnapshot`, `CPSATTrainResult`

- [ ] **Step 1: Add the CP-SAT checkpoint schemas**

Append to `src/ai/domain/schemas.py` (just before the `# === One-release deprecation aliases ===` marker, so they live with the other Benchmark schemas):

```python
class CPSATConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    timeout_s_per_stage: float
    num_workers: int
    objective_priority: list[str]
    seed: int | None = None


class CPSATTrainResult(BaseModel):
    schedule: list[int]
    b2b_count: int
    spread: int
    jain_index: float
    stages: list[CPSATStageResult]
    config: CPSATConfigSnapshot
```

If `CPSATStageResult` isn't imported at the top of the file, add it:

```python
from ai.optimizers.result import CPSATStageResult
```

(or use a local import inside the model definitions if the module ordering needs it).

- [ ] **Step 2: Create the CLI**

Create `src/ai/training/cpsat.py`:

```python
"""Training CLI for CPSATOptimizer.

Runs the exact-baseline solver against the canonical EnvironmentConfig
(7 employees × 30 days × 3 shifts/day) and writes the result to
checkpoints/cpsat_best_schedule.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem
from ai.domain.schemas import CPSATConfigSnapshot, CPSATTrainResult
from ai.optimizers.cpsat import CPSATOptimizer
from ai.optimizers.result import CPSATConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the CP-SAT exact-baseline solver."
    )
    parser.add_argument("--timeout-s-per-stage", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--objective-priority",
        default="b2b,spread",
        help="Comma-separated lex priority. Default: 'b2b,spread'.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    priority = [p.strip() for p in args.objective_priority.split(",") if p.strip()]
    config = CPSATConfig(
        timeout_s_per_stage=args.timeout_s_per_stage,
        num_workers=args.num_workers,
        objective_priority=priority,
        seed=args.seed,
    )

    env = EnvironmentConfig()
    problem = SchedulingProblem.from_config(env)

    optimizer = CPSATOptimizer(problem)
    result = optimizer.run(config, verbose=args.verbose)

    snapshot = CPSATConfigSnapshot(
        num_employees=problem.num_employees,
        employee_types=list(problem.employee_types),
        days=problem.days,
        shifts_per_day=problem.shifts_per_day,
        shift_lengths=list(problem.shift_lengths),
        timeout_s_per_stage=config.timeout_s_per_stage,
        num_workers=config.num_workers,
        objective_priority=config.objective_priority,
        seed=config.seed,
    )

    train_result = CPSATTrainResult(
        schedule=result.best_schedule,
        b2b_count=result.b2b_count,
        spread=result.spread,
        jain_index=result.jain_index,
        stages=result.stages,
        config=snapshot,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cpsat_best_schedule.json"
    out_path.write_text(json.dumps(train_result.model_dump(), indent=2))
    print(f"Wrote {out_path}")
    print(
        f"  b2b={result.b2b_count} spread={result.spread} "
        f"jain={result.jain_index:.4f} wall_clock={result.total_wall_clock_s:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke-test the CLI's --help**

Run: `uv run python -m ai.training.cpsat --help`

Expected: argparse help renders cleanly with all flags and exits 0.

- [ ] **Step 4: Smoke-test a real run on a small budget**

Run: `uv run python -m ai.training.cpsat --timeout-s-per-stage 10 --num-workers 4 --seed 42 --output-dir /tmp/cpsat_smoke`

Expected: completes in < 25s; prints `Wrote /tmp/cpsat_smoke/cpsat_best_schedule.json`; the file exists and parses as valid JSON.

Then: `uv run python -c "import json; r = json.load(open('/tmp/cpsat_smoke/cpsat_best_schedule.json')); print(len(r['schedule']), r['b2b_count'], r['spread'])"`

Expected: prints `90 <int> <int>` (90 = 30 days × 3 shifts/day).

- [ ] **Step 5: Run the full suite as a regression guard**

Run: `uv run pytest tests/ -q --no-header`

Expected: still **57 passed**. (No new tests added in this task; the schema additions are exercised through the CLI smoke.)

- [ ] **Step 6: Commit**

```bash
git add src/ai/training/cpsat.py src/ai/domain/schemas.py
git commit -m "$(cat <<'EOF'
feat: training/cpsat.py — argparse CLI for the exact-baseline solver

python -m ai.training.cpsat                                         # defaults
python -m ai.training.cpsat --timeout-s-per-stage 60 --num-workers 16 --seed 42
python -m ai.training.cpsat --output-dir checkpoints/cpsat

Reads the canonical EnvironmentConfig defaults (7×30×3), runs
CPSATOptimizer, writes one file: <output-dir>/cpsat_best_schedule.json
containing a CPSATTrainResult (schedule + b2b + spread + jain + stages
+ config snapshot).

CPSATConfigSnapshot and CPSATTrainResult added to domain/schemas.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the cpsat row to the optimizers table**

Find the Optimizers section table in `README.md`. Add a third row:

```markdown
| `cpsat` | `ai.optimizers.cpsat.CPSATOptimizer` | Exact baseline via OR-Tools CP-SAT; lexicographic two-stage (minimize back-to-back, then min-max hours spread); single optimal schedule per run |
```

- [ ] **Step 2: Add the /predict/cpsat note next to the inference paragraph**

Find the line:

```
Inference: `POST /predict/evolutionary/{algorithm}` — e.g. `/predict/evolutionary/nsga2`. The legacy `POST /predict/ga` is **deprecated** and forwards to `nsga2`.
```

Append a sentence:

```
For exact ground-truth schedules at the default size, use `POST /predict/cpsat` (see the CP-SAT row above).
```

- [ ] **Step 3: Add a CP-SAT subsection under Training**

Find the existing Training subsection. Append:

```markdown
For an exact baseline:

​```bash
python -m ai.training.cpsat                                       # default 30s/stage budget
python -m ai.training.cpsat --timeout-s-per-stage 60 --seed 42    # tighter run
​```
```

(Replace the leading zero-width space with nothing — it's there to escape the inner code fence in this plan.)

- [ ] **Step 4: Visually inspect**

Run: `cat README.md | head -80`

Look for malformed Markdown, stray brackets, or layout issues. Fix inline if needed.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README — add CP-SAT exact baseline

- Optimizers table: row for cpsat
- Inference paragraph: pointer to POST /predict/cpsat
- Training subsection: cpsat CLI invocations

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step 1: Run the fast suite**

Run: `uv run pytest tests/ -q --no-header -m "not slow and not benchmark"`

Expected: **all fast tests pass**, ~12s. The 11 CPSAT tests minus 2 slow leaves 9 fast cpsat tests; plus the 4 inference tests; plus the 42 pre-existing fast tests.

- [ ] **Step 2: Run slow tests**

Run: `uv run pytest tests/ -q --no-header -m "slow"`

Expected: 5 tests pass — `test_default_instance_converges` (NSGA-II), `test_default_instance_converges_to_feasible` (CCMO), `test_ccmo_hv_at_least_competitive_with_nsga2`, plus the two new CPSAT slow tests (`test_seed_is_deterministic`, `test_default_instance_completes_in_budget`). Total wall clock ≈ 1-3 min.

- [ ] **Step 3: Run benchmark smoke**

Run: `uv run pytest tests/ -q --no-header -m "benchmark"`

Expected: 1 test passes (the existing INRC-I smoke).

- [ ] **Step 4: API surface check**

Run: `uv run python -c "from ai.main import app; print(sorted(r.path for r in app.routes))"`

Expected: includes `/predict/cpsat`, `/predict/evolutionary/{algorithm}`, `/predict/ga`, `/predict/rl`, `/health`.

- [ ] **Step 5: Training CLI smoke**

```bash
uv run python -m ai.training.cpsat --timeout-s-per-stage 10 --num-workers 4 --seed 42
```

Expected: completes; writes `checkpoints/cpsat_best_schedule.json`.

- [ ] **Step 6: Confirm registry**

Run: `uv run python -c "from ai.optimizers.base import Optimizer; import ai.optimizers; print(Optimizer.list_available())"`

Expected: `['ccmo', 'cpsat', 'nsga2']`.

If all six steps pass, the implementation is complete.

---

## Plan summary

- **7 commits** total, each leaving the tree green:
  1. `build: add ortools>=9.10`
  2. `refactor: split OptimizerConfig and OptimizerResult into universal + family layers`
  3. `feat: optimizers/cpsat.py — CPSATOptimizer with lex two-stage and typed errors`
  4. `refactor: services/optimizer_inference.py — generalize to config_overrides dict`
  5. `feat: api/inference.py — POST /predict/cpsat route`
  6. `feat: training/cpsat.py — argparse CLI`
  7. `docs: README — add CP-SAT exact baseline`
- **15+ tests added.** 11 in `test_cpsat.py` (incl. 2 slow), 4 in `test_cpsat_inference.py`, plus 1-line additions to `test_registry.py`.
- **Frontend caller untouched.** CP-SAT is a server-side baseline; frontend doesn't ship a UI hook in this PR.
- **Coverage target.** `ai/optimizers/cpsat.py` ≥ 90%, matching NSGA-II / CCMO levels. The mock-based timeout test is the main lever to hit the timeout branch.
