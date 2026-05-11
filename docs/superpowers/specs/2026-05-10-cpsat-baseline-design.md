# CP-SAT exact baseline — design

**Status:** approved through brainstorming on 2026-05-10. Ready for implementation plan. Replaces an earlier draft of the same name that was queued behind the EvoTorch refactor; the EvoTorch refactor merged in PR #25 and the `Optimizer` ABC it introduced changes the integration shape, so the spec was re-brainstormed end-to-end.

**Tracks:** GitHub issue [#14](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/14) (item #1 of the SOTA roadmap, parent #13).

**Branch:** `feat/cpsat-baseline` (off `main`).

## Goals

- Add an exact CP-SAT optimizer as the ground-truth baseline. At the default size (7 employees × 30 days × 3 shifts/day) the instance is small enough to solve to optimality, giving a yardstick for the GA / RL gap.
- Slot the new optimizer cleanly under the existing `Optimizer` ABC (auto-registration via `__init_subclass__`) so every layer above (FastAPI route, training CLI, future benchmark runner) can reach it by name.
- Take a small but principled refactor of `OptimizerConfig` and `OptimizerResult` to remove evolutionary-only fields from the base — CP-SAT shouldn't carry dead `pop_size` / `generations` / `pareto_front` fields.

## Non-goals

- α-fairness knob (issue [#16](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/16)). The `objective_priority` field is shaped to accept it later, but only `["b2b", "spread"]` (any order) is honored in this PR.
- INRC-I benchmark integration. The benchmark harness in `ai.benchmarks.runner` was built around evolutionary optimizers; wiring CP-SAT into it (3-way A/B) is a separate follow-up.
- Domain-model changes (skills, contract patterns, weekend rotation). `domain/problem.py` is untouched.
- Bit-for-bit reproducibility against multi-worker CP-SAT. Multi-worker search is non-deterministic by design; deterministic tests pin `num_workers=1`.

## Decisions made during brainstorming

| Decision | Choice | Reasoning |
|---|---|---|
| Scope | Narrow — closes #14 only; defer #16 and benchmark integration | Smallest reviewable PR; principled refactor of the base contract is enough surface for one PR |
| ABC fit | Subclass `Optimizer`; refactor base into `OptimizerConfig` (universal) + `EvolutionaryConfig` (evolutionary-only) and `OptimizerResult` (universal) + `MultiObjectiveResult` | Removes dead fields from CP-SAT's view; keeps NSGA-II / CCMO unchanged behaviorally; ABC's `ClassVar[type[OptimizerConfig]]` typing still holds |
| `best_fitness` shape | Same 3-tuple `(imbalance, violations, b2b)` as EAs | CP-SAT becomes a drop-in alternative for the inference layer and any future benchmark runner; spread / Jain / stages live in CP-SAT-specific result fields |
| API route | Standalone `POST /predict/cpsat` | Mirrors `/predict/rl`; CP-SAT isn't evolutionary so it shouldn't share `/predict/evolutionary/{algorithm}`; no premature grouping under an `/exact/` family |
| Inference dispatch | Generalize `run_optimizer_inference` to `config_overrides: dict[str, Any]` | One dispatch function for all optimizers; each route handler builds its own dict from validated query params; Pydantic config models reject unknown keys |
| Training CLI | Separate `python -m ai.training.cpsat` | Mirrors `training/rl.py`; CP-SAT-specific flags (`--timeout-s-per-stage`, `--num-workers`); zero churn to `training/evolutionary.py` |
| Hard / soft constraint split | All hard: one-per-shift, unavailability, max-hours. Soft lex: b2b → spread | Cleanest CP-SAT formulation; `INFEASIBLE` only on genuinely over-constrained inputs |
| Stage-2 fairness primitive | Max-min spread = `max(hours) − min(hours)` | CP-SAT-native (linear after `AddMaxEquality` / `AddMinEquality`); Jain reported post-hoc for cross-comparison |
| Optimization scheme | Lexicographic two-stage: minimize `b2b`, then minimize `spread` under `b2b ≤ b2b★` | Preserves lex priority; two solver calls, both <30s at default size |
| Error model | Typed exceptions: `CPSATInfeasibleError`, `CPSATTimeoutError` (both inherit `CPSATError(RuntimeError)`) | CP-SAT either solves or has nothing to return; CCMO's result-flag pattern doesn't fit |

## Architecture overview

```
                        ┌──────────────────────────────────┐
                        │  Optimizer (ABC)                 │
                        │   • __init_subclass__ registers  │
                        │   • create(name, problem)        │
                        │   • list_available()             │
                        └──────────────────────────────────┘
                          │            │            │
                  ┌───────┘            │            └───────┐
                  ▼                    ▼                    ▼
        ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
        │ NSGAIIOptimizer    │ │ CCMOOptimizer      │ │ CPSATOptimizer     │
        │  name = "nsga2"    │ │  name = "ccmo"     │ │  name = "cpsat"    │
        │  config: NSGAII… ─┐│ │  config: CCMO… ───┐│ │  config: CPSAT… ──┐│
        │  result: NSGAII… ─┤│ │  result: CCMO… ───┤│ │  result: CPSAT… ──┤│
        └────────────────────┘ └────────────────────┘ └────────────────────┘
                              │                                          │
                              ▼                                          ▼
                  EvolutionaryConfig                              OptimizerConfig
                  (gen / pop / cxpb / …)                          (seed)
                  MultiObjectiveResult                            OptimizerResult
                  (pareto_front, …)                               (best_schedule, best_fitness)
```

## Module layout

```
src/ai/
├── optimizers/
│   ├── base.py                    EDIT — ABC unchanged; imports update
│   ├── result.py                  EDIT — split OptimizerConfig / OptimizerResult; add Evolutionary / MultiObjective; reparent NSGAII / CCMO subclasses
│   ├── cpsat.py                   NEW — CPSATOptimizer + CPSATError / CPSATInfeasibleError / CPSATTimeoutError
│   ├── nsga2.py                   EDIT — config_class type-hint reparent (1 line)
│   ├── ccmo.py                    EDIT — same (1 line)
│   └── __init__.py                EDIT — eagerly import cpsat for registration
│
├── services/
│   └── optimizer_inference.py     EDIT — generalize signature to config_overrides: dict; catch CPSAT errors → 422 / 504
│
├── training/
│   └── cpsat.py                   NEW — argparse CLI mirroring training/rl.py
│
├── api/
│   └── inference.py               EDIT — new POST /predict/cpsat route; /evolutionary handler builds config dict
│
├── domain/schemas.py              EDIT — CPSATStageResult, CPSATConfigSnapshot, CPSATTrainResult
│
└── main.py                        UNCHANGED

tests/ai/
├── optimizers/
│   ├── test_cpsat.py              NEW
│   └── test_registry.py           EDIT — assert "cpsat" registered
└── services/
    ├── __init__.py                NEW (empty)
    └── test_cpsat_inference.py    NEW

pyproject.toml                     EDIT — ortools>=9.10 added to [ai] extra
```

`domain/problem.py`, `agents/*`, `services/rl_inference.py`, `services/metrics.py`, `optimizers/operators.py`, `optimizers/rostering_problem.py`, `benchmarks/*`, the frontend — all unchanged.

## Optimizer base class refactor

```python
# src/ai/optimizers/result.py

from pydantic import BaseModel


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


class NSGAIIConfig(EvolutionaryConfig):       # reparent (was OptimizerConfig)
    elitist: bool = True


class CCMOConfig(EvolutionaryConfig):         # reparent
    pass


class CPSATConfig(OptimizerConfig):           # NEW
    timeout_s_per_stage: float = 30.0
    num_workers: int = 8
    objective_priority: list[str] = ["b2b", "spread"]


class OptimizerResult(BaseModel):
    """Universal result every optimizer must return."""
    best_schedule: list[int]
    best_fitness: tuple[float, float, float]   # (imbalance, violations, b2b)


class MultiObjectiveResult(OptimizerResult):
    """Adds Pareto-front telemetry for multi-objective optimizers."""
    pareto_front: list[list[int]]
    pareto_fitnesses: list[tuple[float, float, float]]


class NSGAIIResult(MultiObjectiveResult):     # reparent
    step_history: list[GAStepStatus]


class CCMOResult(MultiObjectiveResult):       # reparent
    feasible_pareto_front: list[list[int]]
    feasible_pareto_fitnesses: list[tuple[float, float, float]]
    auxiliary_pareto_front: list[list[int]]
    auxiliary_pareto_fitnesses: list[tuple[float, float, float]]
    step_history: list[CCMOStepStatus]
    fell_back_to_auxiliary: bool = False


class CPSATStageResult(BaseModel):
    objective: str          # "b2b" | "spread"
    status: str             # "OPTIMAL" | "FEASIBLE"
    objective_value: int
    wall_clock_s: float


class CPSATResult(OptimizerResult):           # NEW
    b2b_count: int
    spread: int
    jain_index: float
    stages: list[CPSATStageResult]
    total_wall_clock_s: float
```

`device` moves out of the base because CP-SAT runs on CPU only. `seed` stays universal — CP-SAT plumbs it into `parameters.random_seed`. The ABC's typing (`config_class: ClassVar[type[OptimizerConfig]]`, `result_class: ClassVar[type[OptimizerResult]]`) keeps holding for every concrete subclass.

## CP-SAT model

**Decision variables.** `x[t, e] ∈ {0, 1}` for shift slot `t ∈ [0, num_shifts)`, employee `e ∈ [0, num_employees)`. 90 × 7 = 630 binaries on the default instance.

**Hard constraints.**

1. *Exactly one assignment per shift.* `model.AddExactlyOne(x[t, :])` for each `t`.
2. *Unavailability.* `model.Add(x[t, e] == 0)` for each `(day, e) ∈ unavailability` and each `t` in that day.
3. *Max-hours per employee.* `model.Add(Σ shift_lengths[t % shifts_per_day] * x[t, e] for t) ≤ max_hours[e])` for each `e`.

**Auxiliary variables.**

```python
hours[e]   = sum(shift_lengths[t % shifts_per_day] * x[t, e] for t)   # IntVar(0, max_hours[e])
b2b[t]     = bool var, True iff x[t, e] = x[t+1, e] = 1 for some e    # AddBoolAnd reified
b2b_total  = sum(b2b[t] for t in range(num_shifts - 1))               # IntVar
h_max, h_min via AddMaxEquality / AddMinEquality over hours[:]
spread     = h_max - h_min
```

**Lexicographic two-stage.**

```python
# Stage 1
model_1, vars_1 = self._build_model()
model_1.Minimize(vars_1.b2b_total)
solver_1, status_1, b2b_star, t1 = self._solve(model_1, config, verbose, stage="b2b")

# Stage 2
model_2, vars_2 = self._build_model()
model_2.Add(vars_2.b2b_total <= b2b_star)
model_2.Minimize(vars_2.spread)
solver_2, status_2, spread_star, t2 = self._solve(model_2, config, verbose, stage="spread")

schedule, hours = self._extract(solver_2, vars_2)
jain = jain_fairness_index(hours)             # reuses src/ai/domain/problem.py:22
return CPSATResult(
    best_schedule=schedule,
    best_fitness=(1.0 - jain, 0.0, float(b2b_star)),
    b2b_count=b2b_star,
    spread=spread_star,
    jain_index=jain,
    stages=[
        CPSATStageResult(objective="b2b",    status=_name(status_1), objective_value=b2b_star,    wall_clock_s=t1),
        CPSATStageResult(objective="spread", status=_name(status_2), objective_value=spread_star, wall_clock_s=t2),
    ],
    total_wall_clock_s=t1 + t2,
)
```

`_solve()` returns `(solver, status, objective_value, wall_clock_s)`. `_extract(solver, vars)` reads `solver.Value(x)` for each decision and aux variable. `_name(status)` maps the int to `"OPTIMAL"` / `"FEASIBLE"` for the result schema.

**Solver settings.**

```python
parameters.max_time_in_seconds = config.timeout_s_per_stage    # default 30.0
parameters.num_search_workers   = config.num_workers           # default 8
if config.seed is not None:
    parameters.random_seed = config.seed
```

**Status mapping.**

| `cp_model` status | Action |
|---|---|
| `OPTIMAL`, `FEASIBLE` | record stage status, continue |
| `INFEASIBLE`, `MODEL_INVALID` | raise `CPSATInfeasibleError(stage, status_name)` |
| `UNKNOWN` | raise `CPSATTimeoutError(stage, elapsed_s)` |

`CPSATInfeasibleError` and `CPSATTimeoutError` both inherit from `CPSATError(RuntimeError)` and live in `src/ai/optimizers/cpsat.py` next to the optimizer.

**`objective_priority` validation.** `["b2b", "spread"]` and `["spread", "b2b"]` are accepted (the latter still uses lex but with stages reversed). Anything else raises `pydantic.ValidationError` via a field validator on `CPSATConfig`. This shape leaves room for #16 to add α-fairness terms later without breaking the field signature.

## Service / API / training surface

**Generalized inference dispatch.**

```python
# src/ai/services/optimizer_inference.py
from typing import Any
from fastapi import HTTPException

from ai.optimizers.base import Optimizer
from ai.optimizers.cpsat import CPSATInfeasibleError, CPSATTimeoutError
from ai.optimizers.result import CCMOResult


def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    config_overrides: dict[str, Any] | None = None,
) -> SchedulingResponse:
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)
    config = optimizer.config_class(**(config_overrides or {}))

    try:
        result = optimizer.run(config)
    except CPSATInfeasibleError as e:
        raise HTTPException(status_code=422, detail=f"Instance is infeasible (stage={e.stage})")
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

**API routes.**

```python
# src/ai/api/inference.py — additions and edits

@router.post("/cpsat", response_model=SchedulingResponse)
async def predict_cpsat(
    request: SchedulingRequest,
    timeout_s_per_stage: float = Query(30.0, ge=1.0, le=300.0),
    num_workers: int = Query(8, ge=1, le=32),
) -> SchedulingResponse:
    return run_optimizer_inference("cpsat", request, config_overrides={
        "timeout_s_per_stage": timeout_s_per_stage,
        "num_workers": num_workers,
    })


# /evolutionary handler updated to build the dict (refactor, no behavior change):
@router.post("/evolutionary/{algorithm}", response_model=SchedulingResponse)
async def predict_evolutionary(
    algorithm: EvolutionaryAlgorithm,
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
    device: str = Query("cpu", pattern=r"^(cpu|cuda)$"),
) -> SchedulingResponse:
    return run_optimizer_inference(algorithm.value, request, config_overrides={
        "generations": generations,
        "pop_size": pop_size,
        "device": device,
    })
```

`EvolutionaryAlgorithm` stays evolutionary-only — built statically as `Enum("EvolutionaryAlgorithm", {"NSGA2": "nsga2", "CCMO": "ccmo"})` — so `cpsat` doesn't accidentally appear under `/predict/evolutionary/cpsat`.

**Training CLI: `src/ai/training/cpsat.py`.**

```bash
# Default: 7×30×3 environment, 30s per stage, 8 workers
python -m ai.training.cpsat

# Custom budget & seed
python -m ai.training.cpsat --timeout-s-per-stage 60 --num-workers 16 --seed 42

# Custom output dir
python -m ai.training.cpsat --output-dir checkpoints/cpsat
```

Argparse flags: `--timeout-s-per-stage`, `--num-workers`, `--objective-priority` (default `b2b,spread`), `--seed`, `--output-dir` (default `checkpoints`).

The CLI reads the same `EnvironmentConfig` defaults `training/evolutionary.py` uses, runs `CPSATOptimizer`, and writes one file:

```
checkpoints/cpsat_best_schedule.json     # CPSATTrainResult (schedule + b2b + spread + jain + stages + config snapshot)
```

Unlike `evolutionary.py`, which separates `step_history` for plotting, CP-SAT has at most 2 stages — a single file is fine.

## Testing strategy

```
tests/ai/
├── optimizers/
│   ├── test_cpsat.py           NEW (≥ 11 tests; see table below)
│   └── test_registry.py        EDIT — assert "cpsat" in list_available()
└── services/
    ├── __init__.py             NEW
    └── test_cpsat_inference.py NEW (4 tests)
```

**`tests/ai/optimizers/test_cpsat.py`**

| Test | What it asserts | Marker |
|---|---|---|
| `test_optimal_zero_violations(tiny_problem)` | **Issue #14 AC.** `OPTIMAL`, all hard constraints satisfied. | — |
| `test_default_instance_completes_in_budget(default_problem)` | **Issue #14 AC.** Returns within `2 × timeout_s_per_stage` (60s default). | `@slow` |
| `test_best_fitness_is_three_tuple_zero_violations` | `best_fitness = (1 - jain_index, 0.0, b2b_count)`. | — |
| `test_lex_priority_b2b_then_spread` | Solve stage-1 alone for `b2b★`. Full pipeline returns `result.b2b_count == b2b★` AND stage 2 status ∈ {`OPTIMAL`, `FEASIBLE`}. | — |
| `test_objective_priority_validation` | `CPSATConfig(objective_priority=["fairness"])` raises `pydantic.ValidationError`. | — |
| `test_unavailability_respected` | Tiny instance with unavailability fixture: no shift on `(day, e) ∈ unavailability` is assigned to `e`. | — |
| `test_max_hours_respected(tiny_problem)` | For each `e`, `Σ x[t,e] · shift_lengths[…] ≤ max_hours[e]`. | — |
| `test_infeasible_raises(over_constrained_problem)` | Stage 1 raises `CPSATInfeasibleError`. | — |
| `test_timeout_raises_via_mock` | Mock `cp_model.CpSolver.Solve` → `cp_model.UNKNOWN`; assert `CPSATTimeoutError`. | — |
| `test_stages_record_optimal_status` | `result.stages` has exactly 2 entries, both `status="OPTIMAL"` with `wall_clock_s > 0`. | — |
| `test_seed_is_deterministic` | Two runs with `num_workers=1, seed=42` return identical `best_schedule`. | `@slow` |

**`tests/ai/services/test_cpsat_inference.py`**

| Test | What it asserts |
|---|---|
| `test_round_trip_through_dispatch(tiny_request)` | `run_optimizer_inference("cpsat", request, config_overrides={...})` returns a valid `SchedulingResponse`. |
| `test_infeasible_returns_422(over_constrained_request)` | `CPSATInfeasibleError` → `HTTPException(422)`. |
| `test_timeout_returns_504_via_mock` | Mocked `CPSATTimeoutError` → `HTTPException(504)`. |
| `test_evolutionary_dispatch_still_works(tiny_request)` | Regression guard: dispatch refactor doesn't break NSGA-II. |

**Determinism contract.** CP-SAT is deterministic only at `num_workers=1`. `test_seed_is_deterministic` pins workers to 1; everything else uses default 8 workers and asserts on objective values, not schedules.

**Existing 42 tests** continue to pass without modification. The base-class refactor only adds parents; concrete configs/results never lose a field.

## Dependencies (`pyproject.toml`)

```toml
[project.optional-dependencies]
ai = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "torch>=2.5.0",
    "numpy>=2.0.0",
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.3.0",
    "sb3-contrib>=2.3.0",
    "evotorch>=0.5.1",
    "ortools>=9.10",            # NEW
    "tensorboard>=2.17.0",
    "sqlalchemy>=2.0.48",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",
]
```

`pytest`, `pytest-cov`, `pytest-mock` are already in `[dev]` from the EvoTorch refactor; no additions needed for the test surface.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CP-SAT may not solve the default instance within 30s/stage on slow hardware | Test `test_default_instance_completes_in_budget` is `@slow` and uses 60s. CI nightly runs slow markers; if it ever fails we tighten or split the test instance |
| The base-class refactor (split `OptimizerConfig` / `OptimizerResult`) ripples to NSGA-II / CCMO | Refactor is mechanical — re-parent only; concrete subclasses never lose a field. Existing 42 tests act as regression guards. Land the refactor in its own commit ahead of CP-SAT for clean review |
| The dispatch shape change (`config_overrides: dict`) is a soft-breaking change for any external caller | The route handlers are the only callers in this repo; one regression test (`test_evolutionary_dispatch_still_works`) covers the shape. The plain-dict signature is also more forward-compatible than positional kwargs |
| ortools wheel availability on user platforms | `ortools>=9.10` ships wheels for Linux/macOS/Windows on x86_64 and aarch64. WSL2 (the user's primary env) is supported. Containers built from `python:3.13-slim` need `apt-get install build-essential` to build from source as a fallback |
| CP-SAT behavior at `num_workers > 1` is non-deterministic | Acceptance and lex-priority tests assert objective values, not schedules. `test_seed_is_deterministic` is the only test that pins `num_workers=1` for byte-identical comparison |
| `objective_priority` field signature might paint us into a corner for #16 (α-fairness) | The current shape (`list[str]`) admits future extensions: `["b2b", "alpha_fair"]` with a separate `alpha: float` config field. The validator is the only place that enforces `["b2b", "spread"]` and is one easy line to relax |

## Implementation sequencing (single-PR commit history)

The PR is medium-sized. Sequence commits so each leaves the tree green:

1. **`refactor: split OptimizerConfig and OptimizerResult into universal + evolutionary/multi-objective layers`** — pure re-parenting; existing tests still pass. NSGA-II / CCMO unchanged behaviorally.
2. **`build: add ortools>=9.10 to [ai] extra`** — additive only.
3. **`feat: optimizers/cpsat.py — CPSATOptimizer with lex two-stage and typed errors`** — adds module + auto-registers; `CPSATConfig` / `CPSATResult` / `CPSATStageResult` schemas.
4. **`refactor: services/optimizer_inference.py — generalize to config_overrides dict`** — dispatch shape change; updates `/evolutionary/{algorithm}` route handler in the same commit so the tree stays green.
5. **`feat: api/inference.py — POST /predict/cpsat route`** — adds new route only.
6. **`feat: training/cpsat.py — argparse CLI`** — new entrypoint.
7. **`test: tests/ai/optimizers/test_cpsat.py and tests/ai/services/test_cpsat_inference.py`** — full test surface; updates `test_registry.py` for the new name. Final state: full suite green.
8. **`docs: README — add cpsat row to optimizers table; new /predict/cpsat endpoint`** — light touch.

## Migration / breaking changes summary

- **Renamed (breaking only for direct importers):**
  - `services/optimizer_inference.run_optimizer_inference()` signature changes from positional kwargs (`generations=…, pop_size=…, device=…`) to `config_overrides: dict[str, Any] | None`. The two route handlers in the same module are the only callers.
- **No deprecated routes added or removed.**
- **New:** `POST /predict/cpsat`; `python -m ai.training.cpsat`; `Optimizer.list_available()` now includes `"cpsat"`.
- **Class hierarchy changes (transparent for instantiation):**
  - `NSGAIIConfig` and `CCMOConfig` now inherit `EvolutionaryConfig` (was `OptimizerConfig`).
  - `NSGAIIResult` and `CCMOResult` now inherit `MultiObjectiveResult` (was `OptimizerResult`).
  - `EvolutionaryConfig` and `MultiObjectiveResult` are intermediate classes; consumers that instantiate the leaf classes by name are unaffected.

## Resume checklist (handover to writing-plans)

The writing-plans skill builds the implementation plan from this spec. The plan should:

- Convert the 8-commit sequencing above into ordered tasks with explicit deliverables.
- Materialize the test plan into ordered test-first directives per file.
- Include verification commands per task (`uv run pytest tests/ai/optimizers/test_cpsat.py`, etc.).
- Flag the one place that needs a verification against the actual ortools API: the reified `b2b[t]` boolean construction (recent ortools versions may prefer `AddMultiplicationEquality` over manually-reified `AddBoolAnd`; the implementer verifies during step 3).
- Include a regression-guard test for the dispatch refactor in commit 4 so commit 4 lands green.

After this PR merges, the natural follow-ups are #16 (α-fairness, builds on `objective_priority` extension) and a 3-way INRC-I benchmark integration (extends `ai.benchmarks.runner` to handle CP-SAT's exception model).
