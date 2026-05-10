# CP-SAT exact baseline — design (queued)

**Status:** queued. Brainstormed and approved through Section 5/6 on 2026-05-10, then deferred behind the DEAP→EvoTorch GA refactor (separate spec). Resume here once the refactor is merged.

**Tracks:** GitHub issue [#14](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/14) (item #1 of the SOTA roadmap, parent #13).

## Goal

Add an exact CP-SAT optimizer as the ground-truth baseline. At the default size (7 employees × 30 days × 3 shifts/day) the instance is small enough to solve to optimality, giving a yardstick for the GA / RL gap.

## Decisions made during brainstorming

| Decision | Choice | Reasoning |
|---|---|---|
| Hard vs soft constraint split | All hard: one-per-shift + unavailability + max-hours. Soft lex: b2b → fairness | Cleanest CP-SAT formulation; default instance has slack so INFEASIBLE only on genuinely over-constrained inputs |
| Fairness primitive (stage 2) | Max-min spread = max(hours) − min(hours) | CP-SAT-native (linear after auxiliary vars). Reports Jain post-hoc for cross-comparison with GA |
| Optimization scheme | Lexicographic two-stage: minimize b2b, then minimize spread under b2b ≤ b2b\* | Preserves lex priority. Two solver calls, both <30s at our size |
| Optimizer interface | Duck-type peer of `GAOptimizer` (no shared `Optimizer` Protocol yet) | Defer Protocol extraction until #2 (CCMO) lands and there are 3 optimizers |

## Module layout

```
src/ai/
├── optimizers/cpsat.py              NEW — CPSATOptimizer, CPSATConfig, CPSATResult
├── services/cpsat_inference.py      NEW — run_cpsat_inference()
├── training/cpsat.py                NEW — CLI entrypoint
├── api/inference.py                 EXTEND — add /predict/cpsat route
└── domain/schemas.py                EXTEND — CPSATConfigSnapshot, CPSATTrainResult

tests/                               NEW directory at repo root
├── conftest.py                      shared fixtures
└── ai/optimizers/test_cpsat.py      NEW

pyproject.toml                       ortools added to [ai] extra; pytest to [dev]
```

No changes to: `optimizers/ga.py`, `agents/*`, `services/ga_inference.py`, `services/rl_inference.py`, `services/metrics.py`, `domain/problem.py`. Purely additive.

## CP-SAT model

**Decision variables.** `x[t, e] ∈ {0, 1}` for shift slot `t ∈ [0, num_shifts)`, employee `e ∈ [0, num_employees)`. 90 × 7 = 630 binaries on the default instance.

**Hard constraints.**

1. *Exactly-one assignment per shift.* `model.AddExactlyOne(x[t, :])` for each `t`.
2. *Unavailability.* `model.Add(x[t, e] == 0)` for each `(day, e) ∈ unavailability` and matching `t`.
3. *Max-hours per employee.* `model.Add(sum(shift_lengths[t % shifts_per_day] * x[t, e] for t) <= max_hours[e])` for each `e`.

**Auxiliary variables.**

- `hours[e] = sum(shift_lengths[t % shifts_per_day] * x[t, e] for t)`, `IntVar(0, max_hours[e])`.
- `b2b[t]` reified `AND(x[t, e], x[t+1, e])` summed over `e`. `b2b_total = sum(b2b_vars)`.
- `h_max`, `h_min` via `model.AddMaxEquality` / `model.AddMinEquality` over `hours`. `spread = h_max - h_min`.

**Stages.**

- Stage 1: `model.Minimize(b2b_total)` → returns `b2b_star`.
- Stage 2: rebuild model, add `model.Add(b2b_total <= b2b_star)`, `model.Minimize(spread)` → returns the schedule.

**Solver settings.** `parameters.max_time_in_seconds = config.timeout_s_per_stage` (default 30s). `parameters.num_search_workers = config.num_workers` (default 8). `parameters.random_seed = 42` for test determinism.

**Status mapping.** `OPTIMAL`/`FEASIBLE` → return result. `INFEASIBLE`/`MODEL_INVALID` → raise `CPSATInfeasibleError`. `UNKNOWN` → raise `CPSATTimeoutError`.

## Schemas

```python
class CPSATConfig(BaseModel):
    timeout_s_per_stage: float = 30.0
    num_workers: int = 8
    objective_priority: list[str] = ["b2b", "spread"]
    log_search_progress: bool = False

class CPSATStageResult(BaseModel):
    objective: str
    status: str                              # "OPTIMAL" | "FEASIBLE"
    objective_value: int
    wall_clock_s: float

class CPSATResult(BaseModel):
    best_schedule: list[int]
    b2b_count: int
    spread: int
    jain_index: float
    stages: list[CPSATStageResult]
    total_wall_clock_s: float

class CPSATConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    timeout_s_per_stage: float
    num_workers: int
    objective_priority: list[str]

class CPSATTrainResult(BaseModel):
    schedule: list[int]
    b2b_count: int
    spread: int
    jain_index: float
    stages: list[CPSATStageResult]
    config: CPSATConfigSnapshot
```

`objective_priority` v1 only honors `["b2b", "spread"]` (any order); other values raise `ValueError` until #3 (α-fairness) lands.

## Inference and training entrypoints

**`services/cpsat_inference.py`.**

```python
def run_cpsat_inference(
    request: SchedulingRequest,
    timeout_s_per_stage: float = 30.0,
    num_workers: int = 8,
) -> SchedulingResponse:
    problem = SchedulingProblem.from_request(request)
    optimizer = CPSATOptimizer(problem)
    config = CPSATConfig(timeout_s_per_stage=timeout_s_per_stage, num_workers=num_workers)
    try:
        result = optimizer.run(config)
    except CPSATInfeasibleError:
        raise HTTPException(status_code=422, detail="Instance is infeasible")
    except CPSATTimeoutError:
        raise HTTPException(status_code=504, detail="Solver did not find a feasible solution within budget")

    converter = ScheduleConverter(problem, request)
    assignments, hours_by_employee = converter.to_assignments(result.best_schedule)
    metrics = compute_metrics(assignments, request, hours_by_employee)
    return SchedulingResponse(schedule=assignments, metrics=metrics)
```

**`api/inference.py` — new route.**

```python
@router.post("/cpsat", response_model=SchedulingResponse)
async def predict_cpsat(
    request: SchedulingRequest,
    timeout_s_per_stage: float = Query(30.0, ge=1.0, le=300.0),
    num_workers: int = Query(8, ge=1, le=32),
) -> SchedulingResponse:
    return run_cpsat_inference(request, timeout_s_per_stage=timeout_s_per_stage, num_workers=num_workers)
```

**`training/cpsat.py` — CLI mirroring `training/ga.py`.**

```bash
python -m ai.training.cpsat
python -m ai.training.cpsat --timeout-s-per-stage 60 --num-workers 16 --output-dir checkpoints/cpsat
```

## Testing strategy

```
tests/
├── conftest.py
└── ai/
    ├── optimizers/test_cpsat.py
    └── services/test_cpsat_inference.py
```

| Test | Asserts |
|---|---|
| `test_optimal_zero_violations(tiny_problem)` | Issue #14 AC. Zero hard-constraint violations on a 3×7×2 instance |
| `test_default_instance_completes_in_budget(default_problem)` | Issue #14 AC. Returns within 60s on the canonical 7×30×3; marked `@pytest.mark.slow` |
| `test_lex_priority_b2b_then_spread` | Stage-1 b2b == b2b\*; stage-2 b2b ≤ b2b\* |
| `test_objective_priority_validation` | Unsupported priority strings raise `ValueError` |
| `test_unavailability_respected` | Unavailable employees are not assigned |
| `test_max_hours_respected` | No employee exceeds their `max_hours` |
| `test_infeasible_raises` | Over-constrained instance raises `CPSATInfeasibleError` |

Plus `tests/ai/services/test_cpsat_inference.py` — one round-trip test through `run_cpsat_inference()`.

**Tooling.** Add `pytest>=8.0`, `pytest-mock>=3.14`, `pytest-cov>=5.0` to a new `[project.optional-dependencies] dev` group. `[tool.pytest.ini_options] testpaths = ["tests"]`, `markers = ["slow: longer-running tests"]`.

## Dependencies

- Add `ortools>=9.10` to `[project.optional-dependencies] ai`.

## Resume checklist (when picking this back up)

- [ ] Confirm the EvoTorch GA refactor merged on `main`; rebase from there.
- [ ] Re-validate the duck-type interface choice now that the GA's class shape may have shifted under EvoTorch — section "Optimizer interface" may need to be re-examined.
- [ ] Re-run brainstorming Section 6 (dependencies & risks) which we did not reach in the original session.
- [ ] Continue from "Write design doc" → "Spec self-review" → "User reviews spec" → "Transition to writing-plans".

## Open items not finalized in the original brainstorm

- Whether `tests/` lives at repo root (current plan) or under `src/ai/tests/`. Decided repo root for standard pytest discovery, but worth re-confirming.
- Whether the inference path should use a per-optimizer route (`/predict/cpsat`, current plan) or a single `/predict` with `?optimizer=cpsat` query param. Current plan chosen for symmetry with `/predict/ga` and `/predict/rl`.
