# α-fairness knob — design

> **Issue:** [#16 `[#3]` Add α-fairness knob in fitness/reward](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/16)
> **Parent:** [#13 SOTA survey](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/13)
> **Scope decision:** full-system rollout — EA fitness + RL reward + service metrics + CPSAT (egalitarian-only).

## Goal

Surface a single configurable parameter `α` controlling the fairness primitive across the AI stack, so stakeholders can dial between utilitarian (α=0), Nash welfare (α=1), Jain-equivalent (α=2, default), and Rawlsian / max-min (α→∞) without code changes. Today fairness is hard-coded to `1 − jain` in two places (`optimizers/rostering_problem.py:70`, `agents/environment.py:99`) and reported as `jain_fairness_index` in several others (`services/metrics.py:24`, `optimizers/cpsat.py:218`, `training/visualize.py:101`). Replace all of them with a single shared primitive that reduces to Jain at α=2 with bit-identical floats.

## Non-goals

- Theil / Gini / Atkinson coefficients (separate ticket if ever needed).
- CPSAT participation at finite α — CP-SAT only encodes integer-linear; only α=∞ is linearizable for our problem. CPSAT enforces `fairness_alpha=inf` via a Pydantic validator and points at NSGA-II/CCMO for other α values.
- Streamlit α-picker in `training/visualize.py` — the visualizer reports at α=2 always; users who want different α edit the script.
- Per-α HV comparison across the benchmark runner — α is part of `config_summary`, never compared blindly across α values.

## Architecture

### New module: `src/ai/domain/fairness.py`

Three public functions plus their batched counterparts.

```python
EPSILON = 1e-9  # clamp for log/inverse at x_i = 0

def alpha_fairness(values, alpha: float) -> float:
    """Generalized α-fairness welfare. Higher = more fair.

    α=0  →  Σ x_i                                  (utilitarian)
    α=1  →  Σ log(max(x_i, ε))                     (Nash welfare)
    α=2  →  (Σx)² / (n · Σx²)                       (Jain-equivalent — uses formula for float parity)
    α=∞  →  min(x_i)                                (Rawlsian / max-min)
    else →  (1/(1−α)) · Σ max(x_i, ε)^(1−α)
    """

def welfare_uniform(alpha: float, total: float, n: int) -> float:
    """alpha_fairness evaluated at the uniform distribution [total/n] × n."""

def aggregate_fairness(values, alpha: float, kind: Literal["welfare", "unfairness"]) -> float:
    """Internally derives `total = sum(values)` and `n = len(values)`.

    `kind='welfare'`    → alpha_fairness(values, α).
    `kind='unfairness'` → 1 − alpha_fairness(values, α) / welfare_uniform(α, total, n).
                          Bounded [0, 1] at α∈{2, ∞}; can exceed 1 at α=1 in adversarial cases.
                          At α=2, exactly equals `1 − jain_fairness_index(values)`."""

def alpha_fairness_batch(values: torch.Tensor, alpha: float) -> torch.Tensor: ...
def unfairness_batch(values: torch.Tensor, alpha: float) -> torch.Tensor: ...
```

Vectorized variants operate on `(B, num_employees)` tensors and stay on-device — required so the EvoTorch hot path doesn't go through Python loops.

### Backwards compatibility shim

`domain/problem.py:jain_fairness_index(values)` stays as a one-line wrapper:

```python
def jain_fairness_index(values):
    return alpha_fairness(values, alpha=2.0)
```

Bit-identical to today's implementation because the α=2 branch uses the Jain formula directly. All current call sites continue working at α=2.

### Normalization rationale

At α=2 the design uses `unfairness = 1 − welfare/welfare_uniform`, which reduces to `1 − jain` exactly. The math at uniform distribution:
- α=2: `welfare_uniform = T² / (n · n · (T/n)²) = 1`, so `unfairness = 1 − jain`. ✓
- α→∞: `welfare_uniform = T/n`, so `unfairness = 1 − n·min/T ∈ [0, 1]`. ✓
- α=1: `welfare_uniform = n · log(T/n)`; adversarial cases (one employee ≈ T, rest ≈ ε) push unfairness past 1. Documented; HV reference bumped to compensate.
- α=0: `unfairness ≡ 0` (welfare is trivially equal to welfare_uniform when total is fixed). Not useful as a fitness target — documented.

## Component changes

### 1. EA path (`optimizers/`)

**`rostering_problem.py`:**
```python
class RosteringProblem(Problem):
    def __init__(self, scheduling_problem, alpha: float = 2.0, device="cpu"):
        self._alpha = alpha
        ...

    def _evaluate_batch(self, solutions):
        ...
        # was: imbalance = 1.0 - jain
        unfairness = unfairness_batch(hours, alpha=self._alpha)
        fitnesses = torch.stack([unfairness, violations, b2b], dim=1)
```

**`optimizers/result.py`:**
```python
class EvolutionaryConfig(OptimizerConfig):
    ...
    fairness_alpha: float = 2.0   # ← new
```
`NSGAIIConfig` and `CCMOConfig` inherit it.

**`nsga2.py` / `ccmo.py`** — both pass `config.fairness_alpha` into `RosteringProblem(sp, alpha=...)`. No other change.

**`training/evolutionary.py`** — new `--fairness-alpha` CLI arg (float, default 2.0) passed through to the config.

**`api/inference.py` route `/predict/evolutionary/{algorithm}`** — new query param `fairness_alpha: float = Query(2.0, ge=0.0)` plumbed into `config_overrides`. Stakeholders dial α per-request without code changes — the issue's stated goal.

**Field renames (with one-release Pydantic alias deprecation, `populate_by_name=True`):**
| Type | Old field | New field |
|---|---|---|
| `GAStepStatus` | `mean_obj0_imbalance` | `mean_obj0_unfairness` |
| `CCMOStepStatus` | `pop1_best_imbalance` | `pop1_best_unfairness` |
| `NSGAIIFitnessResult` | `imbalance` | `unfairness` |
| `CCMOFitnessResult` | `imbalance` | `unfairness` |
| `BenchmarkRunRecord` | `best_imbalance` | `best_unfairness` |

`NSGAIITrainResult` / `CCMOTrainResult` carry the (renamed) fitness type unchanged structurally; they additionally gain a top-level `fairness_alpha: float` field for checkpoint clarity.

### 2. RL path (`agents/`, `training/rl.py`)

**`agents/environment.py`:**
```python
class EnvironmentConfig(BaseModel):
    ...
    fairness_alpha: float = 2.0   # ← new

class SchedulingEnv:
    def step(self, action):
        ...
        if self._hours.sum() > 0:
            penalty = aggregate_fairness(
                self._hours, alpha=self.config.fairness_alpha, kind="unfairness"
            )
            reward -= 0.1 * penalty
```

> **Partial-schedule note (code comment):** penalty is computed against the running total. At α=2 it's scale-invariant and matches `1 − jain` exactly. At α=1 it can spike to >>1 on early states with few employees served — intentional; α=1 incentivizes everyone-gets-something quickly.

**`training/rl.py`** — new `--fairness-alpha` arg propagates into `EnvironmentConfig`. The `model_metadata.json` snapshot includes the new field.

**`services/rl_inference.py`** — inference-time `EnvironmentConfig` uses α=2.0 (the default), not the training-time α. Rationale: reward is not observed during inference; only action-masking and the policy net matter, and neither depends on α. Plumbing training α from the checkpoint into request schemas is out of scope.

### 3. CPSAT path (`optimizers/cpsat.py`, `optimizers/result.py`)

**Rename `spread` → `fairness` internally.** The CP-SAT `IntVar` representing `h_max − h_min` is now `fairness_gap`. The objective-priority name uses `"fairness"`.

**`CPSATConfig`:**
```python
class CPSATConfig(OptimizerConfig):
    timeout_s_per_stage: float = 30.0
    num_workers: int = 8
    objective_priority: list[str] = ["b2b", "fairness"]   # ← was ["b2b", "spread"]
    fairness_alpha: float = float("inf")                  # ← new

    @field_validator("fairness_alpha")
    @classmethod
    def _validate_alpha(cls, v):
        if v != float("inf"):
            raise ValueError(
                f"CPSAT only supports egalitarian fairness (alpha=inf); got {v}. "
                "Use NSGA-II or CCMO for finite alpha values."
            )
        return v
```

The existing `objective_priority` validator updates its valid-list to `(["b2b", "fairness"], ["fairness", "b2b"])`. The "until issue #16 lands" comment is removed.

**`CPSATResult`:**
| Old field | New field |
|---|---|
| `spread: int` | `fairness_gap: int` |
| (none) | `fairness_metric: float` — α=∞ welfare = `min(hours)` |
| (none) | `fairness_alpha: float` |
| `jain_index: float` | unchanged — side metric, always at α=2 for legacy comparability |

`CPSATTrainResult` and `CPSATConfigSnapshot` in `domain/schemas.py` mirror the same rename + additions.

**`CPSATOptimizer.run` population:**
```python
fairness_gap = first_star if first_obj == "fairness" else second_star
b2b_count    = first_star if first_obj == "b2b"      else second_star
hours        = [int(solver.Value(bundle.hours[e])) for e in range(E)]
unfairness   = aggregate_fairness(hours, alpha=float("inf"), kind="unfairness")
jain         = alpha_fairness(hours, alpha=2.0)  # side metric

return CPSATResult(
    best_schedule=schedule,
    best_fitness=(unfairness, 0.0, float(b2b_count)),
    b2b_count=b2b_count,
    fairness_gap=int(fairness_gap),
    fairness_metric=float(min(hours)),
    fairness_alpha=float("inf"),
    jain_index=jain,
    stages=[...],
    total_wall_clock_s=...,
)
```

**`training/cpsat.py`** — `--objective-priority` default `"b2b,fairness"`; add `--fairness-alpha` arg accepting `"inf"` or finite floats (the validator filters).

**API route `/predict/cpsat`** — no query-param change. α is server-side only.

### 4. Service metrics (`services/metrics.py`, `services/optimizer_inference.py`, `services/rl_inference.py`)

**`compute_metrics(..., fairness_alpha: float = 2.0)`** — α flows from the optimizer/env config through the dispatch.

**`ScheduleMetrics`** in `domain/schemas.py`:
| Old field | New field |
|---|---|
| `jain_fairness_index: float` | `fairness_metric: float` — α-fairness welfare value |
| (none) | `fairness_alpha: float` |

Old name remains parseable via `Field(alias="jain_fairness_index")` + `model_config = ConfigDict(populate_by_name=True)` for one release.

**`optimizer_inference.run_optimizer_inference`** passes `getattr(config, "fairness_alpha", 2.0)` into `compute_metrics`.

**`rl_inference.run_rl_inference`** reads `config.fairness_alpha` from the `EnvironmentConfig` it builds (defaults to 2.0).

### 5. Benchmark (`benchmarks/runner.py`)

HV reference point bumped from `(1.0, 1000.0, 100.0)` to `(2.0, 1000.0, 100.0)` to cover α=1 adversarial unfairness > 1. `BenchmarkReport.config_summary` includes `fairness_alpha`. Cross-α HV comparison is undefined and the harness never aggregates across different α.

### 6. Visualize (`training/visualize.py`)

`jain_fairness_index(hours)` → `alpha_fairness(hours, alpha=2.0)`. No CLI flag added (visualizer is for quick inspection at the default α). Bit-identical floats.

## Testing strategy

**New file: `tests/ai/domain/test_fairness.py`**

| Test | Coverage |
|---|---|
| Canonical α values (0, 1, 2, ∞) on a fixed input | Each formula evaluates to the expected closed form within 1e-9 |
| Vectorized parity | `alpha_fairness_batch` matches scalar `alpha_fairness` per row, on CPU and (if available) CUDA |
| Edge cases | `n=0` (empty), `n=1` (single), all-zeros, single-non-zero, NaN guard |
| Welfare-uniform identity | At uniform distribution, `unfairness = 0` for α∈{2, ∞} |
| Jain regression | `alpha_fairness(default_hours, α=2)` equals `_legacy_jain(default_hours)` within 1e-12 |
| EPSILON behavior at x=0 | log/inverse don't NaN |

**Update existing tests:**
- `tests/ai/optimizers/test_nsga2.py`, `test_ccmo.py` — field renames; add explicit `fairness_alpha=2.0` arg in one config-build test
- `tests/ai/optimizers/test_cpsat.py` — `spread` → `fairness_gap` in assertions; new test for `CPSATConfig(fairness_alpha=2.0)` raising `ValidationError`; new test for `objective_priority=["b2b", "spread"]` raising
- `tests/ai/agents/test_environment.py` — reward at α=2 matches legacy bit-for-bit (regression guard); sanity test for α=∞ producing a different reward trajectory
- `tests/ai/services/test_cpsat_inference.py` — field-rename updates
- `tests/ai/services/test_metrics.py` — `fairness_metric` at α=2 equals legacy `jain_fairness_index` within 1e-12
- `tests/ai/training/test_benchmark_smoke.py` — `best_imbalance` → `best_unfairness`

## Documentation

- **README** — CP-SAT row: "minimize fairness gap" (was "min-max hours spread").
- **Wiki AI-Optimizers** — new "Fairness primitive" subsection; document `fairness_alpha` on `EvolutionaryConfig` + CPSAT's inf-only restriction.
- **Wiki AI-Domain** — new section for `domain/fairness.py`.
- **Wiki AI-Training** — `--fairness-alpha` flag on `evolutionary` and `rl` CLIs.
- **Wiki AI-Services** — `compute_metrics(fairness_alpha=...)` parameter.

## Commit plan (7 commits, all green between)

1. `feat(domain): add fairness module — alpha_fairness, aggregate_fairness, welfare_uniform + tests`
2. `refactor(domain): jain_fairness_index becomes wrapper around alpha_fairness(α=2)` — zero behavior change, regression guard
3. `feat(optimizers): EvolutionaryConfig.fairness_alpha; RosteringProblem accepts α; rename imbalance → unfairness across NSGAII/CCMO schemas + step history + deprecation aliases`
4. `feat(agents): EnvironmentConfig.fairness_alpha; SchedulingEnv reward uses aggregate_fairness; --fairness-alpha CLI on training/rl.py`
5. `refactor(cpsat): rename spread → fairness; CPSATConfig.fairness_alpha with inf-only validator; CPSATResult field renames; training/cpsat.py CLI update`
6. `feat(services): compute_metrics + ScheduleMetrics accept fairness_alpha; inference dispatch propagates α; visualize.py uses alpha_fairness; HV reference bumped to 2.0`
7. `docs: README + wiki (AI-Optimizers, AI-Domain, AI-Training, AI-Services) for α-fairness rollout`

## Acceptance criteria (mirrors issue #16)

- [x] α=0 → utilitarian, α=1 → log-sum, α=2 → Jain-equivalent (within 1e-6 on default instance), α→∞ → max-min — `tests/ai/domain/test_fairness.py`
- [x] Unit tests covering canonical α + numerical edge cases (`x_i=0`, `n=0`, `n=1`) — same file
- [x] Default α=2 reproduces current Jain behaviour on default instance within 1e-6 — commit 2 regression guard + multiple suite-wide bit-identical assertions
- [x] `NSGAIIConfig` and `CCMOConfig` (formerly `GAConfig`) accept `fairness_alpha` and pass it through — commit 3
- [x] RL reward path uses the same function — commit 4
- [x] Documented in wiki — commit 7

Effort estimate: 0.5–1 day per the issue, with the CPSAT additions pushing toward 1.
