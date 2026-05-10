# EvoTorch Evolutionary Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DEAP with EvoTorch as the evolutionary-algorithm framework, rename the generic `GAOptimizer` to `NSGAIIOptimizer`, add a new `CCMOOptimizer` (Tian et al. 2021), unify both behind an `Optimizer` ABC with `__init_subclass__` auto-registration, and adopt the INRC-I sprint track as a benchmark suite.

**Architecture:** Two evolutionary optimizers (NSGA-II via EvoTorch's stock `GeneticAlgorithm`; CCMO as a manual two-population coevolution loop) share a common `Optimizer` ABC, a shared `RosteringProblem(evotorch.Problem)`, and three shared `CopyingOperator` subclasses (day-aligned crossover, per-gene uniform integer mutation, unavailability repair). Auto-registration on the ABC lets every layer (inference service, FastAPI route, training CLI, benchmark runner) accept the algorithm by name. INRC-I instances are loaded via a lossy adapter and run through a hypervolume + Wilcoxon A/B harness.

**Tech Stack:** Python 3.12+, EvoTorch ≥0.5.1, PyTorch ≥2.5, FastAPI, Pydantic v2, pymoo (hypervolume), scipy (Wilcoxon), pytest + pytest-cov, uv for dependency resolution.

**Spec:** `docs/superpowers/specs/2026-05-10-evotorch-evolutionary-optimizer-design.md`

---

## File structure

### Files created

| Path | Responsibility |
|---|---|
| `src/ai/optimizers/base.py` | `Optimizer` ABC with `__init_subclass__` registry + factory methods |
| `src/ai/optimizers/result.py` | `OptimizerConfig` / `OptimizerResult` base classes + `NSGAII*` and `CCMO*` subclasses + step-status types |
| `src/ai/optimizers/rostering_problem.py` | `RosteringProblem(evotorch.Problem)` adapter |
| `src/ai/optimizers/operators.py` | `DayAlignedCrossOver`, `UniformIntMutation`, `RepairOperator` |
| `src/ai/optimizers/nsga2.py` | `NSGAIIOptimizer` (replaces old `ga.py`) |
| `src/ai/optimizers/ccmo.py` | `CCMOOptimizer` + `_nsga2_pareto_ranks`, `_select_by_rank_and_crowding` helpers |
| `src/ai/optimizers/__init__.py` | Eager imports of concrete optimizer classes (triggers registration) |
| `src/ai/services/optimizer_inference.py` | `run_optimizer_inference(name, request, ...)` (replaces `ga_inference.py`) |
| `src/ai/training/evolutionary.py` | CLI: `--algorithm {nsga2,ccmo}` (replaces `training/ga.py`) |
| `src/ai/training/benchmark.py` | CLI for INRC-I A/B benchmark |
| `src/ai/benchmarks/__init__.py` | Empty package marker |
| `src/ai/benchmarks/inrc1/__init__.py` | Re-exports `load_instance`, `list_instances` |
| `src/ai/benchmarks/inrc1/parser.py` | `parse_inrc1_instance(text)` → `InrcInstance` intermediate form |
| `src/ai/benchmarks/inrc1/loader.py` | `load_instance(name)` → `SchedulingProblem` (lossy) |
| `src/ai/benchmarks/inrc1/manifest.json` | Sprint instance names + canonical metadata |
| `src/ai/benchmarks/runner.py` | `run_benchmark(algorithms, instances, seeds)` → `BenchmarkReport` |
| `scripts/fetch_inrc1.py` | First-use INRC-I corpus download |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/fixtures/inrc1/sprint01.txt` | Bundled tiny INRC-I fixture for parser tests |
| `tests/ai/optimizers/test_registry.py` | `Optimizer` ABC + auto-registration invariants |
| `tests/ai/optimizers/test_rostering_problem.py` | `_evaluate_batch` parity vs current `GAOptimizer.batch_fitness` |
| `tests/ai/optimizers/test_operators.py` | Unit tests per `CopyingOperator` |
| `tests/ai/optimizers/test_nsga2.py` | Convergence + structural tests |
| `tests/ai/optimizers/test_ccmo.py` | Convergence + dual-pop + fallback tests |
| `tests/ai/benchmarks/inrc1/test_parser.py` | Parser tests against fixture |
| `tests/ai/benchmarks/inrc1/test_loader.py` | Lossy adapter tests |
| `tests/ai/training/test_benchmark_smoke.py` | E2E smoke test for the runner |
| `BENCHMARKS.md` | Benchmark caveats + how-to (lossy mapping disclaimer) |

### Files modified

| Path | Change |
|---|---|
| `pyproject.toml` | Add `evotorch>=0.5.1` to `[ai]`; add new `[benchmarks]` and `[dev]` extras; remove `deap` (after Task 14) |
| `src/ai/api/inference.py` | Add `/predict/evolutionary/{algorithm}` route + `/predict/ga` deprecation shim; remove `/predict/ga` original |
| `src/ai/domain/schemas.py` | Replace `GA*` schemas with `NSGAII*` + `CCMO*` + `Benchmark*`; one-release deprecation aliases |
| `src/frontend/api_client/client.py` | Update GA route call to `/predict/evolutionary/nsga2` |
| `.gitignore` | Add `data/benchmarks/inrc1/` |
| `README.md` | Mention new endpoints + algorithm CLI |

### Files deleted

| Path | Reason |
|---|---|
| `src/ai/optimizers/ga.py` | Replaced by `optimizers/nsga2.py` |
| `src/ai/services/ga_inference.py` | Replaced by `services/optimizer_inference.py` |
| `src/ai/training/ga.py` | Replaced by `training/evolutionary.py` |

---

## Conventions used in this plan

- **All Bash commands assume cwd is repo root** (`/home/daniel/Job_Scheduler_System`).
- **Commands use `uv run`** for Python invocations to match the repo's existing pattern.
- **Test commands use `-v`** for legible output.
- **Each task ends with a commit step.** The 9-commit sequencing in the spec is preserved by grouping related tasks under each commit boundary; the section headers below indicate which commit each task contributes to.
- **Tests are written first** (TDD). Where a test depends on a class that doesn't yet exist, the failure mode is `ImportError` or `AttributeError`; the verify-fail step describes that.
- **Code blocks are the literal content** of files (or function bodies). Engineers should copy them verbatim unless a step says otherwise.

---

# Phase 1 — Build foundation

## Commit 1: pyproject.toml — add EvoTorch + dev/benchmarks groups

### Task 1: Add EvoTorch and new dependency groups (additive only — DEAP stays for now)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`

Expected: existing `[project]`, `[project.optional-dependencies]` with `backend`/`ai`/`frontend` groups, `[tool.hatch.build.targets.wheel]`, `[build-system]`.

- [ ] **Step 2: Replace pyproject.toml content**

Replace the entire file with:

```toml
[project]
name = "job-scheduler-system"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
backend = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "sqlalchemy[asyncio]>=2.0.36",
    "asyncpg>=0.30.0",
    "alembic>=1.14.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.28.0",
    "psycopg2-binary>=2.9.0",
]
ai = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "torch>=2.5.0",
    "numpy>=2.0.0",
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.3.0",
    "sb3-contrib>=2.3.0",
    "evotorch>=0.5.1",
    "deap>=1.4.1",
    "tensorboard>=2.17.0",
    "sqlalchemy>=2.0.48",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",
]
frontend = [
    "gradio>=5.0.0",
    "httpx>=0.28.0",
]
benchmarks = [
    "pymoo>=0.6.1",
    "scipy>=1.13.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
]

[tool.hatch.build.targets.wheel]
packages = [
    "src/ai",
    "src/backend",
    "src/frontend",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=ai --cov-report=term-missing"
markers = [
    "slow: longer-running tests (>10s)",
    "benchmark: tests that touch the INRC-I benchmark runner",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

(Note: `deap` is *kept* in the `ai` extra at this stage; it's removed in Task 14 once the last DEAP caller is gone.)

- [ ] **Step 3: Resolve dependencies**

Run: `uv sync --all-extras`

Expected: lockfile updates; `evotorch`, `pymoo`, `scipy`, `pytest`, `pytest-cov`, `pytest-mock` resolved without conflict. No errors.

- [ ] **Step 4: Verify EvoTorch importable**

Run: `uv run python -c "import evotorch; print(evotorch.__version__)"`

Expected: prints a version string ≥ `0.5.1`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
build: add EvoTorch, dev and benchmarks dependency groups

DEAP stays for now; it is removed in a later commit once the last DEAP
caller (the existing GAOptimizer) is replaced.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 2: Optimizer ABC with auto-registration

### Task 2: Create `tests/conftest.py` with shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create the file**

```python
"""Shared pytest fixtures for the AI service test suite."""

import pytest

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem


@pytest.fixture
def tiny_problem() -> SchedulingProblem:
    """3 employees × 7 days × 2 shifts/day. Solves in <1s."""
    return SchedulingProblem(
        num_employees=3,
        employee_types=("FT", "FT", "PT"),
        days=7,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 20),
        unavailability=frozenset(),
    )


@pytest.fixture
def default_problem() -> SchedulingProblem:
    """The canonical 7×30×3 instance from EnvironmentConfig defaults."""
    return SchedulingProblem.from_config(EnvironmentConfig())


@pytest.fixture
def over_constrained_problem() -> SchedulingProblem:
    """An instance where total demand exceeds total cap. For infeasibility tests.

    3 employees, all PT (max 20h each = 60h cap total), 7 days × 3 shifts × 8h = 168h demand.
    """
    return SchedulingProblem(
        num_employees=3,
        employee_types=("PT", "PT", "PT"),
        days=7,
        shifts_per_day=3,
        shift_lengths=(8, 8, 8),
        max_hours=(20, 20, 20),
        unavailability=frozenset(),
    )
```

- [ ] **Step 2: Verify pytest discovers fixtures**

Run: `uv run pytest tests/conftest.py --collect-only -q`

Expected: no test cases collected (only a conftest); no errors. (If `tests/` doesn't yet exist, `mkdir -p tests` first.)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
test: add shared pytest fixtures (tiny / default / over_constrained problems)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Write the registry tests (failing)

**Files:**
- Create: `tests/ai/optimizers/__init__.py` (empty package marker)
- Create: `tests/ai/__init__.py` (empty package marker)
- Create: `tests/ai/optimizers/test_registry.py`

- [ ] **Step 1: Create empty package markers**

Run:
```bash
mkdir -p tests/ai/optimizers
touch tests/ai/__init__.py tests/ai/optimizers/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/ai/optimizers/test_registry.py`:

```python
"""Tests for the Optimizer ABC and __init_subclass__ auto-registration."""

import pytest

from ai.optimizers.base import Optimizer


def test_init_subclass_registers(tiny_problem):
    """Concrete subclasses with a 'name' attribute are auto-registered."""
    available = Optimizer.list_available()
    assert "nsga2" in available
    assert "ccmo" in available


def test_create_returns_concrete_optimizer(tiny_problem):
    """Optimizer.create() returns an instance of the registered class."""
    from ai.optimizers.nsga2 import NSGAIIOptimizer

    optimizer = Optimizer.create("nsga2", tiny_problem)
    assert isinstance(optimizer, NSGAIIOptimizer)


def test_create_unknown_raises(tiny_problem):
    """Asking for an unknown algorithm raises ValueError listing valid choices."""
    with pytest.raises(ValueError) as exc:
        Optimizer.create("does-not-exist", tiny_problem)
    assert "Unknown optimizer" in str(exc.value)
    assert "nsga2" in str(exc.value)
    assert "ccmo" in str(exc.value)


def test_list_available_returns_sorted_names():
    """list_available() returns names in sorted order for stable display."""
    names = Optimizer.list_available()
    assert names == sorted(names)


def test_duplicate_name_raises():
    """Defining a second class with an existing name raises ValueError."""

    class DuplicateNSGA2(Optimizer):
        name = "nsga2"  # collides with the existing NSGAIIOptimizer

    # The class body itself triggers __init_subclass__; if it executed,
    # ValueError should have been raised. If we got here without the error,
    # registration is broken.
    pytest.fail("Expected ValueError on duplicate registration but none raised")


def test_abstract_intermediate_class_skips_registration():
    """A subclass with name='' (abstract intermediate) does NOT register."""
    pre_count = len(Optimizer.list_available())

    class AbstractMid(Optimizer):
        # No 'name' override; inherits the empty default. Should NOT register.
        def run(self, config=None, verbose=False):
            raise NotImplementedError

    post_count = len(Optimizer.list_available())
    assert post_count == pre_count
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/ai/optimizers/test_registry.py -v`

Expected: collection error or `ImportError: cannot import name 'Optimizer' from 'ai.optimizers.base'` — the module doesn't exist yet.

(No commit yet — implementation comes in the next task, then we commit both together.)

### Task 4: Implement `OptimizerConfig` / `OptimizerResult` base + step-status types

**Files:**
- Create: `src/ai/optimizers/result.py`

- [ ] **Step 1: Create the file**

```python
"""Algorithm-internal result schemas shared across optimizers.

`OptimizerConfig` and `OptimizerResult` are the lowest common denominator
that every optimizer must accept and return. Subclasses extend with
algorithm-specific fields.
"""

from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    """Shared evolutionary hyperparameters."""

    generations: int = 200
    pop_size: int = 100
    cxpb: float = 0.7
    mutpb: float = 0.2
    indpb: float = 0.05
    tournament_size: int = 4
    seed: int | None = None
    device: str = "cpu"


class NSGAIIConfig(OptimizerConfig):
    """NSGA-II hyperparameters. `pop_size` is the single-population size."""

    elitist: bool = True


class CCMOConfig(OptimizerConfig):
    """CCMO hyperparameters. `pop_size` is **per population** — CCMO maintains
    two coevolving populations of this size each, so total memory and
    per-generation evaluation count are 2 × pop_size.
    """

    pass


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


class OptimizerResult(BaseModel):
    """Lowest-common-denominator result every optimizer must return."""

    best_schedule: list[int]
    best_fitness: tuple[float, float, float]
    pareto_front: list[list[int]]
    pareto_fitnesses: list[tuple[float, float, float]]


class NSGAIIResult(OptimizerResult):
    """NSGA-II result. `pareto_front` is the rank-0 front."""

    step_history: list[GAStepStatus]


class CCMOResult(OptimizerResult):
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
```

- [ ] **Step 2: Verify Pydantic accepts the model definitions**

Run: `uv run python -c "from ai.optimizers.result import NSGAIIConfig, NSGAIIResult, CCMOConfig, CCMOResult; print('ok')"`

Expected: `ok`

- [ ] **Step 3: Commit (deferred — bundled with Task 5)**

### Task 5: Implement `Optimizer` ABC with `__init_subclass__` registry

**Files:**
- Create: `src/ai/optimizers/base.py`

- [ ] **Step 1: Create the file**

```python
"""Abstract base class for evolutionary optimizers with auto-registration.

Concrete subclasses are auto-registered when their class body executes,
via __init_subclass__. They must declare a 'name' class attribute.

Adding a new optimizer:
  1. Subclass Optimizer
  2. Set `name`, `config_class`, `result_class` class attributes
  3. Implement run()
  4. Import the module from src/ai/optimizers/__init__.py so the class is loaded
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from ai.domain.problem import SchedulingProblem
from ai.optimizers.result import OptimizerConfig, OptimizerResult


class Optimizer(ABC):
    """Abstract base for all evolutionary optimizers."""

    name: ClassVar[str] = ""
    config_class: ClassVar[type[OptimizerConfig]]
    result_class: ClassVar[type[OptimizerResult]]

    _registry: ClassVar[dict[str, type["Optimizer"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return  # abstract intermediate class, skip
        if cls.name in Optimizer._registry:
            raise ValueError(
                f"Optimizer '{cls.name}' is already registered by "
                f"{Optimizer._registry[cls.name].__name__}"
            )
        Optimizer._registry[cls.name] = cls

    def __init__(self, scheduling_problem: SchedulingProblem):
        self._sp = scheduling_problem

    @abstractmethod
    def run(
        self,
        config: OptimizerConfig | None = None,
        verbose: bool = False,
    ) -> OptimizerResult:
        ...

    @classmethod
    def create(cls, name: str, problem: SchedulingProblem) -> "Optimizer":
        if name not in cls._registry:
            raise ValueError(
                f"Unknown optimizer '{name}'. Available: {sorted(cls._registry)}"
            )
        return cls._registry[name](problem)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._registry)
```

- [ ] **Step 2: Verify the ABC imports**

Run: `uv run python -c "from ai.optimizers.base import Optimizer; print(Optimizer.list_available())"`

Expected: `[]` (empty — no concrete optimizers registered yet).

- [ ] **Step 3: Verify registry tests STILL fail (concrete classes don't exist yet)**

Run: `uv run pytest tests/ai/optimizers/test_registry.py -v`

Expected: still failing — `ImportError: cannot import name 'NSGAIIOptimizer'` and the assertions that "nsga2" and "ccmo" are registered will fail. The ABC itself works; concrete classes are missing.

- [ ] **Step 4: Commit (the ABC + result schemas as one logical unit)**

```bash
git add src/ai/optimizers/base.py src/ai/optimizers/result.py tests/ai/__init__.py tests/ai/optimizers/__init__.py tests/ai/optimizers/test_registry.py
git commit -m "$(cat <<'EOF'
feat: Optimizer ABC with __init_subclass__ auto-registration

- src/ai/optimizers/base.py: abstract base + factory + list-available
- src/ai/optimizers/result.py: OptimizerConfig/Result base + NSGAII/CCMO subclasses
- tests/ai/optimizers/test_registry.py: registry-invariant tests (concrete
  optimizer tests still fail until the algorithms are added in subsequent
  commits)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 3: RosteringProblem

### Task 6: Write failing parity tests for `RosteringProblem`

**Files:**
- Create: `tests/ai/optimizers/test_rostering_problem.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Numerical parity tests for RosteringProblem._evaluate_batch.

Verifies the new vectorized fitness produces identical values to the
existing GAOptimizer.batch_fitness on the same population.
"""

import torch

from ai.domain.problem import SchedulingProblem
from ai.optimizers.ga import GAOptimizer  # current DEAP-based, used as oracle


def test_evaluate_batch_shape(tiny_problem: SchedulingProblem):
    """SolutionBatch evals tensor has shape (popsize, 3)."""
    from ai.optimizers.rostering_problem import RosteringProblem

    problem = RosteringProblem(tiny_problem, device="cpu")
    pop = problem.generate_batch(8)
    problem.evaluate(pop)

    assert pop.evals.shape == (8, 3)
    assert not torch.isnan(pop.evals).any()


def test_imbalance_matches_jain(tiny_problem: SchedulingProblem):
    """Objective 0 (1 - Jain) matches GAOptimizer.batch_fitness output[0]."""
    from ai.optimizers.rostering_problem import RosteringProblem

    new_problem = RosteringProblem(tiny_problem, device="cpu")
    old_optimizer = GAOptimizer(tiny_problem)

    individuals = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],   # length 14 = 7 days × 2 shifts
    ]
    # GA's batch_fitness expects list[list[int]] of length num_shifts.
    old_fitnesses = old_optimizer.batch_fitness(individuals)
    old_imbalance = old_fitnesses[0][0]

    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(individuals[0], dtype=torch.int64)
    new_problem.evaluate(pop)
    new_imbalance = float(pop.evals[0, 0])

    assert abs(new_imbalance - old_imbalance) < 1e-5


def test_violations_count_correct(tiny_problem: SchedulingProblem):
    """Objective 1 (violations) = max-hours-overrun + 10 × unavailability hits."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # All shifts assigned to employee 0 — they go way over their 50h cap.
    schedule = [0] * tiny_problem.num_shifts
    new_problem = RosteringProblem(tiny_problem, device="cpu")
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    # Total assigned hours to employee 0 = num_shifts × shift_length.
    expected_overrun = tiny_problem.num_shifts * 8 - 50
    assert float(pop.evals[0, 1]) == pytest.approx(expected_overrun, rel=1e-5)


def test_b2b_count_correct(tiny_problem: SchedulingProblem):
    """Objective 2 (back-to-back) = count of consecutive same-employee shifts."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # All same employee → every adjacent pair is b2b → count = num_shifts - 1
    schedule = [0] * tiny_problem.num_shifts
    new_problem = RosteringProblem(tiny_problem, device="cpu")
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    assert int(pop.evals[0, 2]) == tiny_problem.num_shifts - 1


def test_unavailability_handling(tiny_problem: SchedulingProblem):
    """When an employee is unavailable but assigned, violations += 10 per hit."""
    from ai.optimizers.rostering_problem import RosteringProblem

    # Inject an unavailability: employee 0 unavailable on day 0.
    sp = tiny_problem.model_copy(update={"unavailability": frozenset({(0, 0)})})
    new_problem = RosteringProblem(sp, device="cpu")
    # Schedule with employee 0 on day 0 (both shifts) — 2 hits.
    schedule = [0, 0] + [1] * (sp.num_shifts - 2)
    pop = new_problem.generate_batch(1)
    pop._data[0] = torch.tensor(schedule, dtype=torch.int64)
    new_problem.evaluate(pop)

    # Violations include 10 × 2 = 20 from unavail, plus any max-hours overrun.
    # Employee 1 assigned 12 shifts × 8h = 96h > 50h cap → overrun 46h.
    # Total expected: 20 + 46 = 66.
    assert float(pop.evals[0, 1]) == pytest.approx(66.0, rel=1e-5)
```

(Add `import pytest` at the top.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/optimizers/test_rostering_problem.py -v`

Expected: `ImportError: cannot import name 'RosteringProblem'` — module doesn't exist yet.

### Task 7: Implement `RosteringProblem`

**Files:**
- Create: `src/ai/optimizers/rostering_problem.py`

- [ ] **Step 1: Create the file**

```python
"""EvoTorch Problem adapter for shift scheduling.

Decision space: integer vector of length num_shifts, each gene in
[0, num_employees-1] = which employee is assigned to that shift.

Objectives (all minimized):
  0: imbalance       = 1 - Jain's fairness index
  1: violations      = max-hours overrun + 10 * unavailability hits
  2: back_to_back    = count of consecutive same-employee shifts
"""

import torch
from evotorch import Problem, SolutionBatch

from ai.domain.problem import SchedulingProblem


class RosteringProblem(Problem):
    def __init__(
        self,
        scheduling_problem: SchedulingProblem,
        device: torch.device | str = "cpu",
    ):
        self._sp = scheduling_problem
        super().__init__(
            objective_sense=["min", "min", "min"],
            solution_length=scheduling_problem.num_shifts,
            initial_bounds=(0, scheduling_problem.num_employees - 1),
            bounds=(0, scheduling_problem.num_employees - 1),
            dtype=torch.int64,
            device=device,
        )
        self._precompute_tensors()

    def _precompute_tensors(self) -> None:
        sp = self._sp
        d = self.device
        self._shift_lens = torch.tensor(
            sp.shift_lengths, dtype=torch.float64, device=d
        )
        self._shift_types = torch.arange(sp.num_shifts, device=d) % sp.shifts_per_day
        self._shift_hour_values = self._shift_lens[self._shift_types]
        self._max_hours_t = torch.tensor(sp.max_hours, dtype=torch.float64, device=d)

        unavail_mask = torch.zeros(
            sp.days, sp.num_employees, dtype=torch.bool, device=d
        )
        for day, emp in sp.unavailability:
            unavail_mask[day, emp] = True
        self._unavail_mask = unavail_mask

    def _evaluate_batch(self, solutions: SolutionBatch) -> None:
        pop = solutions.values.to(torch.long)
        n = pop.shape[0]
        sp = self._sp

        lens = self._shift_hour_values.unsqueeze(0).expand(n, -1)
        hours = torch.zeros(
            n, sp.num_employees, dtype=torch.float64, device=pop.device
        )
        hours.scatter_add_(1, pop, lens)

        sum_h = hours.sum(dim=1)
        sum_sq = hours.pow(2).sum(dim=1)
        jain = torch.where(
            sum_sq > 0,
            sum_h.pow(2) / (sp.num_employees * sum_sq),
            torch.ones(n, dtype=torch.float64, device=pop.device),
        )
        imbalance = 1.0 - jain

        exceed = (hours - self._max_hours_t).clamp(min=0).sum(dim=1)
        day_per_shift = (
            torch.arange(sp.num_shifts, device=pop.device) // sp.shifts_per_day
        )
        violations_per_cell = self._unavail_mask[
            day_per_shift.unsqueeze(0).expand(n, -1), pop
        ]
        unavail_count = violations_per_cell.sum(dim=1).to(torch.float64)
        violations = exceed + unavail_count * 10.0

        b2b = (pop[:, :-1] == pop[:, 1:]).sum(dim=1).to(torch.float64)

        fitnesses = torch.stack([imbalance, violations, b2b], dim=1)
        solutions.set_evals(fitnesses.to(torch.float32))
```

- [ ] **Step 2: Run parity tests to verify they pass**

Run: `uv run pytest tests/ai/optimizers/test_rostering_problem.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/ai/optimizers/rostering_problem.py tests/ai/optimizers/test_rostering_problem.py
git commit -m "$(cat <<'EOF'
feat: RosteringProblem — EvoTorch Problem subclass with vectorized fitness

Vectorizes the unavailability handling that was a Python-level for-loop in
optimizers/ga.py:108 into a single gather over a precomputed (days,
num_employees) bool mask. Parity tests against GAOptimizer.batch_fitness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 4: Operators

### Task 8: Write failing operator tests

**Files:**
- Create: `tests/ai/optimizers/test_operators.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the three CopyingOperator subclasses."""

import pytest
import torch

from ai.domain.problem import SchedulingProblem
from ai.optimizers.rostering_problem import RosteringProblem


@pytest.fixture
def problem(tiny_problem: SchedulingProblem) -> RosteringProblem:
    torch.manual_seed(42)
    return RosteringProblem(tiny_problem, device="cpu")


def test_crossover_preserves_dimensions(problem: RosteringProblem):
    from ai.optimizers.operators import DayAlignedCrossOver

    op = DayAlignedCrossOver(problem, tournament_size=2, cross_over_rate=1.0)
    parents = problem.generate_batch(8)
    problem.evaluate(parents)
    children = op._do(parents)

    assert children.values.shape == parents.values.shape
    assert children.values.dtype == torch.int64
    # Values stay within the problem bounds.
    assert children.values.min() >= 0
    assert children.values.max() <= problem._sp.num_employees - 1


def test_crossover_only_at_day_boundaries(problem: RosteringProblem):
    """When cross_over_rate=1, the cut point should be a multiple of shifts_per_day."""
    from ai.optimizers.operators import DayAlignedCrossOver

    op = DayAlignedCrossOver(problem, tournament_size=2, cross_over_rate=1.0)
    sp = problem._sp

    # Build two parents with distinguishable patterns.
    parents = problem.generate_batch(2)
    parents._data[0] = torch.zeros(sp.num_shifts, dtype=torch.int64)
    parents._data[1] = torch.ones(sp.num_shifts, dtype=torch.int64) + 1  # all 2s
    problem.evaluate(parents)

    torch.manual_seed(0)
    children = op._do(parents)
    # Find positions where child[0] differs from parent[0] (=0).
    diff_positions = torch.where(children.values[0] != 0)[0]
    if diff_positions.numel() > 0:
        first_diff = int(diff_positions.min())
        # The cut must be a day boundary.
        assert first_diff % sp.shifts_per_day == 0


def test_mutation_respects_indpb_distribution(problem: RosteringProblem):
    """With mut_rate=1 and indpb=0.5 over a large batch, ~50% of genes change."""
    from ai.optimizers.operators import UniformIntMutation

    torch.manual_seed(123)
    op = UniformIntMutation(problem, indpb=0.5, mut_rate=1.0)

    # All-zero starting batch.
    n = 200
    batch = problem.generate_batch(n)
    batch._data[:] = 0
    problem.evaluate(batch)

    children = op._do(batch)
    # Fraction of non-zero entries in children should be ≈ 0.5 × (1 - 1/num_employees).
    # Because uniform replacement might pick 0 again with probability 1/num_employees.
    sp = problem._sp
    expected_nonzero_fraction = 0.5 * (1 - 1 / sp.num_employees)
    actual_nonzero_fraction = float((children.values != 0).float().mean())
    assert abs(actual_nonzero_fraction - expected_nonzero_fraction) < 0.05


def test_repair_eliminates_unavailability_violations(default_problem: SchedulingProblem):
    """After repair, no cell has an unavailable employee on the corresponding day."""
    from ai.optimizers.operators import RepairOperator

    # Inject some unavailability on the default problem.
    sp = default_problem.model_copy(
        update={"unavailability": frozenset({(0, 0), (5, 1), (10, 2)})}
    )
    rp = RosteringProblem(sp, device="cpu")
    op = RepairOperator(rp)

    torch.manual_seed(7)
    batch = rp.generate_batch(8)
    rp.evaluate(batch)
    repaired = op._do(batch)

    # No cell should have an unavailable assignment.
    n, T = repaired.values.shape
    day_per_shift = torch.arange(T) // sp.shifts_per_day
    unavail_hits = rp._unavail_mask[
        day_per_shift.unsqueeze(0).expand(n, -1), repaired.values
    ]
    assert unavail_hits.sum() == 0


def test_repair_idempotent(default_problem: SchedulingProblem):
    """Calling repair twice produces no further changes."""
    from ai.optimizers.operators import RepairOperator

    sp = default_problem.model_copy(update={"unavailability": frozenset({(0, 0), (5, 1)})})
    rp = RosteringProblem(sp, device="cpu")
    op = RepairOperator(rp)

    torch.manual_seed(0)
    batch = rp.generate_batch(4)
    rp.evaluate(batch)
    repaired_once = op._do(batch)
    repaired_twice = op._do(repaired_once)

    assert torch.equal(repaired_once.values, repaired_twice.values)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/optimizers/test_operators.py -v`

Expected: `ImportError: cannot import name 'DayAlignedCrossOver'` (or similar) for each test.

### Task 9: Implement the three operators

**Files:**
- Create: `src/ai/optimizers/operators.py`

- [ ] **Step 1: Verify EvoTorch's `CrossOver._do_tournament` API exists**

Run: `uv run python -c "from evotorch.operators.base import CrossOver; print([m for m in dir(CrossOver) if 'tournament' in m.lower()])"`

Expected: a list containing something like `['_do_tournament']` or `['do_tournament']`.

If the helper has a different name, adjust the `DayAlignedCrossOver._do` body in Step 2 to call the correct method. If it's missing entirely, replace the call with the inline tournament select shown in Step 2's fallback comment.

- [ ] **Step 2: Create the operators file**

```python
"""Vectorized CopyingOperator subclasses for the rostering problem.

All three operators transform a SolutionBatch on the GPU/CPU device of the
underlying Problem. None contain Python-level per-individual loops.
"""

import torch
from evotorch import SolutionBatch
from evotorch.operators import CopyingOperator, CrossOver

from ai.optimizers.rostering_problem import RosteringProblem


class DayAlignedCrossOver(CrossOver):
    """Two-parent recombination cut at a day boundary.

    With probability `cross_over_rate`, the right half (from a randomly
    chosen day boundary) is swapped between the two parents.
    """

    def __init__(
        self,
        problem: RosteringProblem,
        *,
        tournament_size: int = 4,
        cross_over_rate: float = 0.7,
    ):
        super().__init__(problem, tournament_size=tournament_size, cross_over_rate=cross_over_rate)
        self._shifts_per_day = problem._sp.shifts_per_day
        self._num_shifts = problem._sp.num_shifts

    def _do_cross_over(self, parents1: torch.Tensor, parents2: torch.Tensor) -> SolutionBatch:
        # parents1, parents2: (n_pairs, num_shifts) int64
        n_pairs = parents1.shape[0]
        d = parents1.device
        num_days = self._num_shifts // self._shifts_per_day

        # Sample a day boundary in [1, num_days) per pair.
        day_cuts = torch.randint(1, max(num_days, 2), (n_pairs,), device=d)
        shift_cuts = day_cuts * self._shifts_per_day  # (n_pairs,)

        children1 = parents1.clone()
        children2 = parents2.clone()

        # For each pair, swap the right half (>= shift_cut).
        idx = torch.arange(self._num_shifts, device=d).unsqueeze(0)  # (1, T)
        right_mask = idx >= shift_cuts.unsqueeze(1)  # (n_pairs, T)
        children1 = torch.where(right_mask, parents2, parents1)
        children2 = torch.where(right_mask, parents1, parents2)

        # Build a SolutionBatch from the offspring (n_pairs * 2 rows).
        out = self._make_children_batch(torch.cat([children1, children2], dim=0))
        return out


class UniformIntMutation(CopyingOperator):
    """Per-gene uniform integer mutation with both per-individual and per-gene gates."""

    def __init__(
        self,
        problem: RosteringProblem,
        *,
        indpb: float = 0.05,
        mut_rate: float = 0.2,
    ):
        super().__init__(problem)
        self._indpb = indpb
        self._mut_rate = mut_rate
        self._num_employees = problem._sp.num_employees

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = batch.clone()
        values = result.values
        n = values.shape[0]
        ind_mask = (
            torch.rand(n, device=values.device) < self._mut_rate
        ).unsqueeze(1)
        gene_mask = torch.rand_like(values, dtype=torch.float32) < self._indpb
        new_values = torch.randint(
            0,
            self._num_employees,
            values.shape,
            device=values.device,
            dtype=values.dtype,
        )
        flip = ind_mask & gene_mask
        values[flip] = new_values[flip]
        return result


class RepairOperator(CopyingOperator):
    """Replace any cell where the assigned employee is unavailable with a random valid one."""

    def __init__(self, problem: RosteringProblem):
        super().__init__(problem)
        sp = problem._sp
        self._shifts_per_day = sp.shifts_per_day
        self._num_employees = sp.num_employees
        self._unavail_mask = problem._unavail_mask

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = batch.clone()
        values = result.values
        n, T = values.shape
        d = values.device

        day_per_shift = torch.arange(T, device=d) // self._shifts_per_day
        is_unavail = self._unavail_mask[
            day_per_shift.unsqueeze(0).expand(n, -1), values
        ]
        if not is_unavail.any():
            return result

        # availability[t, e] = True if employee e is available on day(t).
        availability = ~self._unavail_mask[day_per_shift]  # (T, num_employees)
        rand = torch.rand(n, T, self._num_employees, device=d)
        # Mask infeasible employees per shift.
        rand = rand * availability.unsqueeze(0).float() + (
            -1.0
        ) * (~availability).unsqueeze(0).float()
        replacement = rand.argmax(dim=-1)  # (N, T)

        values[is_unavail] = replacement[is_unavail]
        return result
```

(Note: `DayAlignedCrossOver` extends `CrossOver` rather than `CopyingOperator` so EvoTorch's tournament-selection bookkeeping in the parent class is reused. The actual override is `_do_cross_over`, which receives parent pairs already sampled by tournament.)

- [ ] **Step 2 — fallback if `_do_cross_over` isn't the right hook:**

If EvoTorch's `CrossOver` API uses a different override (verify with `uv run python -c "from evotorch.operators.base import CrossOver; help(CrossOver._do)"`), replace `DayAlignedCrossOver` with this `CopyingOperator` implementation instead:

```python
class DayAlignedCrossOver(CopyingOperator):
    def __init__(self, problem, *, tournament_size=4, cross_over_rate=0.7):
        super().__init__(problem)
        self._tournament_size = tournament_size
        self._cross_over_rate = cross_over_rate
        self._shifts_per_day = problem._sp.shifts_per_day
        self._num_shifts = problem._sp.num_shifts

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        n = len(batch)
        d = batch.values.device
        # Tournament select: 2*n_pairs parents.
        n_pairs = n
        idx_pool = torch.randint(0, n, (n_pairs * 2, self._tournament_size), device=d)
        # Lower rank+crowding wins; for simplicity here, lower obj-sum wins.
        evals_sums = batch.evals.sum(dim=1)
        winners_per_tourn = idx_pool.gather(
            1, evals_sums[idx_pool].argmin(dim=1, keepdim=True)
        ).squeeze(1)  # (n_pairs * 2,)
        parents = batch.values[winners_per_tourn].view(n_pairs, 2, self._num_shifts)
        p1, p2 = parents[:, 0], parents[:, 1]

        num_days = self._num_shifts // self._shifts_per_day
        day_cuts = torch.randint(1, max(num_days, 2), (n_pairs,), device=d)
        shift_cuts = day_cuts * self._shifts_per_day
        idx = torch.arange(self._num_shifts, device=d).unsqueeze(0)
        right_mask = idx >= shift_cuts.unsqueeze(1)
        c1 = torch.where(right_mask, p2, p1)
        c2 = torch.where(right_mask, p1, p2)

        # Apply with cross_over_rate; otherwise pass parents through.
        apply_mask = torch.rand(n_pairs, device=d) < self._cross_over_rate
        c1 = torch.where(apply_mask.unsqueeze(1), c1, p1)
        c2 = torch.where(apply_mask.unsqueeze(1), c2, p2)

        children_values = torch.cat([c1, c2], dim=0)
        new_batch = SolutionBatch(self.problem, popsize=children_values.shape[0])
        new_batch._data = children_values
        return new_batch
```

- [ ] **Step 3: Run operator tests to verify they pass**

Run: `uv run pytest tests/ai/optimizers/test_operators.py -v`

Expected: all 5 tests PASS.

If `test_crossover_only_at_day_boundaries` fails because the fallback impl doesn't preserve the day-boundary invariant exactly, audit the cut-point construction (`shift_cuts = day_cuts * self._shifts_per_day`) — that line must produce shift indices that are multiples of `shifts_per_day`.

- [ ] **Step 4: Commit**

```bash
git add src/ai/optimizers/operators.py tests/ai/optimizers/test_operators.py
git commit -m "$(cat <<'EOF'
feat: vectorized day-aligned crossover, uniform-int mutation, repair operators

Three CopyingOperator subclasses operating on full SolutionBatch tensors;
no Python-level per-individual loops. Crossover only cuts at day
boundaries; repair replaces unavailable assignments with random valid
employees via a (T, num_employees) availability mask.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 2 — Replace DEAP NSGA-II with EvoTorch NSGA-II

## Commit 5: NSGA-II + service/training/api/frontend rewrite + drop DEAP

This is the biggest commit in the plan. It contains all the renames and the deletion of the old DEAP code, in a single atomic step so that every reverse caller is fixed before DEAP is removed.

### Task 10: Write failing tests for `NSGAIIOptimizer`

**Files:**
- Create: `tests/ai/optimizers/test_nsga2.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Convergence + structural tests for NSGAIIOptimizer."""

import math

import pytest
import torch

from ai.domain.problem import SchedulingProblem


def test_result_shape_correct(tiny_problem: SchedulingProblem):
    """run() returns an NSGAIIResult with the expected fields."""
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(tiny_problem)
    config = NSGAIIConfig(generations=5, pop_size=20, seed=42)
    result = optimizer.run(config)

    assert len(result.best_schedule) == tiny_problem.num_shifts
    assert len(result.best_fitness) == 3
    assert all(0 <= s < tiny_problem.num_employees for s in result.best_schedule)
    assert len(result.pareto_front) >= 1
    assert len(result.step_history) == 5


def test_no_nan_fitnesses(tiny_problem: SchedulingProblem):
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(tiny_problem)
    result = optimizer.run(NSGAIIConfig(generations=5, pop_size=20, seed=42))

    assert all(not math.isnan(f) for f in result.best_fitness)
    for fits in result.pareto_fitnesses:
        assert all(not math.isnan(f) for f in fits)


def test_fitness_improves_over_generations(default_problem: SchedulingProblem):
    """gen-0 mean objectives should be worse than gen-N mean objectives."""
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(default_problem)
    result = optimizer.run(NSGAIIConfig(generations=20, pop_size=50, seed=42))

    gen0 = result.step_history[0]
    gen_last = result.step_history[-1]
    assert gen_last.mean_obj1_violations <= gen0.mean_obj1_violations


@pytest.mark.slow
def test_default_instance_converges(default_problem: SchedulingProblem):
    """Issue #15 AC: with seed=42, 50 generations, 100 popsize, the best
    solution has imbalance < 0.05, violations == 0, b2b < 30.
    """
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(default_problem)
    result = optimizer.run(NSGAIIConfig(generations=50, pop_size=100, seed=42))

    assert result.best_fitness[0] < 0.05, f"imbalance too high: {result.best_fitness[0]}"
    assert result.best_fitness[1] == 0, f"violations: {result.best_fitness[1]}"
    assert result.best_fitness[2] < 30, f"b2b too high: {result.best_fitness[2]}"


def test_pareto_front_non_empty(default_problem: SchedulingProblem):
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(default_problem)
    result = optimizer.run(NSGAIIConfig(generations=10, pop_size=50, seed=42))

    assert len(result.pareto_front) >= 1
    assert all(len(s) == default_problem.num_shifts for s in result.pareto_front)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/optimizers/test_nsga2.py -v -m "not slow"`

Expected: `ImportError: cannot import name 'NSGAIIOptimizer'`.

### Task 11: Implement `NSGAIIOptimizer`

**Files:**
- Create: `src/ai/optimizers/nsga2.py`

- [ ] **Step 1: Create the file**

```python
"""NSGA-II for multi-objective shift scheduling on EvoTorch.

Selection (Pareto-rank + crowding distance) is automatic because the
underlying RosteringProblem declares 3 minimized objectives.
"""

from typing import ClassVar

import torch
from evotorch.algorithms import GeneticAlgorithm

from ai.optimizers.base import Optimizer
from ai.optimizers.operators import (
    DayAlignedCrossOver,
    RepairOperator,
    UniformIntMutation,
)
from ai.optimizers.result import (
    GAStepStatus,
    NSGAIIConfig,
    NSGAIIResult,
    OptimizerConfig,
    OptimizerResult,
)
from ai.optimizers.rostering_problem import RosteringProblem


class NSGAIIOptimizer(Optimizer):
    name: ClassVar[str] = "nsga2"
    config_class: ClassVar[type[OptimizerConfig]] = NSGAIIConfig
    result_class: ClassVar[type[OptimizerResult]] = NSGAIIResult

    def run(
        self,
        config: NSGAIIConfig | None = None,
        verbose: bool = False,
    ) -> NSGAIIResult:
        config = config or NSGAIIConfig()
        if config.seed is not None:
            torch.manual_seed(config.seed)

        problem = RosteringProblem(self._sp, device=config.device)
        operators = [
            DayAlignedCrossOver(
                problem,
                tournament_size=config.tournament_size,
                cross_over_rate=config.cxpb,
            ),
            UniformIntMutation(problem, indpb=config.indpb, mut_rate=config.mutpb),
            RepairOperator(problem),
        ]
        searcher = GeneticAlgorithm(
            problem,
            operators=operators,
            popsize=config.pop_size,
            elitist=config.elitist,
        )

        history: list[GAStepStatus] = []
        for gen in range(config.generations):
            searcher.step()
            history.append(self._snapshot(gen, searcher))
            if verbose and gen % 10 == 0:
                self._print_snapshot(history[-1])

        return self._collect_result(searcher, history)

    @staticmethod
    def _snapshot(gen: int, searcher) -> GAStepStatus:
        evals = searcher.population.evals  # (popsize, 3)
        means = evals.mean(dim=0).tolist()
        return GAStepStatus(
            generation=gen,
            mean_obj0_imbalance=float(means[0]),
            mean_obj1_violations=float(means[1]),
            mean_obj2_b2b=float(means[2]),
            pareto_front_size=int((searcher.population.compute_pareto_ranks() == 0).sum()),
        )

    @staticmethod
    def _print_snapshot(s: GAStepStatus) -> None:
        print(
            f"gen={s.generation} imbalance={s.mean_obj0_imbalance:.4f} "
            f"violations={s.mean_obj1_violations:.1f} b2b={s.mean_obj2_b2b:.1f} "
            f"pareto={s.pareto_front_size}"
        )

    def _collect_result(self, searcher, history: list[GAStepStatus]) -> NSGAIIResult:
        population = searcher.population
        ranks = population.compute_pareto_ranks()
        front_idx = torch.where(ranks == 0)[0].tolist()
        front_values = [list(map(int, population.values[i].tolist())) for i in front_idx]
        front_evals = [tuple(map(float, population.evals[i].tolist())) for i in front_idx]

        # Best = lowest objective sum.
        best_local = min(range(len(front_evals)), key=lambda i: sum(front_evals[i]))

        return NSGAIIResult(
            best_schedule=front_values[best_local],
            best_fitness=front_evals[best_local],
            pareto_front=front_values,
            pareto_fitnesses=front_evals,
            step_history=history,
        )
```

### Task 12: Wire `__init__.py` so registration triggers on import

**Files:**
- Create: `src/ai/optimizers/__init__.py` (overwriting any current one)

- [ ] **Step 1: Read the current `optimizers/__init__.py`**

Run: `cat src/ai/optimizers/__init__.py`

(If empty, that's fine; if it has content, note it — we'll preserve nothing because the file contained only an empty docstring.)

- [ ] **Step 2: Replace with eager-imports**

```python
"""Optimizer package.

Concrete optimizer classes are eagerly imported here so the Optimizer ABC's
__init_subclass__ hook fires and registers them, regardless of whether
the caller imports them directly.
"""

from ai.optimizers.base import Optimizer
from ai.optimizers.nsga2 import NSGAIIOptimizer  # noqa: F401 — import for registration
# ai.optimizers.ccmo is imported in a later commit; until then, only nsga2 registers.

__all__ = ["Optimizer", "NSGAIIOptimizer"]
```

(The `ccmo` import line is added in Task 18 after `CCMOOptimizer` exists. This commit reflects the intermediate state where only NSGA-II is registered. This intentional sequencing keeps each commit's tree functional.)

- [ ] **Step 3: Verify NSGA-II tests pass**

Run: `uv run pytest tests/ai/optimizers/test_nsga2.py -v -m "not slow"`

Expected: 4 tests PASS (all except `test_default_instance_converges` which is `@pytest.mark.slow`).

- [ ] **Step 4: Verify the registry test now sees nsga2 (but ccmo is still expected to be missing for now)**

The current `test_registry.py::test_init_subclass_registers` asserts both `nsga2` AND `ccmo` are registered. Until Task 18, `ccmo` is not yet implemented. **Skip the ccmo assertion temporarily** by adjusting the test:

Edit `tests/ai/optimizers/test_registry.py:test_init_subclass_registers`:

```python
def test_init_subclass_registers(tiny_problem):
    """Concrete subclasses with a 'name' attribute are auto-registered."""
    available = Optimizer.list_available()
    assert "nsga2" in available
    # "ccmo" gets added in a later commit
```

(In Task 18 we'll add the `assert "ccmo" in available` back.)

- [ ] **Step 5: Run registry tests**

Run: `uv run pytest tests/ai/optimizers/test_registry.py -v`

Expected: 5 of 6 tests PASS. The `test_create_unknown_raises` test asserts both `nsga2` and `ccmo` are mentioned in the error message; until ccmo is registered, only `nsga2` will appear. Adjust this test similarly:

```python
def test_create_unknown_raises(tiny_problem):
    with pytest.raises(ValueError) as exc:
        Optimizer.create("does-not-exist", tiny_problem)
    assert "Unknown optimizer" in str(exc.value)
    assert "nsga2" in str(exc.value)
    # "ccmo" gets added in a later commit
```

Run again: `uv run pytest tests/ai/optimizers/test_registry.py -v`

Expected: all 6 tests PASS.

### Task 13: Replace `services/ga_inference.py` with `services/optimizer_inference.py`

**Files:**
- Delete: `src/ai/services/ga_inference.py`
- Create: `src/ai/services/optimizer_inference.py`

- [ ] **Step 1: Create the new service**

```python
"""Single inference service that dispatches to any registered optimizer."""

from fastapi import HTTPException

from ai.domain.problem import ScheduleConverter, SchedulingProblem
from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.optimizers.result import CCMOResult
from ai.services.metrics import compute_metrics


def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
    device: str = "cpu",
) -> SchedulingResponse:
    """Dispatch to the named optimizer; convert its best schedule to the API response."""
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)

    config = optimizer.config_class(
        generations=generations,
        pop_size=pop_size,
        device=device,
    )
    result = optimizer.run(config)

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

- [ ] **Step 2: Delete the old service**

Run: `git rm src/ai/services/ga_inference.py`

- [ ] **Step 3: Verify the new service imports**

Run: `uv run python -c "from ai.services.optimizer_inference import run_optimizer_inference; print('ok')"`

Expected: `ok`.

### Task 14: Replace `training/ga.py` with `training/evolutionary.py`

**Files:**
- Delete: `src/ai/training/ga.py`
- Create: `src/ai/training/evolutionary.py`

- [ ] **Step 1: Create the new training CLI**

```python
"""Train an evolutionary optimizer for shift scheduling.

Usage:
    python -m ai.training.evolutionary --algorithm nsga2 --generations 200 --pop-size 100
    python -m ai.training.evolutionary --algorithm ccmo  --generations 200 --pop-size 100 --device cuda
"""

import argparse
import json
from pathlib import Path

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem, jain_fairness_index
from ai.optimizers.base import Optimizer


def train_evolutionary(
    algorithm: str,
    generations: int,
    pop_size: int,
    cxpb: float,
    mutpb: float,
    indpb: float,
    seed: int | None,
    device: str,
    output_dir: str,
) -> None:
    config_environment = EnvironmentConfig()
    problem = SchedulingProblem.from_config(config_environment)

    optimizer = Optimizer.create(algorithm, problem)
    config = optimizer.config_class(
        generations=generations,
        pop_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        indpb=indpb,
        seed=seed,
        device=device,
    )

    print(f"Running {algorithm}: {generations} generations, pop_size={pop_size}")
    print(f"  cxpb={cxpb}, mutpb={mutpb}, indpb={indpb}, device={device}")
    print(f"  Problem: {problem.num_employees} employees, {problem.num_shifts} shifts")
    print()

    result = optimizer.run(config, verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schedule_path = output_path / f"{algorithm}_best_schedule.json"
    history_path = output_path / f"{algorithm}_step_history.json"

    schedule_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "schedule": result.best_schedule,
                "best_fitness": list(result.best_fitness),
                "pareto_front_size": len(result.pareto_front),
                "config": config.model_dump(),
            },
            indent=2,
        )
    )
    history_path.write_text(
        json.dumps([s.model_dump() for s in result.step_history], indent=2)
    )

    hours = problem.compute_hours(result.best_schedule)
    jain = jain_fairness_index(hours)
    print(
        f"\nBest: imbalance={result.best_fitness[0]:.4f}, "
        f"violations={result.best_fitness[1]:.1f}, b2b={result.best_fitness[2]:.0f}"
    )
    print(f"Jain's fairness index: {jain:.4f}")
    print(f"Pareto front size: {len(result.pareto_front)} solutions")

    print("\nHours distribution:")
    for i, h in enumerate(hours):
        emp_type = problem.employee_types[i]
        max_h = problem.max_hours[i]
        marker = " OVER" if h > max_h else ""
        print(f"  Employee {i} ({emp_type}): {h:.0f}h / {max_h}h{marker}")

    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an evolutionary optimizer for shift scheduling"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=Optimizer.list_available(),
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument("--indpb", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train_evolutionary(**vars(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Delete the old training script**

Run: `git rm src/ai/training/ga.py`

- [ ] **Step 3: Verify the CLI starts without errors**

Run: `uv run python -m ai.training.evolutionary --algorithm nsga2 --generations 2 --pop-size 10 --seed 42`

Expected: prints generation progress, finishes with a hours-distribution table, writes `checkpoints/nsga2_best_schedule.json` and `checkpoints/nsga2_step_history.json`. No exceptions.

### Task 15: Update `api/inference.py` — add `/predict/evolutionary/{algorithm}` route + deprecation shim

**Files:**
- Modify: `src/ai/api/inference.py`

- [ ] **Step 1: Replace the file content**

```python
"""FastAPI inference routes."""

from enum import Enum

from fastapi import APIRouter, Query

from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.services.optimizer_inference import run_optimizer_inference
from ai.services.rl_inference import run_rl_inference

router = APIRouter(prefix="/predict", tags=["inference"])

# Built dynamically so adding to the registry auto-extends the API enum.
EvolutionaryAlgorithm = Enum(
    "EvolutionaryAlgorithm",
    {n.upper(): n for n in Optimizer.list_available()},
)


@router.post("/rl", response_model=SchedulingResponse)
async def predict_rl(
    request: SchedulingRequest,
    checkpoint: str = Query("best_model.zip"),
) -> SchedulingResponse:
    """Run SB3 model inference for schedule optimization."""
    return run_rl_inference(request, checkpoint=checkpoint)


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
        generations=generations,
        pop_size=pop_size,
        device=device,
    )


@router.post("/ga", response_model=SchedulingResponse, deprecated=True)
async def predict_ga(
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
) -> SchedulingResponse:
    """DEPRECATED: use /predict/evolutionary/nsga2."""
    return run_optimizer_inference(
        "nsga2", request, generations=generations, pop_size=pop_size
    )
```

- [ ] **Step 2: Verify FastAPI app starts**

Run: `uv run python -c "from ai.main import app; print([r.path for r in app.routes])"`

Expected: list of routes including `/predict/rl`, `/predict/evolutionary/{algorithm}`, `/predict/ga`.

### Task 16: Update Pydantic schemas — rename `GA*` to `NSGAII*`, add `CCMO*`, deprecation aliases

**Files:**
- Modify: `src/ai/domain/schemas.py`

- [ ] **Step 1: Append new schemas + deprecation aliases**

Replace the existing GA-specific schemas at the end of `src/ai/domain/schemas.py` (currently `GAFitnessResult`, `GAConfigSnapshot`, `GATrainResult`) with:

```python
# === NSGA-II (renamed from GA*) ===

class NSGAIIFitnessResult(BaseModel):
    imbalance: float
    constraint_violations: float
    back_to_back: float


class NSGAIIConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    generations: int
    pop_size: int
    cxpb: float
    mutpb: float
    indpb: float
    tournament_size: int
    elitist: bool
    seed: int | None
    device: str


class NSGAIITrainResult(BaseModel):
    schedule: list[int]
    fitness: NSGAIIFitnessResult
    pareto_front_size: int
    config: NSGAIIConfigSnapshot


# === CCMO ===

class CCMOFitnessResult(BaseModel):
    imbalance: float
    constraint_violations: float
    back_to_back: float


class CCMOConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    generations: int
    pop_size: int
    cxpb: float
    mutpb: float
    indpb: float
    tournament_size: int
    seed: int | None
    device: str


class CCMOTrainResult(BaseModel):
    schedule: list[int]
    fitness: CCMOFitnessResult
    feasible_front_size: int
    auxiliary_front_size: int
    fell_back_to_auxiliary: bool
    config: CCMOConfigSnapshot


# === Benchmark ===

class BenchmarkRunRecord(BaseModel):
    instance: str
    algorithm: str
    seed: int
    hypervolume: float
    feasible_front_size: int
    best_imbalance: float
    best_violations: float
    best_b2b: int
    wall_clock_s: float


class BenchmarkAggregate(BaseModel):
    instance: str
    nsga2_hv_mean: float | None = None
    nsga2_hv_std: float | None = None
    nsga2_n_seeds: int = 0
    ccmo_hv_mean: float | None = None
    ccmo_hv_std: float | None = None
    ccmo_n_seeds: int = 0
    wilcoxon_p: float | None = None


class BenchmarkReport(BaseModel):
    config_summary: dict
    per_run: list[BenchmarkRunRecord]
    aggregate: list[BenchmarkAggregate]


# === One-release deprecation aliases ===

import warnings


class GAFitnessResult(NSGAIIFitnessResult):
    """Deprecated alias for NSGAIIFitnessResult."""

    def __init__(self, **data):
        warnings.warn(
            "GAFitnessResult is deprecated; use NSGAIIFitnessResult",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class GAConfigSnapshot(NSGAIIConfigSnapshot):
    """Deprecated alias for NSGAIIConfigSnapshot."""

    def __init__(self, **data):
        warnings.warn(
            "GAConfigSnapshot is deprecated; use NSGAIIConfigSnapshot",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class GATrainResult(NSGAIITrainResult):
    """Deprecated alias for NSGAIITrainResult."""

    def __init__(self, **data):
        warnings.warn(
            "GATrainResult is deprecated; use NSGAIITrainResult",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
```

- [ ] **Step 2: Verify schemas import**

Run: `uv run python -c "from ai.domain.schemas import NSGAIIFitnessResult, CCMOTrainResult, BenchmarkReport, GATrainResult; print('ok')"`

Expected: `ok` (with no errors; deprecation warnings only fire when the alias classes are instantiated).

### Task 17: Update frontend api_client + delete old `ga.py`; drop DEAP from pyproject.toml

**Files:**
- Modify: `src/frontend/api_client/client.py`
- Delete: `src/ai/optimizers/ga.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current frontend client**

Run: `grep -n "predict/ga\|/ga" src/frontend/api_client/client.py`

Note the exact line numbers and quoted strings that need to be updated.

- [ ] **Step 2: Update the route call**

Edit `src/frontend/api_client/client.py`: replace any reference to `/predict/ga` with `/predict/evolutionary/nsga2` (do not touch `/predict/rl` or unrelated paths). For example, if the line reads:

```python
response = client.post(f"{BACKEND_URL}/predict/ga", ...)
```

Change it to:

```python
response = client.post(f"{BACKEND_URL}/predict/evolutionary/nsga2", ...)
```

- [ ] **Step 3: Delete the old GAOptimizer**

Run: `git rm src/ai/optimizers/ga.py`

- [ ] **Step 3a: Clean up the parity test that imported GAOptimizer**

The parity test added in Task 6 (`test_imbalance_matches_jain` in `tests/ai/optimizers/test_rostering_problem.py`) imports `from ai.optimizers.ga import GAOptimizer` for cross-validation. With GAOptimizer gone, that import breaks the whole test module.

Edit `tests/ai/optimizers/test_rostering_problem.py`:

- Delete the line `from ai.optimizers.ga import GAOptimizer` at the top.
- Delete the entire `test_imbalance_matches_jain` test function (its purpose was a one-shot oracle check; the four other tests in the file continue to enforce shape, violations math, b2b math, and unavailability handling using hardcoded expected values).

After this edit, `test_rostering_problem.py` has 4 tests, all standalone.

- [ ] **Step 4: Drop DEAP from pyproject.toml**

Edit `pyproject.toml`: remove the line `"deap>=1.4.1",` from the `ai = [...]` extra.

- [ ] **Step 5: Re-resolve dependencies**

Run: `uv sync --all-extras`

Expected: `deap` removed from the lockfile; no errors.

- [ ] **Step 6: Run all NSGA-II + registry tests**

Run: `uv run pytest tests/ai/optimizers/test_registry.py tests/ai/optimizers/test_nsga2.py -v -m "not slow"`

Expected: all tests PASS. (Slow convergence test deferred to nightly CI.)

- [ ] **Step 7: Run the slow convergence test as a final smoke check**

Run: `uv run pytest tests/ai/optimizers/test_nsga2.py::test_default_instance_converges -v`

Expected: PASS within 60-90s wall clock. If it fails, the most likely cause is that EvoTorch's tournament selection differs subtly from DEAP's; check the operator outputs and adjust hyperparameters before declaring the test broken.

- [ ] **Step 8: Run the full ai-service smoke check via docker-compose lint**

Run: `uv run python -c "from ai.main import app; print('routes:', [r.path for r in app.routes])"`

Expected: lists `/predict/rl`, `/predict/evolutionary/{algorithm}`, `/predict/ga` (deprecated). No import errors.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: replace DEAP GAOptimizer with EvoTorch NSGAIIOptimizer; drop deap

Single atomic commit so every reverse caller is fixed before DEAP leaves
the dependency set.

- src/ai/optimizers/nsga2.py: NSGAIIOptimizer extending the Optimizer ABC
  (auto-registered as 'nsga2'); thin wrapper over evotorch GeneticAlgorithm
- src/ai/optimizers/__init__.py: eager import for registration
- src/ai/services/optimizer_inference.py: replaces ga_inference.py;
  dispatches to any registered optimizer by name
- src/ai/training/evolutionary.py: replaces training/ga.py with
  --algorithm flag accepting any registered optimizer
- src/ai/api/inference.py: new /predict/evolutionary/{algorithm} route;
  /predict/ga shim kept with deprecated=True for one release
- src/ai/domain/schemas.py: GA* renamed to NSGAII*; CCMO* and Benchmark*
  schemas added; one-release deprecation aliases for old names
- src/frontend/api_client/client.py: call /predict/evolutionary/nsga2
- pyproject.toml: deap removed
- src/ai/optimizers/ga.py: deleted

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 3 — Add CCMO

## Commit 6: CCMOOptimizer

### Task 18: Write failing CCMO tests

**Files:**
- Create: `tests/ai/optimizers/test_ccmo.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Convergence + dual-population invariant tests for CCMOOptimizer."""

import math

import pytest
import torch

from ai.domain.problem import SchedulingProblem


def test_result_shape_correct(tiny_problem: SchedulingProblem):
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.result import CCMOConfig

    optimizer = CCMOOptimizer(tiny_problem)
    result = optimizer.run(CCMOConfig(generations=5, pop_size=20, seed=42))

    assert len(result.best_schedule) == tiny_problem.num_shifts
    assert len(result.best_fitness) == 3
    assert len(result.feasible_pareto_front) >= 1 or result.fell_back_to_auxiliary
    assert len(result.step_history) == 5


def test_pop2_explores_infeasible(default_problem: SchedulingProblem):
    """Pop2 should produce some infeasible members during the run.

    With a moderately-constrained instance, Pop2 (which ignores constraints)
    should drift into infeasible territory at least once.
    """
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.result import CCMOConfig

    optimizer = CCMOOptimizer(default_problem)
    result = optimizer.run(CCMOConfig(generations=10, pop_size=30, seed=42))

    pop2_violations = [s.pop2_mean_violations for s in result.step_history]
    # At least one generation should show pop2 mean violations > 0.
    assert max(pop2_violations) > 0


def test_fall_back_when_no_feasible(over_constrained_problem: SchedulingProblem):
    """An over-constrained instance triggers fell_back_to_auxiliary=True."""
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.result import CCMOConfig

    optimizer = CCMOOptimizer(over_constrained_problem)
    result = optimizer.run(CCMOConfig(generations=10, pop_size=20, seed=42))

    assert result.fell_back_to_auxiliary is True


@pytest.mark.slow
def test_default_instance_converges_to_feasible(default_problem: SchedulingProblem):
    """With seed=42, 50 generations, 100 popsize, CCMO returns a feasible solution."""
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.result import CCMOConfig

    optimizer = CCMOOptimizer(default_problem)
    result = optimizer.run(CCMOConfig(generations=50, pop_size=100, seed=42))

    assert result.fell_back_to_auxiliary is False
    assert result.best_fitness[1] == 0
    assert len(result.feasible_pareto_front) >= 1


@pytest.mark.slow
def test_ccmo_hv_at_least_competitive_with_nsga2(default_problem: SchedulingProblem):
    """CCMO's hypervolume should be ≥ 90% of NSGA-II's at matched seed/budget."""
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import CCMOConfig, NSGAIIConfig

    nsga2 = NSGAIIOptimizer(default_problem).run(
        NSGAIIConfig(generations=50, pop_size=100, seed=42)
    )
    ccmo = CCMOOptimizer(default_problem).run(
        CCMOConfig(generations=50, pop_size=100, seed=42)
    )

    # Lower-is-better on each obj; sum-of-best-fitness as a quick HV proxy.
    nsga2_score = sum(nsga2.best_fitness)
    ccmo_score = sum(ccmo.best_fitness)
    # CCMO should not be more than 10% worse than NSGA-II.
    assert ccmo_score <= nsga2_score * 1.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ai/optimizers/test_ccmo.py -v -m "not slow"`

Expected: `ImportError: cannot import name 'CCMOOptimizer'`.

### Task 19: Implement CCMO selection helpers

**Files:**
- Create: `src/ai/optimizers/ccmo.py` (just helpers in this step; the full optimizer follows)

- [ ] **Step 1: Add a brute-force test for `_nsga2_pareto_ranks`**

Append to `tests/ai/optimizers/test_ccmo.py`:

```python
def test_pareto_ranks_against_brute_force():
    """Vectorized fast non-dominated sort matches naive O(N²) brute force."""
    from ai.optimizers.ccmo import _nsga2_pareto_ranks

    torch.manual_seed(0)
    objs = torch.randn(20, 2)
    fast_ranks = _nsga2_pareto_ranks(objs)

    # Brute force.
    n = objs.shape[0]
    dominates = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            le = (objs[i] <= objs[j]).all()
            lt = (objs[i] < objs[j]).any()
            if le and lt:
                dominates[i, j] = True

    # Brute-force rank assignment.
    domination_count = dominates.sum(dim=0)
    front_idx = torch.where(domination_count == 0)[0]
    brute_ranks = torch.full((n,), -1, dtype=torch.long)
    rank = 0
    seen = front_idx.tolist()
    while front_idx.numel() > 0:
        for i in front_idx.tolist():
            brute_ranks[i] = rank
        # Next front.
        next_front = []
        for i in front_idx.tolist():
            for j in range(n):
                if dominates[i, j]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
        front_idx = torch.tensor(list(set(next_front)))
        rank += 1
    assert torch.equal(fast_ranks, brute_ranks)
```

- [ ] **Step 2: Create `ccmo.py` with helpers (skeleton)**

```python
"""Constrained MOEA via Coevolution (Tian et al. IEEE TEC 2021)."""

from typing import ClassVar

import torch
from evotorch import SolutionBatch

from ai.optimizers.base import Optimizer
from ai.optimizers.operators import (
    DayAlignedCrossOver,
    RepairOperator,
    UniformIntMutation,
)
from ai.optimizers.result import (
    CCMOConfig,
    CCMOResult,
    CCMOStepStatus,
    OptimizerConfig,
    OptimizerResult,
)
from ai.optimizers.rostering_problem import RosteringProblem


def _nsga2_pareto_ranks(obj: torch.Tensor) -> torch.Tensor:
    """Vectorized fast non-dominated sort. Returns (N,) int rank tensor (0 = best front)."""
    n = obj.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=obj.device)
    # i dominates j if (obj[i] <= obj[j]).all() and (obj[i] < obj[j]).any().
    le = (obj.unsqueeze(1) <= obj.unsqueeze(0)).all(dim=-1)  # (N, N)
    lt = (obj.unsqueeze(1) < obj.unsqueeze(0)).any(dim=-1)
    dominates = le & lt  # dominates[i, j] = True iff i dominates j

    domination_count = dominates.sum(dim=0).clone()
    ranks = torch.full((n,), -1, dtype=torch.long, device=obj.device)
    current_rank = 0
    while True:
        front = (domination_count == 0) & (ranks == -1)
        if not front.any():
            break
        ranks[front] = current_rank
        # Decrement domination count for those each member of `front` dominates.
        front_idx = torch.where(front)[0]
        for i in front_idx.tolist():
            domination_count -= dominates[i].long()
        current_rank += 1
    return ranks


def _crowding_distance(obj: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Per-front crowding distance. Returns (N,) float tensor."""
    n = obj.shape[0]
    distances = torch.zeros(n, device=obj.device)
    for r in ranks.unique().tolist():
        idx = torch.where(ranks == r)[0]
        if idx.numel() <= 2:
            distances[idx] = float("inf")
            continue
        for m in range(obj.shape[1]):
            sorted_idx = idx[obj[idx, m].argsort()]
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
            if obj_range == 0:
                continue
            for k in range(1, sorted_idx.numel() - 1):
                distances[sorted_idx[k]] += float(
                    (obj[sorted_idx[k + 1], m] - obj[sorted_idx[k - 1], m]) / obj_range
                )
    return distances


def _select_by_rank_and_crowding(
    pool: SolutionBatch,
    ranks: torch.Tensor,
    obj: torch.Tensor,
    pop_size: int,
) -> SolutionBatch:
    """Select pop_size members from pool, preferring lower rank then higher crowding."""
    distances = _crowding_distance(obj, ranks)
    # Sort by (rank ascending, distance descending).
    sort_keys = ranks.float() - distances * 1e-9  # break ties by negative distance
    order = sort_keys.argsort()
    selected_idx = order[:pop_size].tolist()
    new_batch = SolutionBatch(pool.problem, popsize=pop_size)
    new_batch._data = pool.values[selected_idx].clone()
    new_batch._evals = pool.evals[selected_idx].clone()
    return new_batch
```

- [ ] **Step 3: Run the brute-force comparison test**

Run: `uv run pytest tests/ai/optimizers/test_ccmo.py::test_pareto_ranks_against_brute_force -v`

Expected: PASS.

### Task 20: Implement `CCMOOptimizer.run()`

**Files:**
- Modify: `src/ai/optimizers/ccmo.py` (append)

- [ ] **Step 1: Append the optimizer class**

Add to the end of `src/ai/optimizers/ccmo.py`:

```python
class CCMOOptimizer(Optimizer):
    """Two coevolving populations:
      Pop1 — constraint-aware (Deb's constraint-domination principle)
      Pop2 — unconstrained (NSGA-II on (imbalance, b2b) only)
    Both populations' offspring feed both selections each generation.
    """

    name: ClassVar[str] = "ccmo"
    config_class: ClassVar[type[OptimizerConfig]] = CCMOConfig
    result_class: ClassVar[type[OptimizerResult]] = CCMOResult

    def run(
        self,
        config: CCMOConfig | None = None,
        verbose: bool = False,
    ) -> CCMOResult:
        config = config or CCMOConfig()
        if config.seed is not None:
            torch.manual_seed(config.seed)

        problem = RosteringProblem(self._sp, device=config.device)
        operators = [
            DayAlignedCrossOver(
                problem,
                tournament_size=config.tournament_size,
                cross_over_rate=config.cxpb,
            ),
            UniformIntMutation(problem, indpb=config.indpb, mut_rate=config.mutpb),
            RepairOperator(problem),
        ]

        pop1 = problem.generate_batch(config.pop_size)
        pop2 = problem.generate_batch(config.pop_size)
        problem.evaluate(pop1)
        problem.evaluate(pop2)

        history: list[CCMOStepStatus] = []
        for gen in range(config.generations):
            o1 = self._apply_operators(pop1, operators)
            o2 = self._apply_operators(pop2, operators)
            problem.evaluate(o1)
            problem.evaluate(o2)
            offspring = SolutionBatch.cat([o1, o2])

            pop1 = self._constraint_aware_select(
                SolutionBatch.cat([pop1, offspring]), config.pop_size
            )
            pop2 = self._unconstrained_select(
                SolutionBatch.cat([pop2, offspring]), config.pop_size
            )

            history.append(self._snapshot(gen, pop1, pop2))
            if verbose and gen % 10 == 0:
                self._print_snapshot(history[-1])

        return self._collect_result(pop1, pop2, history)

    @staticmethod
    def _apply_operators(pop: SolutionBatch, operators: list) -> SolutionBatch:
        current = pop
        for op in operators:
            current = op._do(current)
        return current

    @staticmethod
    def _constraint_aware_select(pool: SolutionBatch, pop_size: int) -> SolutionBatch:
        evals = pool.evals
        feasible_mask = evals[:, 1] <= 0.0
        obj = torch.stack([evals[:, 0], evals[:, 2]], dim=1)

        ranks = torch.empty(evals.shape[0], dtype=torch.long, device=evals.device)
        feas_idx = torch.where(feasible_mask)[0]
        inf_idx = torch.where(~feasible_mask)[0]

        if feas_idx.numel() > 0:
            ranks[feas_idx] = _nsga2_pareto_ranks(obj[feas_idx])
        if inf_idx.numel() > 0:
            v_ranks = evals[inf_idx, 1].argsort().argsort()
            ranks[inf_idx] = v_ranks + (
                feas_idx.numel() if feas_idx.numel() > 0 else 0
            ) + 1

        return _select_by_rank_and_crowding(pool, ranks, obj, pop_size)

    @staticmethod
    def _unconstrained_select(pool: SolutionBatch, pop_size: int) -> SolutionBatch:
        evals = pool.evals
        obj = torch.stack([evals[:, 0], evals[:, 2]], dim=1)
        ranks = _nsga2_pareto_ranks(obj)
        return _select_by_rank_and_crowding(pool, ranks, obj, pop_size)

    @staticmethod
    def _snapshot(
        gen: int, pop1: SolutionBatch, pop2: SolutionBatch
    ) -> CCMOStepStatus:
        e1 = pop1.evals
        e2 = pop2.evals
        feas1 = e1[:, 1] <= 0.0
        feas_e1 = e1[feas1] if feas1.any() else e1[:0]
        ranks1 = _nsga2_pareto_ranks(
            torch.stack([feas_e1[:, 0], feas_e1[:, 2]], dim=1)
        ) if feas_e1.numel() > 0 else torch.empty(0, dtype=torch.long)
        ranks2 = _nsga2_pareto_ranks(torch.stack([e2[:, 0], e2[:, 2]], dim=1))
        return CCMOStepStatus(
            generation=gen,
            pop1_feasible_count=int(feas1.sum()),
            pop1_best_imbalance=float(feas_e1[:, 0].min()) if feas_e1.numel() > 0 else float("nan"),
            pop1_best_b2b=float(feas_e1[:, 2].min()) if feas_e1.numel() > 0 else float("nan"),
            pop1_pareto_size=int((ranks1 == 0).sum()) if ranks1.numel() > 0 else 0,
            pop2_pareto_size=int((ranks2 == 0).sum()),
            pop2_mean_violations=float(e2[:, 1].mean()),
        )

    @staticmethod
    def _print_snapshot(s: CCMOStepStatus) -> None:
        print(
            f"gen={s.generation} pop1_feas={s.pop1_feasible_count} "
            f"pop1_pareto={s.pop1_pareto_size} pop2_pareto={s.pop2_pareto_size}"
        )

    def _collect_result(
        self,
        pop1: SolutionBatch,
        pop2: SolutionBatch,
        history: list[CCMOStepStatus],
    ) -> CCMOResult:
        e1 = pop1.evals
        feas_mask = e1[:, 1] <= 0.0

        if feas_mask.any():
            feas_idx = torch.where(feas_mask)[0]
            feas_obj = torch.stack(
                [e1[feas_idx, 0], e1[feas_idx, 2]], dim=1
            )
            ranks = _nsga2_pareto_ranks(feas_obj)
            front_local = torch.where(ranks == 0)[0]
            front_idx = feas_idx[front_local]
            best_local = front_idx[
                e1[front_idx].sum(dim=1).argmin()
            ].item()
            fell_back = False
        else:
            # Pop1 has no feasibles; fall back to lowest-violation auxiliary member.
            best_local = int(e1[:, 1].argmin())
            front_idx = torch.tensor([best_local])
            fell_back = True

        # Feasible front from Pop1.
        feas_pareto_front = [list(map(int, pop1.values[i].tolist())) for i in front_idx.tolist()]
        feas_pareto_fits = [
            tuple(map(float, pop1.evals[i].tolist())) for i in front_idx.tolist()
        ]
        # Auxiliary front from Pop2.
        aux_obj = torch.stack([pop2.evals[:, 0], pop2.evals[:, 2]], dim=1)
        aux_ranks = _nsga2_pareto_ranks(aux_obj)
        aux_front_idx = torch.where(aux_ranks == 0)[0].tolist()
        aux_pareto_front = [list(map(int, pop2.values[i].tolist())) for i in aux_front_idx]
        aux_pareto_fits = [
            tuple(map(float, pop2.evals[i].tolist())) for i in aux_front_idx
        ]

        return CCMOResult(
            best_schedule=list(map(int, pop1.values[best_local].tolist())),
            best_fitness=tuple(map(float, e1[best_local].tolist())),
            pareto_front=feas_pareto_front,
            pareto_fitnesses=feas_pareto_fits,
            feasible_pareto_front=feas_pareto_front,
            feasible_pareto_fitnesses=feas_pareto_fits,
            auxiliary_pareto_front=aux_pareto_front,
            auxiliary_pareto_fitnesses=aux_pareto_fits,
            step_history=history,
            fell_back_to_auxiliary=fell_back,
        )
```

- [ ] **Step 2: Update `optimizers/__init__.py` to register CCMO**

Edit `src/ai/optimizers/__init__.py`:

```python
"""Optimizer package."""

from ai.optimizers.base import Optimizer
from ai.optimizers.ccmo import CCMOOptimizer  # noqa: F401 — import for registration
from ai.optimizers.nsga2 import NSGAIIOptimizer  # noqa: F401 — import for registration

__all__ = ["Optimizer", "NSGAIIOptimizer", "CCMOOptimizer"]
```

- [ ] **Step 3: Restore the registry tests' ccmo assertion**

Edit `tests/ai/optimizers/test_registry.py`:

In `test_init_subclass_registers`, add back:
```python
    assert "ccmo" in available
```

In `test_create_unknown_raises`, add back:
```python
    assert "ccmo" in str(exc.value)
```

- [ ] **Step 4: Run all CCMO + registry tests**

Run: `uv run pytest tests/ai/optimizers/test_ccmo.py tests/ai/optimizers/test_registry.py -v -m "not slow"`

Expected: all tests PASS.

- [ ] **Step 5: Run the slow CCMO tests as a final check**

Run: `uv run pytest tests/ai/optimizers/test_ccmo.py -v -m "slow"`

Expected: 2 tests PASS (`test_default_instance_converges_to_feasible`, `test_ccmo_hv_at_least_competitive_with_nsga2`). Wall clock ≈ 2-3 min total.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: CCMOOptimizer — Constrained MOEA via Coevolution (Tian et al. 2021)

Two-population coevolution loop sharing the same RosteringProblem and
operators as NSGAIIOptimizer; auto-registered as 'ccmo' via
__init_subclass__. Pop1 selection uses Deb's constraint-domination on
(imbalance, b2b) with violations as constraint; Pop2 is unconstrained
NSGA-II on (imbalance, b2b). Best-solution selection falls back to
Pop2's lowest-violation auxiliary member if Pop1 has no feasible
solution at termination.

Closes #15.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Phase 4 — INRC-I benchmark

## Commit 7: INRC-I parser + loader + fixture + fetch script

### Task 21: Bundle the sprint01 fixture and write parser tests

**Files:**
- Create: `tests/fixtures/inrc1/sprint01.txt` (literal INRC-I content)
- Create: `tests/ai/benchmarks/__init__.py`
- Create: `tests/ai/benchmarks/inrc1/__init__.py`
- Create: `tests/ai/benchmarks/inrc1/test_parser.py`

- [ ] **Step 1: Obtain the sprint01 instance**

Run: `mkdir -p tests/fixtures/inrc1`

Then download `sprint01.txt` from the INRC-I corpus. The most reliable source is the SAPS (KU Leuven) mirror or the official competition archive. If unsure of the URL, the implementer should:

1. Search "INRC-I sprint01.txt" or check https://www.kuleuven-kortrijk.be/sse/dwsg/research/nrp_test_instances
2. Save the content to `tests/fixtures/inrc1/sprint01.txt`
3. Verify the file starts with the line `SCHEDULING_PERIOD` and contains sections labeled `SKILLS`, `SHIFT_TYPES`, `CONTRACTS`, `EMPLOYEES`, `DAY_OFF_REQUESTS` (etc.).

If the canonical mirror is unreachable, the implementer can construct a synthetic minimal instance matching the INRC-I format spec from Haspeslagh et al. 2014 §3.1 — but the test then becomes a structural sanity check, not a parse-published-data check.

- [ ] **Step 2: Create test package markers**

Run:
```bash
mkdir -p tests/ai/benchmarks/inrc1
touch tests/ai/benchmarks/__init__.py tests/ai/benchmarks/inrc1/__init__.py
```

- [ ] **Step 3: Write the failing parser test**

```python
"""Tests for the INRC-I instance parser."""

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent.parent.parent.parent / "fixtures" / "inrc1" / "sprint01.txt"


def test_parses_sprint01_fixture():
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    text = FIXTURE_PATH.read_text()
    instance = parse_inrc1_instance(text)

    # Sprint01 published metadata (Haspeslagh et al. 2014 Table 1):
    # 10 nurses, 28 days, 4 shift types, 4 contracts.
    assert instance.num_nurses == 10
    assert instance.num_days == 28
    assert len(instance.shift_types) == 4
    assert len(instance.contracts) == 4
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/ai/benchmarks/inrc1/test_parser.py -v`

Expected: `ImportError: cannot import name 'parse_inrc1_instance'`.

### Task 22: Implement the INRC-I parser

**Files:**
- Create: `src/ai/benchmarks/__init__.py`
- Create: `src/ai/benchmarks/inrc1/__init__.py`
- Create: `src/ai/benchmarks/inrc1/parser.py`

- [ ] **Step 1: Create package markers**

Run:
```bash
mkdir -p src/ai/benchmarks/inrc1
touch src/ai/benchmarks/__init__.py src/ai/benchmarks/inrc1/__init__.py
```

- [ ] **Step 2: Implement the parser**

```python
"""INRC-I instance parser.

Parses the .txt format defined by Haspeslagh et al. (2014, Annals of OR
218(1)). Returns a structured intermediate that preserves the full
INRC-I shape so the lossy adapter to SchedulingProblem doesn't have to
re-parse.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InrcShiftType:
    name: str
    start_time: str       # "HH:MM"
    end_time: str
    duration_minutes: int


@dataclass(frozen=True)
class InrcContract:
    name: str
    max_assignments: int
    min_assignments: int


@dataclass(frozen=True)
class InrcNurse:
    name: str
    contract: str
    skills: tuple[str, ...]


@dataclass(frozen=True)
class InrcDayOffRequest:
    nurse: str
    day: int


@dataclass(frozen=True)
class InrcInstance:
    name: str
    num_nurses: int
    num_days: int
    shift_types: tuple[InrcShiftType, ...]
    contracts: tuple[InrcContract, ...]
    nurses: tuple[InrcNurse, ...]
    day_off_requests: tuple[InrcDayOffRequest, ...]
    # Other fields (skills, patterns, weekend rotation, etc.) intentionally not modeled
    # in this initial parser — they are parsed as opaque sections and not exposed.


def parse_inrc1_instance(text: str) -> InrcInstance:
    """Parse a single INRC-I .txt file into an InrcInstance.

    The parser is forgiving of whitespace and section ordering. Sections
    that this loader doesn't model (PATTERNS, etc.) are read but discarded.
    """
    sections = _split_sections(text)
    period = sections.get("SCHEDULING_PERIOD", {})
    name = period.get("ID", "unknown")
    num_days = int(period.get("Days", 0))

    shift_types = tuple(_parse_shift_types(sections.get("SHIFT_TYPES", "")))
    contracts = tuple(_parse_contracts(sections.get("CONTRACTS", "")))
    nurses = tuple(_parse_nurses(sections.get("EMPLOYEES", "")))
    day_off_requests = tuple(_parse_day_offs(sections.get("DAY_OFF_REQUESTS", "")))

    return InrcInstance(
        name=name,
        num_nurses=len(nurses),
        num_days=num_days,
        shift_types=shift_types,
        contracts=contracts,
        nurses=nurses,
        day_off_requests=day_off_requests,
    )


def _split_sections(text: str) -> dict:
    """Split the .txt into a {section_name: section_text} dict.

    INRC-I sections start with a header like 'SECTION_NAME = {' and end at
    the matching '}'. This is a lightweight bracket-matching parser.
    """
    sections: dict[str, str] = {}
    i = 0
    while i < len(text):
        # Find next 'KEYWORD = {' or 'KEYWORD ='.
        eq = text.find("=", i)
        if eq < 0:
            break
        # Section name is the last identifier before '='.
        before = text[:eq].rstrip()
        last_ws = max(before.rfind(c) for c in (" ", "\n", "\t", "{", "}"))
        section_name = before[last_ws + 1:].strip() or "UNKNOWN"
        # Section body starts after '=' or '= {'.
        body_start = eq + 1
        while body_start < len(text) and text[body_start] in " \t\n":
            body_start += 1
        if body_start < len(text) and text[body_start] == "{":
            depth = 1
            j = body_start + 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            body = text[body_start + 1: j - 1].strip()
            i = j
        else:
            # Single-line value.
            line_end = text.find("\n", body_start)
            if line_end < 0:
                line_end = len(text)
            body = text[body_start:line_end].strip()
            i = line_end
        sections[section_name] = body if section_name != "SCHEDULING_PERIOD" else _parse_kv(body)
    return sections


def _parse_kv(text: str) -> dict:
    """Parse 'Key = Value' lines into a dict."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _parse_shift_types(text: str):
    """One ShiftType per line: 'Name, HH:MM, HH:MM, duration_min'."""
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            yield InrcShiftType(
                name=parts[0],
                start_time=parts[1],
                end_time=parts[2],
                duration_minutes=int(parts[3]),
            )


def _parse_contracts(text: str):
    """Best-effort parser; returns InrcContract with whatever fields we can extract."""
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                yield InrcContract(
                    name=parts[0],
                    max_assignments=int(parts[1]),
                    min_assignments=int(parts[2]),
                )
            except ValueError:
                continue


def _parse_nurses(text: str):
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            yield InrcNurse(
                name=parts[0],
                contract=parts[1],
                skills=tuple(parts[2:]) if len(parts) > 2 else (),
            )


def _parse_day_offs(text: str):
    for line in text.splitlines():
        line = line.strip().rstrip(",;")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                yield InrcDayOffRequest(nurse=parts[0], day=int(parts[1]))
            except ValueError:
                continue
```

- [ ] **Step 3: Run parser test**

Run: `uv run pytest tests/ai/benchmarks/inrc1/test_parser.py -v`

Expected: PASS. (If the actual sprint01.txt format differs from the assumed `KEYWORD = { ... }` shape, adjust the parser. The numbers asserted (10 nurses, 28 days, 4 shift types) are from Haspeslagh et al. 2014 Table 1 and are non-negotiable.)

### Task 23: Implement INRC-I loader + manifest + fetch script

**Files:**
- Create: `src/ai/benchmarks/inrc1/loader.py`
- Create: `src/ai/benchmarks/inrc1/manifest.json`
- Create: `src/ai/benchmarks/inrc1/__init__.py` (overwriting if needed)
- Create: `scripts/fetch_inrc1.py`
- Create: `tests/ai/benchmarks/inrc1/test_loader.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `data/benchmarks/inrc1/` to `.gitignore`**

Append to `.gitignore`:

```gitignore

# Benchmark instances (downloaded on first use)
data/benchmarks/inrc1/
```

- [ ] **Step 2: Create the manifest**

```json
{
  "sprint": [
    {"name": "sprint01", "num_nurses": 10, "num_days": 28},
    {"name": "sprint02", "num_nurses": 10, "num_days": 28},
    {"name": "sprint03", "num_nurses": 10, "num_days": 28},
    {"name": "sprint04", "num_nurses": 10, "num_days": 28},
    {"name": "sprint05", "num_nurses": 10, "num_days": 28},
    {"name": "sprint06", "num_nurses": 10, "num_days": 28},
    {"name": "sprint07", "num_nurses": 10, "num_days": 28},
    {"name": "sprint08", "num_nurses": 10, "num_days": 28},
    {"name": "sprint09", "num_nurses": 10, "num_days": 28},
    {"name": "sprint10", "num_nurses": 10, "num_days": 28}
  ]
}
```

- [ ] **Step 3: Implement the loader**

```python
"""INRC-I → SchedulingProblem lossy adapter.

Maps a parsed InrcInstance to our SchedulingProblem domain model. Drops:
  - Nurse skills/grades
  - Contract patterns (5/2 etc.)
  - Weekend rotation constraints
  - Shift-type sequencing constraints (e.g. late→early forbidden)
  - Soft-constraint weights from the original INRC-I scoring

Mapped:
  - num_nurses → num_employees
  - num_days → days
  - shift_types (count) → shifts_per_day
  - per-shift duration → shift_lengths
  - Contract.max_assignments × shift duration → max_hours per nurse
  - DAY_OFF_REQUESTS → unavailability
"""

import json
from pathlib import Path

from ai.benchmarks.inrc1.parser import InrcInstance, parse_inrc1_instance
from ai.domain.problem import SchedulingProblem

DATA_DIR = Path("data/benchmarks/inrc1")
MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def list_instances(track: str = "sprint") -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return [item["name"] for item in manifest.get(track, [])]


def load_instance(name: str) -> SchedulingProblem:
    """Load an INRC-I instance by name as a SchedulingProblem (lossy)."""
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"INRC-I instance '{name}' not found at {path}. "
            "Run `python -m scripts.fetch_inrc1` to download the corpus."
        )
    instance = parse_inrc1_instance(path.read_text())
    return _to_scheduling_problem(instance)


def _to_scheduling_problem(instance: InrcInstance) -> SchedulingProblem:
    num_employees = instance.num_nurses
    days = instance.num_days
    shift_types = instance.shift_types
    shifts_per_day = len(shift_types)
    shift_lengths = tuple(s.duration_minutes // 60 for s in shift_types)

    # Per-nurse max hours: lookup contract by name, multiply max_assignments by avg shift hours.
    avg_shift_hours = (
        sum(s.duration_minutes for s in shift_types) / max(len(shift_types), 1)
    ) / 60
    contract_by_name = {c.name: c for c in instance.contracts}
    max_hours_list: list[int] = []
    employee_types_list: list[str] = []
    for nurse in instance.nurses:
        contract = contract_by_name.get(nurse.contract)
        if contract is None:
            max_hours_list.append(int(days * 8))  # fallback: ~full-time
            employee_types_list.append("FT")
        else:
            max_hours_list.append(
                int(contract.max_assignments * avg_shift_hours)
            )
            employee_types_list.append(
                "FT" if contract.max_assignments >= days * 0.5 else "PT"
            )

    # Day-off requests → unavailability.
    nurse_idx = {n.name: i for i, n in enumerate(instance.nurses)}
    unavailability = frozenset(
        (req.day, nurse_idx[req.nurse])
        for req in instance.day_off_requests
        if req.nurse in nurse_idx
    )

    return SchedulingProblem(
        num_employees=num_employees,
        employee_types=tuple(employee_types_list),
        days=days,
        shifts_per_day=shifts_per_day,
        shift_lengths=shift_lengths,
        max_hours=tuple(max_hours_list),
        unavailability=unavailability,
    )
```

- [ ] **Step 4: Write the failing loader test**

Create `tests/ai/benchmarks/inrc1/test_loader.py`:

```python
"""INRC-I lossy adapter tests."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def stub_inrc1_data(tmp_path, monkeypatch):
    """Copy the bundled sprint01 fixture into the data dir for the loader."""
    fixture = (
        Path(__file__).parent.parent.parent.parent / "fixtures" / "inrc1" / "sprint01.txt"
    )
    target_dir = tmp_path / "data" / "benchmarks" / "inrc1"
    target_dir.mkdir(parents=True)
    shutil.copy(fixture, target_dir / "sprint01.txt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ai.benchmarks.inrc1.loader.DATA_DIR", target_dir)
    return target_dir


def test_loads_as_scheduling_problem(stub_inrc1_data):
    from ai.benchmarks.inrc1.loader import load_instance

    sp = load_instance("sprint01")
    assert sp.num_employees == 10
    assert sp.days == 28
    assert sp.shifts_per_day == 4


def test_load_missing_raises(tmp_path, monkeypatch):
    from ai.benchmarks.inrc1.loader import load_instance

    monkeypatch.setattr(
        "ai.benchmarks.inrc1.loader.DATA_DIR", tmp_path / "nope"
    )
    with pytest.raises(FileNotFoundError) as exc:
        load_instance("sprint01")
    assert "fetch_inrc1" in str(exc.value)


def test_list_instances():
    from ai.benchmarks.inrc1.loader import list_instances

    names = list_instances("sprint")
    assert len(names) == 10
    assert "sprint01" in names
```

- [ ] **Step 5: Run loader tests**

Run: `uv run pytest tests/ai/benchmarks/inrc1/test_loader.py -v`

Expected: all 3 tests PASS.

- [ ] **Step 6: Update `benchmarks/inrc1/__init__.py`**

```python
"""INRC-I benchmark loader."""

from ai.benchmarks.inrc1.loader import list_instances, load_instance

__all__ = ["load_instance", "list_instances"]
```

- [ ] **Step 7: Create the fetch script**

```python
"""Download INRC-I instances on first use.

INRC-I is a public dataset; we don't bundle it in the repo to keep things
lean and to side-step distribution licensing ambiguity.

Usage:
    python -m scripts.fetch_inrc1
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CANDIDATE_URLS = [
    # Add canonical mirror URLs in preference order. The implementer
    # validates these at first run; if all 404, instructions print below.
    "https://www.kuleuven-kortrijk.be/sse/dwsg/research/nrp_test_instances/inrc1.zip",
    "https://github.com/nielshenrik/INRC-I-instances/releases/download/v1/inrc1.zip",
]
TARGET_DIR = Path("data/benchmarks/inrc1")


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for url in CANDIDATE_URLS:
        try:
            print(f"Attempting download from {url}")
            data = urllib.request.urlopen(url, timeout=30).read()
            tmp = TARGET_DIR / "inrc1.zip"
            tmp.write_bytes(data)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(TARGET_DIR)
            tmp.unlink()
            print(f"INRC-I corpus extracted to {TARGET_DIR}")
            return 0
        except (urllib.error.URLError, zipfile.BadZipFile) as e:
            print(f"  FAILED: {e}")
    print(
        "\nAll mirrors failed. Place the INRC-I sprint instances manually:\n"
        f"  1. Obtain inrc1 instance .txt files from Haspeslagh et al. 2014\n"
        f"     (Annals of OR 218(1), https://doi.org/10.1007/s10479-014-1683-6)\n"
        f"  2. Place files at {TARGET_DIR}/sprint01.txt ... sprint10.txt\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Run all benchmark tests**

Run: `uv run pytest tests/ai/benchmarks -v`

Expected: parser test + 3 loader tests all PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: INRC-I sprint-track parser, lossy loader, and fetch script

- src/ai/benchmarks/inrc1/parser.py: parses .txt format from
  Haspeslagh et al. 2014 into a structured InrcInstance
- src/ai/benchmarks/inrc1/loader.py: maps InrcInstance to
  SchedulingProblem (lossy: drops skills, contract patterns, weekend
  rotation, shift sequencing). load_instance raises
  FileNotFoundError pointing to scripts/fetch_inrc1.py if data missing
- src/ai/benchmarks/inrc1/manifest.json: sprint instances metadata
- scripts/fetch_inrc1.py: first-use download with multi-mirror fallback
- tests/fixtures/inrc1/sprint01.txt: bundled fixture for parser tests
- .gitignore: data/benchmarks/inrc1/ excluded

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 8: Benchmark runner + CLI

### Task 24: Implement the benchmark runner

**Files:**
- Create: `src/ai/benchmarks/runner.py`
- Create: `tests/ai/training/__init__.py`
- Create: `tests/ai/training/test_benchmark_smoke.py`

- [ ] **Step 1: Create the runner**

```python
"""A/B benchmark runner: hypervolume + Wilcoxon signed-rank.

Runs registered optimizers across a list of INRC-I instances × seeds and
emits a BenchmarkReport. Hypervolume is computed only on the feasible
front to prevent reward-hacking via large infeasible fronts.
"""

import statistics
import time
from typing import Any

from pymoo.indicators.hv import HV
from scipy.stats import wilcoxon

from ai.benchmarks.inrc1.loader import load_instance
from ai.domain.schemas import BenchmarkAggregate, BenchmarkReport, BenchmarkRunRecord
from ai.optimizers.base import Optimizer

REFERENCE_POINT = (1.0, 1000.0, 100.0)


def run_benchmark(
    algorithms: list[str],
    instance_names: list[str],
    seeds: list[int],
    config_overrides: dict[str, Any] | None = None,
    reference_point: tuple[float, float, float] = REFERENCE_POINT,
) -> BenchmarkReport:
    records: list[BenchmarkRunRecord] = []
    config_overrides = config_overrides or {}

    for algo in algorithms:
        for instance_name in instance_names:
            problem = load_instance(instance_name)
            for seed in seeds:
                optimizer = Optimizer.create(algo, problem)
                config = optimizer.config_class(seed=seed, **config_overrides)

                t0 = time.perf_counter()
                result = optimizer.run(config)
                wall_clock = time.perf_counter() - t0

                feasible_fits = [
                    f for f in result.pareto_fitnesses if f[1] <= 0.0
                ]
                hv = _compute_hypervolume(feasible_fits, reference_point)

                records.append(
                    BenchmarkRunRecord(
                        instance=instance_name,
                        algorithm=algo,
                        seed=seed,
                        hypervolume=hv,
                        feasible_front_size=len(feasible_fits),
                        best_imbalance=result.best_fitness[0],
                        best_violations=result.best_fitness[1],
                        best_b2b=int(result.best_fitness[2]),
                        wall_clock_s=wall_clock,
                    )
                )

    aggregate = _aggregate(records)
    return BenchmarkReport(
        config_summary={
            "algorithms": algorithms,
            "instance_count": len(instance_names),
            "seeds": seeds,
            "config_overrides": config_overrides,
            "reference_point": list(reference_point),
        },
        per_run=records,
        aggregate=aggregate,
    )


def _compute_hypervolume(
    fits: list[tuple[float, float, float]],
    ref: tuple[float, float, float],
) -> float:
    if not fits:
        return 0.0
    import numpy as np

    indicator = HV(ref_point=np.array(ref))
    return float(indicator(np.array(fits)))


def _aggregate(records: list[BenchmarkRunRecord]) -> list[BenchmarkAggregate]:
    by_instance: dict[str, dict[str, list[float]]] = {}
    for r in records:
        by_instance.setdefault(r.instance, {}).setdefault(r.algorithm, []).append(r.hypervolume)

    out: list[BenchmarkAggregate] = []
    for instance, by_algo in by_instance.items():
        nsga = by_algo.get("nsga2", [])
        ccmo = by_algo.get("ccmo", [])
        wp = None
        if len(nsga) == len(ccmo) and len(nsga) >= 6:  # minimum for Wilcoxon
            try:
                wp = float(wilcoxon(nsga, ccmo).pvalue)
            except ValueError:
                wp = None

        out.append(
            BenchmarkAggregate(
                instance=instance,
                nsga2_hv_mean=statistics.mean(nsga) if nsga else None,
                nsga2_hv_std=statistics.stdev(nsga) if len(nsga) > 1 else None,
                nsga2_n_seeds=len(nsga),
                ccmo_hv_mean=statistics.mean(ccmo) if ccmo else None,
                ccmo_hv_std=statistics.stdev(ccmo) if len(ccmo) > 1 else None,
                ccmo_n_seeds=len(ccmo),
                wilcoxon_p=wp,
            )
        )
    return out
```

- [ ] **Step 2: Write the failing benchmark smoke test**

```python
"""Smoke test: run NSGA-II for 5 generations on sprint01 × 1 seed."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def stub_inrc1_data(tmp_path, monkeypatch):
    fixture = (
        Path(__file__).parent.parent.parent / "fixtures" / "inrc1" / "sprint01.txt"
    )
    target_dir = tmp_path / "data" / "benchmarks" / "inrc1"
    target_dir.mkdir(parents=True)
    shutil.copy(fixture, target_dir / "sprint01.txt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ai.benchmarks.inrc1.loader.DATA_DIR", target_dir)
    return target_dir


@pytest.mark.benchmark
def test_run_nsga2_on_sprint01_one_seed_smoke(stub_inrc1_data):
    from ai.benchmarks.runner import run_benchmark

    report = run_benchmark(
        algorithms=["nsga2"],
        instance_names=["sprint01"],
        seeds=[42],
        config_overrides={"generations": 5, "pop_size": 20},
    )
    assert len(report.per_run) == 1
    assert report.per_run[0].hypervolume >= 0
    assert report.per_run[0].wall_clock_s < 30.0
```

- [ ] **Step 3: Run the smoke test**

Run: `uv run pytest tests/ai/training/test_benchmark_smoke.py -v -m "benchmark"`

Expected: PASS within 30s wall clock.

### Task 25: Implement the benchmark CLI

**Files:**
- Create: `src/ai/training/benchmark.py`

- [ ] **Step 1: Create the CLI**

```python
"""Benchmark CLI: A/B compare optimizers on the INRC-I sprint track.

Usage:
    python -m ai.training.benchmark --algorithm nsga2 --track sprint --seeds 10
    python -m ai.training.benchmark --algorithm nsga2,ccmo --track sprint --seeds 10 --report a_b.json
"""

import argparse
import json
from pathlib import Path

from ai.benchmarks.inrc1.loader import list_instances
from ai.benchmarks.runner import run_benchmark
from ai.optimizers.base import Optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INRC-I benchmark A/B")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Comma-separated optimizer names (e.g. 'nsga2' or 'nsga2,ccmo').",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="sprint",
        choices=["sprint"],
        help="INRC-I track. Only 'sprint' is supported in this PR.",
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="Number of seeds per algorithm × instance."
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--report", type=str, default=None, help="Path to write JSON report.")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithm.split(",")]
    for a in algorithms:
        if a not in Optimizer.list_available():
            raise SystemExit(
                f"Unknown algorithm '{a}'. Available: {Optimizer.list_available()}"
            )

    instances = list_instances(args.track)
    seeds = list(range(args.seeds))

    report = run_benchmark(
        algorithms=algorithms,
        instance_names=instances,
        seeds=seeds,
        config_overrides={
            "generations": args.generations,
            "pop_size": args.pop_size,
        },
    )

    output = json.dumps(report.model_dump(), indent=2)
    if args.report:
        Path(args.report).write_text(output)
        print(f"Report saved to {args.report}")
    else:
        print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI starts**

Run: `uv run python -m ai.training.benchmark --help`

Expected: argparse help message; no errors.

- [ ] **Step 3: Run all tests as a final integration check**

Run: `uv run pytest tests/ -v -m "not slow and not benchmark"`

Expected: all fast tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: A/B benchmark runner and CLI with hypervolume + Wilcoxon

- src/ai/benchmarks/runner.py: run_benchmark over algorithms × instances ×
  seeds; HV via pymoo (feasible-only); Wilcoxon p-value via scipy
- src/ai/training/benchmark.py: CLI with --algorithm comma-list and
  --report JSON output
- tests/ai/training/test_benchmark_smoke.py: 5-gen × 1-seed × 1-instance
  smoke under @pytest.mark.benchmark

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 9: Documentation

### Task 26: Update README and add BENCHMARKS.md

**Files:**
- Modify: `README.md`
- Create: `BENCHMARKS.md`

- [ ] **Step 1: Update README.md**

Append to README's "Repo layout" section (after the existing block) and add a new section:

```markdown
## Optimizers

The AI service ships two evolutionary optimizers, both registered on the
`Optimizer` ABC and selectable by name:

| Algorithm | Module | Notes |
|---|---|---|
| `nsga2` | `ai.optimizers.nsga2.NSGAIIOptimizer` | EvoTorch GeneticAlgorithm + day-aligned crossover, repair |
| `ccmo` | `ai.optimizers.ccmo.CCMOOptimizer` | Constrained MOEA via Coevolution (Tian et al. 2021) |

Inference: `POST /predict/evolutionary/{algorithm}` (e.g. `/predict/evolutionary/nsga2`).
The legacy `POST /predict/ga` is **deprecated** and forwards to `nsga2`.

### Training

```bash
python -m ai.training.evolutionary --algorithm nsga2 --generations 200 --pop-size 100
python -m ai.training.evolutionary --algorithm ccmo  --generations 200 --pop-size 100 --device cuda
```

### Benchmark

See `BENCHMARKS.md` for the INRC-I A/B harness.
```

- [ ] **Step 2: Create BENCHMARKS.md**

```markdown
# Benchmarks

## INRC-I sprint track — A/B between NSGA-II and CCMO

The benchmark harness compares our two evolutionary optimizers on the
sprint track of the **First International Nurse Rostering Competition**
(Haspeslagh et al. 2014, *Annals of Operations Research* 218(1)).

### What this is and isn't

- **What it is:** an internal A/B harness between `NSGAIIOptimizer` and
  `CCMOOptimizer` on real-shaped instances, using hypervolume on the
  feasible Pareto front and Wilcoxon signed-rank for statistical
  significance.
- **What it isn't:** a comparison against the published INRC-I leaderboard.
  Our `SchedulingProblem` is simpler than the full INRC-I model — it drops
  nurse skills, contract patterns, weekend rotation, shift sequencing, and
  the original soft-constraint weights. The hypervolume scores reported
  here are **not** INRC-I aggregate competition scores.

### Setup

```bash
python -m scripts.fetch_inrc1     # one-time download to data/benchmarks/inrc1/
```

### Running

```bash
# Single algorithm
python -m ai.training.benchmark --algorithm nsga2 --track sprint --seeds 10 --report nsga2.json
python -m ai.training.benchmark --algorithm ccmo  --track sprint --seeds 10 --report ccmo.json

# A/B
python -m ai.training.benchmark --algorithm nsga2,ccmo --track sprint --seeds 10 --report a_b.json
```

### What gets reported

- **Per run:** instance, algorithm, seed, hypervolume, feasible-front size,
  best (`imbalance`, `violations`, `b2b`), wall clock.
- **Aggregate:** per-instance HV mean and std for each algorithm + Wilcoxon
  p-value when both algorithms have ≥6 matched seeds.

### Reference point

`(imbalance, violations, b2b)` reference point: `(1.0, 1000.0, 100.0)`. This
dominates any plausible objective tuple on the sprint track.
```

- [ ] **Step 3: Verify rendering**

Run: `cat README.md BENCHMARKS.md`

Inspect the output for malformed Markdown or stray brackets.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: README and BENCHMARKS.md for evolutionary optimizers and INRC-I

- README: optimizer registry summary, inference route shape, training and
  benchmark CLI invocations, deprecation note for /predict/ga
- BENCHMARKS.md: scope of the INRC-I A/B harness, lossy-mapping caveat,
  setup, runner usage, reported metrics

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# Final Verification

### Task 27: Whole-suite + smoke tests + slow-mark sweep

- [ ] **Step 1: Run the full fast suite**

Run: `uv run pytest tests/ -v -m "not slow and not benchmark"`

Expected: all tests PASS, ≥80% coverage on `ai.optimizers` and `ai.benchmarks` modules.

- [ ] **Step 2: Run slow tests**

Run: `uv run pytest tests/ -v -m "slow"`

Expected: 3 tests PASS — `test_default_instance_converges` (NSGA-II), `test_default_instance_converges_to_feasible` (CCMO), `test_ccmo_hv_at_least_competitive_with_nsga2`. Total wall clock ≈ 3-4 min.

- [ ] **Step 3: Run benchmark smoke test**

Run: `uv run pytest tests/ -v -m "benchmark"`

Expected: 1 test PASSES within 30s.

- [ ] **Step 4: API smoke test**

Run: `uv run python -c "from ai.main import app; print(sorted([r.path for r in app.routes]))"`

Expected: list including `/health`, `/predict/evolutionary/{algorithm}`, `/predict/ga`, `/predict/rl`. No errors.

- [ ] **Step 5: Training CLI smoke test**

```bash
uv run python -m ai.training.evolutionary --algorithm nsga2 --generations 5 --pop-size 20 --seed 42
uv run python -m ai.training.evolutionary --algorithm ccmo  --generations 5 --pop-size 20 --seed 42
```

Expected: both finish without exception; `checkpoints/{nsga2,ccmo}_best_schedule.json` and `..._step_history.json` written.

- [ ] **Step 6: Commit if any small fixes were needed**

If any of Steps 1-5 surfaced issues that needed inline fixes, commit them as separate small commits with messages of the form:

```
fix: <short description>

Discovered during final verification of <task>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Otherwise, the plan is complete.

---

# Plan summary

- **9 commits** total, each leaving the tree in a working state.
- **27 tasks** with bite-sized 2–5 minute steps.
- **DEAP retired** in commit 5; until then it co-exists with EvoTorch.
- **Tests added in this PR:** registry, RosteringProblem, operators, NSGA-II convergence, CCMO convergence + dual-pop + fallback + brute-force-vs-fast Pareto sort, INRC-I parser + loader, benchmark runner smoke. ≥80% coverage gate enforced via pyproject.
- **Frontend caller** updated within the same PR (commit 5) so we never ship a frontend that pings a deprecated route.
- **CP-SAT design** (in `docs/superpowers/specs/2026-05-10-cpsat-baseline-design.md`) resumes after this PR merges.
