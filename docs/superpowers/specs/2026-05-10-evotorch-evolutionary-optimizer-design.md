# EvoTorch evolutionary optimizer refactor — design

**Status:** approved through brainstorming on 2026-05-10. Ready for implementation plan.

**Tracks:** GitHub issues [#15](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/15) (CCMO; roadmap item #2 of survey [#13](https://github.com/FJCU-AI-APPLICATION/Job_Scheduler_System/issues/13)) plus a framework swap (DEAP → EvoTorch) and INRC-I benchmark adoption that aren't tracked in any single issue today. Closes #15 directly; the framework swap and benchmark are scoped here.

**Branch:** new feature branch off `main` (the CP-SAT design at `docs/superpowers/specs/2026-05-10-cpsat-baseline-design.md` is queued behind this one and resumes after merge).

## Goals

- Replace DEAP with EvoTorch as the evolutionary-algorithm framework. EvoTorch is GPU-native and integrates cleanly with the rest of the AI service's PyTorch stack.
- Rename the current generic `GAOptimizer` to `NSGAIIOptimizer` so the algorithm is named exactly. Preserve all current behavior of NSGA-II.
- Add `CCMOOptimizer` (Tian et al. IEEE TEC 2021, "A Coevolutionary Framework for Constrained Multiobjective Optimization Problems"). Closes issue #15.
- Adopt INRC-I (sprint track) as a benchmark suite for A/B comparing the two algorithms.
- Encapsulate both optimizers under a common `Optimizer` ABC with `__init_subclass__` auto-registration so the algorithm is selected by name (`"nsga2"` | `"ccmo"`) at the inference, training, and benchmark layers.

## Non-goals

- CP-SAT exact baseline (issue #14, queued behind this PR).
- α-fairness knob (issue #16; orthogonal, can land before or after).
- Domain-model expansion to support INRC-I's full richness (skills, contract patterns, weekend rotations, shift-sequencing constraints). Sprint-track lossy mapping only in this PR.
- Bit-for-bit reproducibility against the existing DEAP NSGA-II. Different RNGs and dispatch order make this impossible; we use absolute-threshold convergence tests instead.
- Domain-comparable scoring against the published INRC-I leaderboard. We use INRC-I instances for *internal* A/B between our two algorithms, not for score chasing.

## Decisions made during brainstorming

| Decision | Choice | Reasoning |
|---|---|---|
| Refactor scope | Idiomatic — vectorized operators on `SolutionBatch`, GPU-native everywhere, `_evaluate_batch` override | Single-PR delivery; bigger upside if we'll later scale population size for #2 (CCMO) and #9 (EoH/ReEvo) |
| Algorithm scope | Both — rename to NSGA-II AND add CCMO in this PR | Closes issue #15 in the same branch as the framework swap; benchmark gives apples-to-apples A/B |
| Benchmark | INRC-I sprint track (10 instances), lossy adapter | Smallest viable benchmark; literature-shaped data without a domain-model rewrite |
| Test strategy | Convergence + structural tests added in this PR | First test infra in the repo; convergence with seed=42 + threshold checks; no exact-output reproduction |
| Constraint partitioning in CCMO | 2-D objectives `(imbalance, b2b)`; `violations` as a constraint | Per Tian et al. 2021 §IV — that's CCMO's intended formulation |
| Result schema for CCMO | Best is the lowest-objective-sum feasible solution from Pop1; auxiliary Pop2 front exposed separately | Keeps the `OptimizerResult` base contract simple while preserving full dual-population telemetry |
| Registry pattern | `__init_subclass__` auto-registration on the `Optimizer` ABC (Option B) | Idiomatic Python; collapses registry into the base class; adding a new optimizer = subclass + `name` attr |
| Inference route shape | Single `/predict/evolutionary/{algorithm}` with FastAPI Enum path param | Adding a future optimizer auto-extends the API; deprecated `/predict/ga` shim for one release |

## Architecture overview

```
                        ┌──────────────────────────────────┐
                        │  Optimizer (ABC)                 │
                        │   • __init_subclass__ registers  │
                        │   • create(name, problem)        │
                        │   • list_available()             │
                        └──────────────────────────────────┘
                                  │            │
                  ┌───────────────┘            └───────────────┐
                  ▼                                            ▼
        ┌────────────────────┐                       ┌────────────────────┐
        │ NSGAIIOptimizer    │                       │ CCMOOptimizer      │
        │  name = "nsga2"    │                       │  name = "ccmo"     │
        └────────────────────┘                       └────────────────────┘
                  │                                            │
                  │       Both share:                          │
                  └────────► RosteringProblem(Problem) ◄───────┘
                            DayAlignedCrossOver
                            UniformIntMutation
                            RepairOperator
```

Both optimizers consume the same `RosteringProblem` (an `evotorch.Problem` adapter over our domain `SchedulingProblem`) and the same vectorized operators. They differ only in their loop:

- `NSGAIIOptimizer` calls EvoTorch's stock `GeneticAlgorithm` (which performs Pareto-rank + crowding selection automatically when the Problem has multiple objectives).
- `CCMOOptimizer` runs a manual two-population coevolution loop because EvoTorch's algorithms are single-population by design.

A registry on the `Optimizer` ABC lets every layer above (inference service, FastAPI route, training CLI, benchmark runner) accept the algorithm by name.

## Module layout

```
src/ai/
├── optimizers/
│   ├── __init__.py                  exports Optimizer, NSGAIIOptimizer, CCMOOptimizer
│   │                                eagerly imports concrete classes to trigger registration
│   ├── base.py                      Optimizer ABC + __init_subclass__ registry + factory
│   ├── result.py                    OptimizerConfig/Result base + NSGAII/CCMO subclasses
│   ├── rostering_problem.py         RosteringProblem(evotorch.Problem)
│   ├── operators.py                 DayAlignedCrossOver, UniformIntMutation, RepairOperator
│   ├── nsga2.py                     NSGAIIOptimizer
│   └── ccmo.py                      CCMOOptimizer + Pareto-rank/crowding helpers
│
├── benchmarks/
│   ├── __init__.py
│   ├── inrc1/
│   │   ├── __init__.py
│   │   ├── parser.py                INRC-I .txt parser → InrcInstance intermediate form
│   │   ├── loader.py                load_instance(name) → SchedulingProblem (lossy)
│   │   └── manifest.json            sprint01..sprint10 names + canonical metadata
│   └── runner.py                    run_benchmark(algorithms, instances, seeds) → BenchmarkReport
│
├── services/
│   └── optimizer_inference.py       run_optimizer_inference(name, request, ...)
│                                    (replaces ga_inference.py; ccmo never gets its own file)
│
├── training/
│   ├── evolutionary.py              CLI: --algorithm {nsga2,ccmo}
│   │                                (replaces training/ga.py; ccmo never gets its own file)
│   └── benchmark.py                 CLI: --algorithm nsga2,ccmo --track sprint
│
├── api/
│   └── inference.py                 EDIT — /predict/evolutionary/{algorithm}; deprecate /predict/ga
│
├── domain/
│   └── schemas.py                   EDIT — replace GA* schemas with NSGA-II/CCMO/Benchmark variants;
│                                    keep one-release deprecation aliases for old names
│
└── main.py                          UNCHANGED

src/frontend/api_client/client.py    EDIT — call /predict/evolutionary/nsga2 instead of /predict/ga

scripts/
└── fetch_inrc1.py                   Downloads INRC-I instances on first use

data/benchmarks/inrc1/               GITIGNORED — fetched instances land here

tests/
├── conftest.py                      shared fixtures (problem instances, seeds)
├── fixtures/inrc1/sprint01.txt      ~5 KB single instance bundled for parser tests
└── ai/
    ├── benchmarks/inrc1/
    │   ├── test_parser.py
    │   └── test_loader.py
    ├── optimizers/
    │   ├── test_rostering_problem.py
    │   ├── test_operators.py
    │   ├── test_nsga2.py
    │   ├── test_ccmo.py
    │   └── test_registry.py
    └── training/
        └── test_benchmark_smoke.py

pyproject.toml                       EDIT — drop deap; add evotorch>=0.5.1, pymoo, scipy;
                                     new [dev] group with pytest, pytest-cov, pytest-mock
```

The Pydantic domain `SchedulingProblem` in `domain/problem.py` is **untouched**. RL, future CP-SAT, and existing services keep reading the framework-agnostic domain model.

## Optimizer base class and registry

```python
# src/ai/optimizers/base.py
from abc import ABC, abstractmethod
from typing import ClassVar

from ai.domain.problem import SchedulingProblem
from ai.optimizers.result import OptimizerConfig, OptimizerResult


class Optimizer(ABC):
    """Abstract base for all evolutionary optimizers.

    Concrete subclasses are auto-registered when their class body executes,
    via __init_subclass__. They must declare a 'name' class attribute.

    Adding a new optimizer:
      1. Subclass Optimizer
      2. Set `name`, `config_class`, `result_class` class attributes
      3. Implement run()
      4. Import the module from `src/ai/optimizers/__init__.py` so the class is loaded
    """

    name: ClassVar[str] = ""
    config_class: ClassVar[type[OptimizerConfig]]
    result_class: ClassVar[type[OptimizerResult]]

    _registry: ClassVar[dict[str, type["Optimizer"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return                                              # abstract intermediate class, skip
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

```python
# src/ai/optimizers/__init__.py
from ai.optimizers.base import Optimizer
from ai.optimizers.ccmo import CCMOOptimizer        # noqa: F401 — import for registration
from ai.optimizers.nsga2 import NSGAIIOptimizer     # noqa: F401 — import for registration

__all__ = ["Optimizer", "NSGAIIOptimizer", "CCMOOptimizer"]
```

Usage everywhere downstream:

```python
optimizer = Optimizer.create("nsga2", problem)
choices  = Optimizer.list_available()                # ["ccmo", "nsga2"]
```

## RosteringProblem

```python
# src/ai/optimizers/rostering_problem.py
import torch
from evotorch import Problem, SolutionBatch

from ai.domain.problem import SchedulingProblem


class RosteringProblem(Problem):
    """EvoTorch Problem adapter for shift scheduling.

    Decision space: integer vector of length num_shifts, each gene in
    [0, num_employees-1] = which employee is assigned to that shift.

    Objectives (all minimized):
      0: imbalance       = 1 - Jain's fairness index
      1: violations      = max-hours overrun + 10 * unavailability hits
      2: back_to_back    = count of consecutive same-employee shifts
    """

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
        self._shift_lens = torch.tensor(sp.shift_lengths, dtype=torch.float64, device=d)
        self._shift_types = (torch.arange(sp.num_shifts, device=d) % sp.shifts_per_day)
        self._shift_hour_values = self._shift_lens[self._shift_types]   # (num_shifts,)
        self._max_hours_t = torch.tensor(sp.max_hours, dtype=torch.float64, device=d)

        unavail_mask = torch.zeros(sp.days, sp.num_employees, dtype=torch.bool, device=d)
        for day, emp in sp.unavailability:
            unavail_mask[day, emp] = True
        self._unavail_mask = unavail_mask                                # (days, num_employees)

    def _evaluate_batch(self, solutions: SolutionBatch) -> None:
        pop = solutions.values.to(torch.long)                            # (N, num_shifts)
        n = pop.shape[0]
        sp = self._sp

        lens = self._shift_hour_values.unsqueeze(0).expand(n, -1)
        hours = torch.zeros(n, sp.num_employees, dtype=torch.float64, device=pop.device)
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
        day_per_shift = torch.arange(sp.num_shifts, device=pop.device) // sp.shifts_per_day
        violations_per_cell = self._unavail_mask[day_per_shift.unsqueeze(0).expand(n, -1), pop]
        unavail_count = violations_per_cell.sum(dim=1).to(torch.float64)
        violations = exceed + unavail_count * 10.0

        b2b = (pop[:, :-1] == pop[:, 1:]).sum(dim=1).to(torch.float64)

        fitnesses = torch.stack([imbalance, violations, b2b], dim=1)
        solutions.set_evals(fitnesses.to(torch.float32))
```

The unavailability handling that was a Python-level loop in `optimizers/ga.py:108` becomes a single gather over a precomputed boolean mask. Fully vectorized; CUDA-friendly.

## Operators

All three live in `src/ai/optimizers/operators.py`. Each is a `CopyingOperator` subclass operating on the full `SolutionBatch` in pure tensor ops.

```python
class DayAlignedCrossOver(CopyingOperator):
    """Two-parent recombination cut at a day boundary."""

    def __init__(self, problem, *, tournament_size=4, cross_over_rate=0.7):
        super().__init__(problem)
        self._tournament_size = tournament_size
        self._cross_over_rate = cross_over_rate
        self._shifts_per_day = problem._sp.shifts_per_day
        self._num_shifts = problem._sp.num_shifts

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        parents = self._do_tournament(
            batch,
            num_tournaments=batch.solution_length,
            tournament_size=self._tournament_size,
        )
        n_pairs = parents.shape[0] // 2
        p = parents[: n_pairs * 2].view(n_pairs, 2, -1)

        num_days = self._num_shifts // self._shifts_per_day
        day_cuts = torch.randint(1, num_days, (n_pairs,), device=p.device)
        shift_cuts = day_cuts * self._shifts_per_day

        apply_mask = torch.rand(n_pairs, device=p.device) < self._cross_over_rate
        children = p.clone()
        idx = torch.arange(self._num_shifts, device=p.device).unsqueeze(0).unsqueeze(0)
        right = idx >= shift_cuts.view(n_pairs, 1, 1)
        do_swap = apply_mask.view(n_pairs, 1, 1) & right
        swapped = children.flip(dim=1)
        children = torch.where(do_swap, swapped, children)
        return self._make_children_batch(children.view(-1, self._num_shifts))


class UniformIntMutation(CopyingOperator):
    """Per-gene uniform integer mutation with both per-individual and per-gene gates."""

    def __init__(self, problem, *, indpb=0.05, mut_rate=0.2):
        super().__init__(problem)
        self._indpb = indpb
        self._mut_rate = mut_rate
        self._num_employees = problem._sp.num_employees

    def _do(self, batch: SolutionBatch) -> SolutionBatch:
        result = batch.clone()
        values = result.values
        n = values.shape[0]
        ind_mask = (torch.rand(n, device=values.device) < self._mut_rate).unsqueeze(1)
        gene_mask = torch.rand_like(values, dtype=torch.float32) < self._indpb
        new_values = torch.randint(
            0, self._num_employees, values.shape, device=values.device, dtype=values.dtype
        )
        flip = ind_mask & gene_mask
        values[flip] = new_values[flip]
        return result


class RepairOperator(CopyingOperator):
    """Replace any cell where the assigned employee is unavailable with a random valid one."""

    def __init__(self, problem):
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
        is_unavail = self._unavail_mask[day_per_shift.unsqueeze(0).expand(n, -1), values]
        if not is_unavail.any():
            return result

        availability = ~self._unavail_mask[day_per_shift]                   # (T, num_employees)
        rand = torch.rand(n, T, self._num_employees, device=d)
        rand[:, ~availability] = -1.0
        replacement = rand.argmax(dim=-1)
        values[is_unavail] = replacement[is_unavail]
        return result
```

**Pipeline order** passed to algorithms: crossover → mutation → repair. Repair is last so any unavailability hit introduced by mutation is fixed before fitness evaluation.

**CCMO compatibility.** Operators don't know whether they're acting on Pop1 or Pop2 — they just transform a `SolutionBatch`. CCMO-specific logic lives in `CCMOOptimizer`'s loop; the operators are reused as-is.

**Memory note.** `RepairOperator` allocates `(N, T, num_employees)` floats. On the default 7×30×3 instance with N=100, that's 63K floats. On INRC-I sprint instances with N=100, T≈100, employees≈20, that's 200K floats. Trivial. If a future PR scales to larger instances and OOMs, switch to a sparse formulation.

## NSGAIIOptimizer

```python
# src/ai/optimizers/nsga2.py
import torch
from evotorch.algorithms import GeneticAlgorithm
from typing import ClassVar

from ai.optimizers.base import Optimizer
from ai.optimizers.operators import DayAlignedCrossOver, RepairOperator, UniformIntMutation
from ai.optimizers.result import GAStepStatus, NSGAIIConfig, NSGAIIResult, OptimizerConfig
from ai.optimizers.rostering_problem import RosteringProblem


class NSGAIIOptimizer(Optimizer):
    """Multi-objective NSGA-II on top of EvoTorch's GeneticAlgorithm.

    Selection (Pareto-rank + crowding distance) is automatic because the
    underlying RosteringProblem declares 3 minimized objectives.
    """

    name: ClassVar[str] = "nsga2"
    config_class: ClassVar[type[OptimizerConfig]] = NSGAIIConfig
    result_class: ClassVar[type] = NSGAIIResult

    def run(self, config: NSGAIIConfig | None = None, verbose: bool = False) -> NSGAIIResult:
        config = config or NSGAIIConfig()
        if config.seed is not None:
            torch.manual_seed(config.seed)

        problem = RosteringProblem(self._sp, device=config.device)
        operators = [
            DayAlignedCrossOver(problem, tournament_size=config.tournament_size, cross_over_rate=config.cxpb),
            UniformIntMutation(problem, indpb=config.indpb, mut_rate=config.mutpb),
            RepairOperator(problem),
        ]
        searcher = GeneticAlgorithm(problem, operators=operators, popsize=config.pop_size, elitist=config.elitist)

        history: list[GAStepStatus] = []
        for gen in range(config.generations):
            searcher.step()
            history.append(self._snapshot(gen, searcher))
            if verbose and gen % 10 == 0:
                self._print_snapshot(history[-1])

        return self._collect_result(searcher, history)

    # ... _snapshot, _collect_result, _first_pareto_front_indices helpers
```

The DEAP-specific creator/toolbox setup, the manual generation loop with clone/select/mate/mutate, the `tools.ParetoFront` archive, and the `tools.Logbook` are all gone. EvoTorch's `searcher.step()` runs the operator pipeline; `searcher.population.compute_pareto_ranks()` extracts the front.

## CCMOOptimizer

CCMO doesn't fit `GeneticAlgorithm` (single population). Manual two-population loop using the same Problem and operators.

```python
# src/ai/optimizers/ccmo.py
class CCMOOptimizer(Optimizer):
    """Constrained MOEA via Coevolution (Tian et al. IEEE TEC 2021).

    Two coevolving populations:
      Pop1 — constraint-aware (Deb's constraint-domination principle)
      Pop2 — unconstrained (NSGA-II on (imbalance, b2b) only)
    Both populations' offspring feed both selections each generation.
    """

    name: ClassVar[str] = "ccmo"
    config_class: ClassVar[type[OptimizerConfig]] = CCMOConfig
    result_class: ClassVar[type] = CCMOResult

    def run(self, config: CCMOConfig | None = None, verbose: bool = False) -> CCMOResult:
        config = config or CCMOConfig()
        if config.seed is not None:
            torch.manual_seed(config.seed)

        problem = RosteringProblem(self._sp, device=config.device)
        operators = self._build_operators(problem, config)

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

            pop1 = self._constraint_aware_select(SolutionBatch.cat([pop1, offspring]), config.pop_size)
            pop2 = self._unconstrained_select(SolutionBatch.cat([pop2, offspring]), config.pop_size)

            history.append(self._snapshot(gen, pop1, pop2))
            if verbose and gen % 10 == 0:
                self._print_snapshot(history[-1])

        return self._collect_result(pop1, pop2, history)

    @staticmethod
    def _constraint_aware_select(pool: SolutionBatch, pop_size: int) -> SolutionBatch:
        """Deb's constraint-domination on (imbalance, b2b) with violations as constraint."""
        evals = pool.evals                                   # (M, 3)
        feasible_mask = evals[:, 1] <= 0.0
        obj = torch.stack([evals[:, 0], evals[:, 2]], dim=1)

        ranks = torch.empty(evals.shape[0], dtype=torch.long, device=evals.device)
        feas_idx = torch.where(feasible_mask)[0]
        inf_idx = torch.where(~feasible_mask)[0]

        if feas_idx.numel() > 0:
            ranks[feas_idx] = _nsga2_pareto_ranks(obj[feas_idx])
        if inf_idx.numel() > 0:
            v_ranks = evals[inf_idx, 1].argsort().argsort()
            ranks[inf_idx] = v_ranks + (feas_idx.numel() if feas_idx.numel() > 0 else 0) + 1

        return _select_by_rank_and_crowding(pool, ranks, obj, pop_size)

    @staticmethod
    def _unconstrained_select(pool: SolutionBatch, pop_size: int) -> SolutionBatch:
        """Pure NSGA-II on (imbalance, b2b). Violations ignored."""
        evals = pool.evals
        obj = torch.stack([evals[:, 0], evals[:, 2]], dim=1)
        ranks = _nsga2_pareto_ranks(obj)
        return _select_by_rank_and_crowding(pool, ranks, obj, pop_size)
```

`_nsga2_pareto_ranks` is a vectorized fast non-dominated sort. `_select_by_rank_and_crowding` truncates to `pop_size` by `(rank, -crowding_distance)`. Both are pure-torch helpers in `optimizers/ccmo.py`.

**Best-solution selection.** `result.best_schedule` = lowest `sum(imbalance, b2b)` member of Pop1's feasibles. If Pop1 has no feasible at termination, `fell_back_to_auxiliary=True` and we surface that via the inference path so callers know not to ship the schedule.

## Schemas

### `src/ai/optimizers/result.py` (algorithm-internal)

```python
class OptimizerConfig(BaseModel):
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
    two coevolving populations of this size each, so the total memory and
    per-generation evaluation count are 2 × pop_size.
    """
    pass


class GAStepStatus(BaseModel):
    generation: int
    mean_obj0_imbalance: float
    mean_obj1_violations: float
    mean_obj2_b2b: float
    pareto_front_size: int


class CCMOStepStatus(BaseModel):
    generation: int
    pop1_feasible_count: int
    pop1_best_imbalance: float
    pop1_best_b2b: float
    pop1_pareto_size: int
    pop2_pareto_size: int
    pop2_mean_violations: float


class OptimizerResult(BaseModel):
    best_schedule: list[int]
    best_fitness: tuple[float, float, float]
    pareto_front: list[list[int]]
    pareto_fitnesses: list[tuple[float, float, float]]


class NSGAIIResult(OptimizerResult):
    step_history: list[GAStepStatus]


class CCMOResult(OptimizerResult):
    feasible_pareto_front: list[list[int]]
    feasible_pareto_fitnesses: list[tuple[float, float, float]]
    auxiliary_pareto_front: list[list[int]]
    auxiliary_pareto_fitnesses: list[tuple[float, float, float]]
    step_history: list[CCMOStepStatus]
    fell_back_to_auxiliary: bool = False
```

`pareto_front` / `pareto_fitnesses` on `CCMOResult` mirror the *feasible* Pop1 front (what most consumers want); the auxiliary fields preserve dual-population telemetry for research scripts.

### `src/ai/domain/schemas.py` (training output + benchmark)

`GAFitnessResult` / `GAConfigSnapshot` / `GATrainResult` are renamed to `NSGAIIFitnessResult` / `NSGAIIConfigSnapshot` / `NSGAIITrainResult`. New parallels `CCMOFitnessResult` / `CCMOConfigSnapshot` / `CCMOTrainResult`. New `BenchmarkRunRecord` / `BenchmarkAggregate` / `BenchmarkReport`. One-release deprecation aliases for the old names.

Full schemas already specified in brainstorming Section 7 — implement verbatim from there.

## INRC-I benchmark

### Scope

- Sprint track only (`sprint01..sprint10`). Medium and long deferred until the domain model supports skills/contracts.
- Lossy adapter: number of nurses, days, distinct shift types, per-shift duration, per-nurse `MaximumNumberOfAssignments`, day-off requests.
- Drops: skills/grades, contract patterns, weekend rotation, shift sequencing, soft-constraint weights.

### Files

```python
# src/ai/benchmarks/inrc1/parser.py
def parse_inrc1_instance(text: str) -> InrcInstance:
    """Parse a single INRC-I .txt into a structured intermediate (full INRC-I shape)."""
    ...

# src/ai/benchmarks/inrc1/loader.py
def load_instance(name: str) -> SchedulingProblem:
    """Lossy. Raises FileNotFoundError pointing to scripts/fetch_inrc1.py if missing."""
    ...

def list_instances(track: str = "sprint") -> list[str]:
    ...

# src/ai/benchmarks/runner.py
def run_benchmark(
    algorithms: list[str],
    instance_names: list[str],
    seeds: list[int],
    config_overrides: dict | None = None,
    reference_point: tuple[float, float, float] = (1.0, 1000.0, 100.0),
) -> BenchmarkReport:
    records: list[BenchmarkRunRecord] = []
    for algo in algorithms:
        for instance_name in instance_names:
            problem = load_instance(instance_name)
            for seed in seeds:
                optimizer = Optimizer.create(algo, problem)
                config = optimizer.config_class(seed=seed, **(config_overrides or {}))
                result = optimizer.run(config)
                hv = compute_hypervolume(result.pareto_fitnesses, reference_point)
                records.append(BenchmarkRunRecord(...))
    return aggregate(records)
```

### Hypervolume + statistics

- HV: `pymoo.indicators.hv.HV` with reference point `(1.0, 1000.0, 100.0)` — dominates all plausible objective tuples on sprint instances.
- HV is computed only on the **feasible** front (CCMO Pop1 rank-0; NSGA-II rank-0 filtered to `violations == 0`). Prevents reward-hacking via large infeasible fronts.
- Pairwise stats: Wilcoxon signed-rank p-value via `scipy.stats.wilcoxon` (Demšar 2006).

### Data acquisition

`scripts/fetch_inrc1.py` downloads the public corpus on first use. Multiple candidate mirror URLs tried in order. `data/benchmarks/inrc1/` is gitignored.

### Documented caveats (in `BENCHMARKS.md` and the runner output)

> The hypervolume scores reported here are **not** the INRC-I aggregate competition scores. We map INRC-I instances to our simpler `SchedulingProblem` (single skill class, no contract patterns, no shift sequencing) and optimize three objectives (`imbalance`, `violations`, `b2b`). Comparison is **NSGA-II vs CCMO on identical inputs**, not against the published leaderboard.

## Service, API, training entrypoints

### Single inference service

```python
# src/ai/services/optimizer_inference.py
def run_optimizer_inference(
    algorithm: str,
    request: SchedulingRequest,
    generations: int = 100,
    pop_size: int = 50,
    device: str = "cpu",
) -> SchedulingResponse:
    problem = SchedulingProblem.from_request(request)
    optimizer = Optimizer.create(algorithm, problem)

    config = optimizer.config_class(
        generations=generations, pop_size=pop_size, device=device,
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

### FastAPI route, single endpoint with Enum path param

```python
# src/ai/api/inference.py
EvolutionaryAlgorithm = Enum(
    "EvolutionaryAlgorithm",
    {n.upper(): n for n in Optimizer.list_available()},
)


@router.post("/evolutionary/{algorithm}", response_model=SchedulingResponse)
async def predict_evolutionary(
    algorithm: EvolutionaryAlgorithm,
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
    device: str = Query("cpu", pattern=r"^(cpu|cuda)$"),
) -> SchedulingResponse:
    return run_optimizer_inference(
        algorithm.value, request, generations=generations, pop_size=pop_size, device=device
    )


@router.post("/ga", response_model=SchedulingResponse, deprecated=True)
async def predict_ga(request: SchedulingRequest, generations: int = Query(100), pop_size: int = Query(50)):
    """DEPRECATED: use /predict/evolutionary/nsga2."""
    return run_optimizer_inference("nsga2", request, generations=generations, pop_size=pop_size)
```

### Training CLIs

```bash
# Single-algorithm training
python -m ai.training.evolutionary --algorithm nsga2 --generations 200 --pop-size 100
python -m ai.training.evolutionary --algorithm ccmo  --generations 200 --pop-size 100 --device cuda

# Benchmark A/B
python -m ai.training.benchmark --algorithm nsga2,ccmo --track sprint --seeds 10 --report a_b.json
```

Output filenames in `checkpoints/`:
- `<algorithm>_best_schedule.json` — `<Algorithm>TrainResult`
- `<algorithm>_step_history.json` — `list[<Algorithm>StepStatus]` (replaces the old `ga_logbook.pkl`)

### Frontend caller

`src/frontend/api_client/client.py` updates the GA route call from `/predict/ga` to `/predict/evolutionary/nsga2`. One-line change in the same PR so we don't ship a frontend that pings a deprecated route.

## Testing strategy

```
tests/
├── conftest.py
├── fixtures/inrc1/sprint01.txt
└── ai/
    ├── benchmarks/inrc1/{test_parser.py,test_loader.py}
    ├── optimizers/
    │   ├── test_rostering_problem.py
    │   ├── test_operators.py
    │   ├── test_nsga2.py
    │   ├── test_ccmo.py
    │   └── test_registry.py
    └── training/test_benchmark_smoke.py
```

| File | Tests | Role |
|---|---|---|
| `test_rostering_problem.py` | `test_evaluate_batch_shape`, `test_imbalance_matches_jain`, `test_violations_count_correct`, `test_b2b_count_correct`, `test_gpu_path_consistent_with_cpu` | Numerical parity with current GA `batch_fitness` on a fixed population |
| `test_operators.py` | `test_crossover_preserves_dimensions`, `test_crossover_only_at_day_boundaries`, `test_mutation_respects_indpb_distribution`, `test_repair_eliminates_unavailability_violations`, `test_repair_idempotent` | Per-operator unit tests on small fixed batches |
| `test_nsga2.py` | **Convergence** (seed=42, 50 generations, 100 popsize on default 7×30×3): `best_fitness[0] < 0.05`, `best_fitness[1] == 0`, `best_fitness[2] < 30`. **Structural**: `test_result_shape_correct`, `test_pareto_front_non_empty`, `test_no_nan_fitnesses`, `test_fitness_improves_over_generations` | Issue AC + regression guards |
| `test_ccmo.py` | **Convergence**: `test_default_instance_converges_to_feasible`. **Coevolution**: `test_pop2_explores_infeasible`. **A/B sanity**: `test_ccmo_hv_at_least_competitive_with_nsga2` (CCMO HV ≥ 90% of NSGA-II HV at matched seed). **Fallback**: `test_fall_back_when_no_feasible` | CCMO correctness + dual-population invariant |
| `test_registry.py` | `test_init_subclass_registers`, `test_create_unknown_raises`, `test_list_available_returns_sorted_names`, `test_duplicate_name_raises` | Registry invariants |
| `test_parser.py` | `test_parses_sprint01_fixture` against published Haspeslagh metadata | Lossless parse to intermediate form |
| `test_loader.py` | `test_loads_as_scheduling_problem`, `test_drops_documented_fields_silently` | Lossy adapter correctness |
| `test_benchmark_smoke.py` | `test_run_nsga2_on_sprint01_one_seed_smoke` (5 generations, 1 seed, 1 instance, <30s) | E2E wiring |

**Markers.** `@pytest.mark.slow` on convergence tests (50+ generations); `@pytest.mark.benchmark` on the benchmark-runner test. Default `pytest` runs neither; CI nightly adds `-m "slow or benchmark"`.

**Determinism contract.** All convergence tests pass `seed=42`. Bit-for-bit reproduction not promised across hardware/library updates; same hardware should yield same result.

**pyproject `[tool.pytest.ini_options]`:**
```toml
testpaths = ["tests"]
addopts = "--cov=ai --cov-report=term-missing --cov-fail-under=80"
markers = [
    "slow: longer-running tests (>10s)",
    "benchmark: tests that touch the INRC-I benchmark runner",
]
```

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
    "evotorch>=0.5.1",         # NEW
    # "deap>=1.4.1",           # REMOVED
    "tensorboard>=2.17.0",
    "sqlalchemy>=2.0.48",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",
]

benchmarks = [                  # NEW extra
    "pymoo>=0.6.1",            # hypervolume
    "scipy>=1.13.0",           # Wilcoxon signed-rank
]

dev = [                         # NEW group
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
]
```

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| EvoTorch's `GeneticAlgorithm` may not handle integer multi-obj edge cases cleanly (their docs are sparser there than for continuous problems) | Test plan catches fundamental mismatches early via convergence + structural tests. Fallback: thin `EvoTorchGA` subclass adjusting whatever piece doesn't fit. |
| `_do_tournament` helper inside `evotorch.operators.base.CrossOver` may not exist with that exact name | Verify against actual source during implementation; if private, write tournament select inline (~10 lines of vectorized argsort). |
| CCMO selection helpers are fresh code; bugs in non-dominated sort are easy to write | Add `test_pareto_ranks_against_brute_force` comparing the vectorized fast non-dominated sort against a naive O(N²) brute force on small batches. |
| INRC-I instance URL may have moved (the official site has migrated several times) | `scripts/fetch_inrc1.py` tries multiple mirrors in order; if all fail, prints citation and instructions for manual placement. |
| INRC-I lossy mapping may misrepresent some sprint instances | Benchmark report explicitly states "not literature-comparable"; we use INRC-I shapes for *internal* A/B, not score chasing. If lossy mapping makes an instance trivial, the convergence-immediate note shows up in the report. |
| Behavior drift between current DEAP NSGA-II and new EvoTorch NSGA-II — same problem, different RNG and dispatch order | Convergence tests use absolute thresholds, not bit-for-bit equality. Benchmark A/B (NSGA-II vs CCMO) is the real measurement of relative quality. |
| `evotorch>=0.5.1` is younger than DEAP; API stability less battle-tested | Pin minimum version exactly; document the chosen version. If a future EvoTorch release breaks our subclasses, blast radius is bounded to `operators.py` + `nsga2.py` + `ccmo.py` + `rostering_problem.py`. |
| `/predict/ga` deprecated shim could rot | Shim is one line; CI test asserts the deprecated route still returns a `SchedulingResponse`. Remove in a single-commit follow-up after one release cycle. |

## Implementation sequencing (single-PR commit history)

The PR is large. Sequence commits so reviewers can read it in chunks AND each commit leaves the tree in a working state. The trick is keeping DEAP installed alongside EvoTorch through commits 1–4, then removing DEAP in the same commit that retires the last DEAP caller.

1. **`build: pyproject.toml — add EvoTorch, add [dev] and [benchmarks] groups`** — additive only. DEAP still present. Tree functional.
2. **`feat: optimizers/base.py + result.py — Optimizer ABC with __init_subclass__ auto-registration`** — pure addition; registry tests. Tree functional (the existing `GAOptimizer` is unchanged and unaware of the ABC).
3. **`feat: optimizers/rostering_problem.py — EvoTorch Problem subclass`** — pure addition; `_evaluate_batch` parity tests against the existing GA's `batch_fitness`.
4. **`feat: optimizers/operators.py — vectorized day-aligned crossover, mutation, repair`** — pure addition; per-operator tests.
5. **`refactor: replace DEAP GAOptimizer with EvoTorch NSGAIIOptimizer; drop deap`** — single commit that does **all** of: deletes old `optimizers/ga.py`; adds `optimizers/nsga2.py` with `NSGAIIOptimizer`; replaces `services/ga_inference.py` with `services/optimizer_inference.py`; renames `training/ga.py` to `training/evolutionary.py`; updates `api/inference.py` to add `/predict/evolutionary/{algorithm}` and the `/predict/ga` deprecated shim; updates `src/frontend/api_client/client.py` to call the new route; removes `deap` from `pyproject.toml`; adds Pydantic schema renames in `domain/schemas.py` with one-release deprecation aliases; convergence tests added. Tree functional after this commit.
6. **`feat: optimizers/ccmo.py — Constrained MOEA via Coevolution`** — new optimizer + Pareto-rank/crowding helpers. Auto-registers via `__init_subclass__`. Convergence + A/B tests. The route Enum auto-extends to include `ccmo` thanks to `Optimizer.list_available()`.
7. **`feat: benchmarks/inrc1/{parser,loader}.py — INRC-I sprint-track adapter`** — parser, loader, manifest, single bundled fixture for tests, `scripts/fetch_inrc1.py`. Adds `data/benchmarks/inrc1/` to `.gitignore`. Tests against the fixture only (no network).
8. **`feat: benchmarks/runner.py + training/benchmark.py — A/B harness with hypervolume + Wilcoxon`** — runner, CLI, smoke test (5 generations × 1 seed × 1 instance, <30s).
9. **`docs: README, BENCHMARKS.md, wiki pointers`** — describe new endpoints, the deprecation, the lossy INRC-I mapping caveat, and how to invoke the benchmark CLI.

## Migration / breaking changes summary (for PR description and wiki)

- **Renamed (breaking for direct importers, not for HTTP clients):**
  - `GAOptimizer` → `NSGAIIOptimizer`
  - `GAConfig` → `NSGAIIConfig`
  - `GAResult` → `NSGAIIResult` (`logbook` field replaced with `step_history: list[GAStepStatus]`)
  - `services/ga_inference.py` → `services/optimizer_inference.py`
  - `training/ga.py` → `training/evolutionary.py` (with `--algorithm nsga2`)
  - `GAFitnessResult` → `NSGAIIFitnessResult`, `GAConfigSnapshot` → `NSGAIIConfigSnapshot`, `GATrainResult` → `NSGAIITrainResult` (one-release deprecation aliases)
- **Deprecated (still works):** `POST /predict/ga` → use `POST /predict/evolutionary/nsga2`.
- **New:** `POST /predict/evolutionary/ccmo`; `python -m ai.training.evolutionary --algorithm ccmo`; `python -m ai.training.benchmark`.
- **Removed (no replacement):** DEAP dependency; `tools.Logbook` pickle output (replaced by JSON `step_history`).

## Resume checklist (handover to writing-plans)

The writing-plans skill builds the implementation plan from this spec. The plan should:

- Convert the 9-commit sequencing above into ordered tasks with explicit deliverables.
- Materialize the test plan into ordered test-first or test-after directives per file.
- Include verification commands per task (`uv run pytest tests/ai/optimizers/test_rostering_problem.py`, etc.).
- Flag points where implementation needs to verify against EvoTorch source rather than guessing (specifically the `_do_tournament` helper, see Risks).
- Plan for the frontend caller update (`src/frontend/api_client/client.py`) within the same PR.
- Include the `data/benchmarks/inrc1/` gitignore entry.

After the PR merges, return to `docs/superpowers/specs/2026-05-10-cpsat-baseline-design.md` (CP-SAT, queued).
