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

from ai.domain.schemas import CPSATStageResult

__all__ = [
    "CPSATStageResult",   # re-exported for callers that imported from here historically
    "OptimizerConfig",
    "EvolutionaryConfig",
    "NSGAIIConfig",
    "CCMOConfig",
    "CPSATConfig",
    "GAStepStatus",
    "CCMOStepStatus",
    "OptimizerResult",
    "MultiObjectiveResult",
    "NSGAIIResult",
    "CCMOResult",
    "CPSATResult",
]


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
    fairness_alpha: float = 2.0


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
    ["b2b", "fairness"],
    ["fairness", "b2b"],
)


class CPSATConfig(OptimizerConfig):
    """CP-SAT exact-baseline hyperparameters."""

    timeout_s_per_stage: float = 30.0
    num_workers: int = 8
    objective_priority: list[str] = ["b2b", "fairness"]
    fairness_alpha: float = float("inf")

    @field_validator("objective_priority")
    @classmethod
    def _validate_priority(cls, v: list[str]) -> list[str]:
        if v not in _VALID_OBJECTIVE_PRIORITIES:
            raise ValueError(
                f"Unsupported objective_priority {v}; "
                "only ['b2b','fairness'] or ['fairness','b2b'] are valid."
            )
        return v

    @field_validator("fairness_alpha")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        if v != float("inf"):
            raise ValueError(
                f"CPSAT only supports egalitarian fairness (alpha=inf); got {v}. "
                "Use NSGA-II or CCMO for finite alpha values."
            )
        return v


# === Step-status types ===


class GAStepStatus(BaseModel):
    """Per-generation snapshot for the NSGA-II loop."""

    generation: int
    mean_obj0_unfairness: float
    mean_obj1_violations: float
    mean_obj2_b2b: float
    pareto_front_size: int


class CCMOStepStatus(BaseModel):
    """Per-generation snapshot for CCMO. Tracks both populations."""

    generation: int
    pop1_feasible_count: int
    pop1_best_unfairness: float
    pop1_best_b2b: float
    pop1_pareto_size: int
    pop2_pareto_size: int
    pop2_mean_violations: float


# CPSATStageResult lives in ai.domain.schemas (re-exported above) so the
# checkpoint schema CPSATTrainResult can reference it without an import cycle.


# === Result hierarchy ===


class OptimizerResult(BaseModel):
    """Universal result every optimizer must return.

    `best_fitness` is the same 3-tuple `(unfairness, violations, b2b)` for
    every family, so the inference layer and any future benchmark runner
    can index it uniformly. CP-SAT reports `(1 - jain_index, 0.0, b2b_count)`.
    At α=2.0 (the default) unfairness is bit-identical to the legacy
    `1 - jain` formulation.
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
    fairness_gap: int            # h_max - h_min, the CP-SAT optimization variable
    fairness_metric: float       # α-fairness welfare (at α=∞, equals min(hours))
    fairness_alpha: float        # always float('inf') for CPSAT
    jain_index: float            # side metric, always at α=2 for legacy comparability
    stages: list[CPSATStageResult]
    total_wall_clock_s: float
