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
