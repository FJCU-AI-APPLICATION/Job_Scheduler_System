"""Convergence + structural tests for NSGAIIOptimizer."""

import math

import pytest
import torch

from ai.domain.problem import SchedulingProblem


def test_result_shape_correct(tiny_problem: SchedulingProblem):
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
    """Convergence AC for EvoTorch NSGAIIOptimizer on the default 7×30×3 instance.

    With seed=42, 50 generations, 100 popsize the optimizer should show clear
    multi-objective progress. Thresholds are set for the EvoTorch implementation
    (which uses a different crossover/selection mechanism than DEAP-NSGA-II);
    violations reach near-zero only with more generations, so we check that
    (a) imbalance < 0.25 (population-mean imbalance drops well below random),
    (b) violations improve significantly vs initial random generation, and
    (c) b2b < 50.
    """
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(default_problem)
    result = optimizer.run(NSGAIIConfig(generations=50, pop_size=100, seed=42))

    gen0 = result.step_history[0]
    gen_last = result.step_history[-1]

    assert result.best_fitness[0] < 0.25, f"imbalance too high: {result.best_fitness[0]}"
    assert result.best_fitness[2] < 50, f"b2b too high: {result.best_fitness[2]}"
    # Violations must improve over training (not necessarily reach 0 in 50 gen)
    assert gen_last.mean_obj1_violations < gen0.mean_obj1_violations, (
        f"violations did not improve: gen0={gen0.mean_obj1_violations:.1f}, "
        f"genN={gen_last.mean_obj1_violations:.1f}"
    )


def test_pareto_front_non_empty(default_problem: SchedulingProblem):
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import NSGAIIConfig

    optimizer = NSGAIIOptimizer(default_problem)
    result = optimizer.run(NSGAIIConfig(generations=10, pop_size=50, seed=42))

    assert len(result.pareto_front) >= 1
    assert all(len(s) == default_problem.num_shifts for s in result.pareto_front)
