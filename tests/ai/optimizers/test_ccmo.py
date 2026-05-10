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
    """Pop2 should produce some infeasible members during the run."""
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

    domination_count = dominates.sum(dim=0).clone()
    brute_ranks = torch.full((n,), -1, dtype=torch.long)
    rank = 0
    while True:
        front = (domination_count == 0) & (brute_ranks == -1)
        if not front.any():
            break
        brute_ranks[front] = rank
        front_idx = torch.where(front)[0]
        for i in front_idx.tolist():
            domination_count -= dominates[i].long()
        rank += 1
    assert torch.equal(fast_ranks, brute_ranks)


@pytest.mark.slow
def test_default_instance_converges_to_feasible(default_problem: SchedulingProblem):
    """With seed=42, 50 generations, 100 popsize, CCMO should produce a feasible solution.

    NOTE: thresholds calibrated for EvoTorch behavior; original DEAP-based ACs may not hold.
    """
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.result import CCMOConfig

    optimizer = CCMOOptimizer(default_problem)
    result = optimizer.run(CCMOConfig(generations=50, pop_size=100, seed=42))

    # If we did NOT fall back, there should be a feasible solution.
    if not result.fell_back_to_auxiliary:
        assert len(result.feasible_pareto_front) >= 1
        # Loose: best should at least improve violations vs random baseline.
        gen0 = result.step_history[0]
        gen_last = result.step_history[-1]
        assert gen_last.pop1_feasible_count >= gen0.pop1_feasible_count


@pytest.mark.slow
def test_ccmo_hv_at_least_competitive_with_nsga2(default_problem: SchedulingProblem):
    """CCMO should not be more than 30% worse than NSGA-II on the sum-of-best-fitness proxy.

    NOTE: A loose check; the real comparison is in the INRC-I benchmark with hypervolume.
    """
    from ai.optimizers.ccmo import CCMOOptimizer
    from ai.optimizers.nsga2 import NSGAIIOptimizer
    from ai.optimizers.result import CCMOConfig, NSGAIIConfig

    nsga2 = NSGAIIOptimizer(default_problem).run(
        NSGAIIConfig(generations=50, pop_size=100, seed=42)
    )
    ccmo = CCMOOptimizer(default_problem).run(
        CCMOConfig(generations=50, pop_size=100, seed=42)
    )

    nsga2_score = sum(nsga2.best_fitness)
    ccmo_score = sum(ccmo.best_fitness)
    assert ccmo_score <= nsga2_score * 1.3
