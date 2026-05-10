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
        # compute_pareto_ranks returns (ranks, crowdsort_ranks) tuple
        ranks, _ = searcher.population.compute_pareto_ranks()
        return GAStepStatus(
            generation=gen,
            mean_obj0_imbalance=float(means[0]),
            mean_obj1_violations=float(means[1]),
            mean_obj2_b2b=float(means[2]),
            pareto_front_size=int((ranks == 0).sum()),
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
        # compute_pareto_ranks returns (ranks, crowdsort_ranks) tuple
        ranks, _ = population.compute_pareto_ranks()
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
