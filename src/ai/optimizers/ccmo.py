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
    # Convert to plain tensor in case obj is a ReadOnlyTensor (e.g. from pool.evals).
    obj = obj.clone()
    le = (obj.unsqueeze(1) <= obj.unsqueeze(0)).all(dim=-1)
    lt = (obj.unsqueeze(1) < obj.unsqueeze(0)).any(dim=-1)
    dominates = (le & lt).clone()  # ensure mutable

    domination_count = dominates.sum(dim=0).clone()
    ranks = torch.full((n,), -1, dtype=torch.long, device=obj.device)
    current_rank = 0
    while True:
        front = (domination_count == 0) & (ranks == -1)
        if not front.any():
            break
        ranks[front] = current_rank
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
    sort_keys = ranks.float() - distances * 1e-9  # break ties by negative distance
    order = sort_keys.argsort()
    selected_idx = order[:pop_size].tolist()
    return pool.take(selected_idx)


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
            best_local = int(e1[:, 1].argmin())
            front_idx = torch.tensor([best_local])
            fell_back = True

        feas_pareto_front = [list(map(int, pop1.values[i].tolist())) for i in front_idx.tolist()]
        feas_pareto_fits = [
            tuple(map(float, pop1.evals[i].tolist())) for i in front_idx.tolist()
        ]
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
