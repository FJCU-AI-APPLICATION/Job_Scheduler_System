"""NSGA-II Genetic Algorithm optimizer for shift scheduling.

Encapsulates fitness evaluation, day-aligned crossover, constraint repair,
and the NSGA-II evolutionary loop. Used by both the training script and
the inference service. Batch fitness evaluation is GPU-accelerated via PyTorch.
"""

import random

import torch
from deap import base, creator, tools
from pydantic import BaseModel, ConfigDict

from domain.problem import SchedulingProblem, get_device

try:
    creator.FitnessMulti
except AttributeError:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
try:
    creator.Individual
except AttributeError:
    creator.create("Individual", list, fitness=creator.FitnessMulti)


class GAConfig(BaseModel):
    """Hyperparameters for the NSGA-II genetic algorithm."""

    generations: int = 200
    pop_size: int = 100
    cxpb: float = 0.7
    mutpb: float = 0.2
    indpb: float = 0.05


class GAResult(BaseModel):
    """Result of a GA optimization run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_schedule: list[int]
    best_fitness: tuple[float, float, float]
    pareto_front: list[list[int]]
    pareto_fitnesses: list[tuple[float, float, float]]
    logbook: tools.Logbook | None = None


class GAOptimizer:
    """NSGA-II optimizer for employee shift scheduling.

    Objectives (all minimized):
      1. Hours imbalance (1 - Jain's fairness index)
      2. Constraint violations (exceeded hours + unavailability)
      3. Back-to-back shift count
    """

    def __init__(self, problem: SchedulingProblem):
        self._problem = problem
        self._device = get_device()
        self._precompute_tensors()

    def _precompute_tensors(self) -> None:
        p = self._problem
        d = self._device

        self._shift_lens = torch.tensor(
            p.shift_lengths, dtype=torch.float64, device=d
        )
        self._shift_types = (
            torch.arange(p.num_shifts, device=d) % p.shifts_per_day
        )
        self._shift_hour_values = self._shift_lens[self._shift_types]

        self._max_hours_t = torch.tensor(
            p.max_hours, dtype=torch.float64, device=d
        )

        self._unavail_day_emp: dict[int, set[int]] = {}
        for day, emp in p.unavailability:
            self._unavail_day_emp.setdefault(day, set()).add(emp)

    def batch_fitness(
        self, individuals: list[list[int]]
    ) -> list[tuple[float, float, float]]:
        """Batch-evaluate fitness for an entire population on GPU/CPU."""
        p = self._problem
        d = self._device
        n = len(individuals)

        pop_t = torch.tensor(individuals, dtype=torch.long, device=d)

        lens = self._shift_hour_values.unsqueeze(0).expand(n, -1)
        hours = torch.zeros(n, p.num_employees, dtype=torch.float64, device=d)
        hours.scatter_add_(1, pop_t, lens)

        sum_h = hours.sum(dim=1)
        sum_sq = hours.pow(2).sum(dim=1)
        jain = torch.where(
            sum_sq > 0,
            sum_h.pow(2) / (p.num_employees * sum_sq),
            torch.ones(n, dtype=torch.float64, device=d),
        )
        imbalance = 1.0 - jain

        exceed = (hours - self._max_hours_t).clamp(min=0).sum(dim=1)

        unavail_count = torch.zeros(n, dtype=torch.float64, device=d)
        for shift_idx in range(p.num_shifts):
            day = shift_idx // p.shifts_per_day
            unavail_emps = self._unavail_day_emp.get(day)
            if unavail_emps is None:
                continue
            assigned = pop_t[:, shift_idx]
            for emp in unavail_emps:
                unavail_count += (assigned == emp).to(torch.float64)

        violations = exceed + unavail_count * 10

        b2b = (pop_t[:, :-1] == pop_t[:, 1:]).sum(dim=1).to(torch.float64)

        return list(
            zip(
                imbalance.tolist(),
                violations.tolist(),
                b2b.tolist(),
            )
        )

    def fitness(self, individual: list[int]) -> tuple[float, float, float]:
        return self.batch_fitness([individual])[0]

    def crossover(self, ind1: list, ind2: list) -> tuple[list, list]:
        """Day-aligned crossover: cut only at day boundaries."""
        day_points = list(
            range(self._problem.shifts_per_day, len(ind1), self._problem.shifts_per_day)
        )
        if not day_points:
            return ind1, ind2
        cx_point = random.choice(day_points)
        ind1[cx_point:], ind2[cx_point:] = ind2[cx_point:].copy(), ind1[cx_point:].copy()
        return ind1, ind2

    def repair(self, individual: list) -> list:
        """Fix assignments where employee is unavailable."""
        p = self._problem
        for i, emp in enumerate(individual):
            day = i // p.shifts_per_day
            if (day, emp) in p.unavailability:
                valid = [
                    e for e in range(p.num_employees) if (day, e) not in p.unavailability
                ]
                if valid:
                    individual[i] = random.choice(valid)
        return individual

    def _evaluate_population(self, individuals: list) -> None:
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        if not invalid_ind:
            return
        fitnesses = self.batch_fitness(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    def _build_toolbox(self, config: GAConfig) -> base.Toolbox:
        p = self._problem
        toolbox = base.Toolbox()
        toolbox.register("attr_emp", random.randint, 0, p.num_employees - 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_emp,
            n=p.num_shifts,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.crossover)
        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=p.num_employees - 1,
            indpb=config.indpb,
        )
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def run(
        self,
        config: GAConfig | None = None,
        verbose: bool = False,
    ) -> GAResult:
        """Run the NSGA-II evolutionary optimization."""
        if config is None:
            config = GAConfig()

        toolbox = self._build_toolbox(config)

        stats: tools.Statistics | None = None
        if verbose:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register(
                "avg_imbalance",
                lambda fits: sum(f[0] for f in fits) / len(fits),
            )
            stats.register(
                "avg_violations",
                lambda fits: sum(f[1] for f in fits) / len(fits),
            )
            stats.register(
                "avg_b2b",
                lambda fits: sum(f[2] for f in fits) / len(fits),
            )

        pareto_front = tools.ParetoFront()
        population = toolbox.population(n=config.pop_size)

        logbook = tools.Logbook()
        if stats:
            logbook.header = ["gen", "nevals"] + stats.fields

        self._evaluate_population(population)
        pareto_front.update(population)

        if stats:
            record = stats.compile(population)
            logbook.record(gen=0, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

        for gen in range(1, config.generations + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for i in range(1, len(offspring), 2):
                if random.random() < config.cxpb:
                    offspring[i - 1], offspring[i] = toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < config.mutpb:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            for ind in offspring:
                self.repair(ind)

            self._evaluate_population(offspring)

            population = toolbox.select(population + offspring, config.pop_size)
            pareto_front.update(population)

            if stats:
                invalid_count = sum(1 for ind in offspring if not ind.fitness.valid)
                record = stats.compile(population)
                logbook.record(gen=gen, nevals=len(offspring) - invalid_count, **record)
                if verbose and gen % 10 == 0:
                    print(logbook.stream)

        best = min(pareto_front, key=lambda ind: sum(ind.fitness.values))

        return GAResult(
            best_schedule=list(best),
            best_fitness=best.fitness.values,
            pareto_front=[list(ind) for ind in pareto_front],
            pareto_fitnesses=[ind.fitness.values for ind in pareto_front],
            logbook=logbook if verbose else None,
        )
