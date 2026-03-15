"""
Train a Genetic Algorithm for shift scheduling using DEAP.

Uses tournament selection, two-point crossover, per-gene uniform integer
mutation, and a HallOfFame to track the best solution across generations.

Usage:
    python -m training.ga.train --generations 200 --pop-size 100
    python -m training.ga.train --generations 500 --pop-size 200 --cxpb 0.8 --mutpb 0.3
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

# Default scheduling parameters
NUM_EMPLOYEES = 7
EMPLOYEE_TYPES = ["FT", "FT", "FT", "FT", "PT", "PT", "PT"]
DAYS = 30
SHIFTS_PER_DAY = 3
SHIFT_LENGTHS = [9, 8, 7]
FT_MAX_HOURS = 160
PT_MAX_HOURS = 40

NUM_SHIFTS = DAYS * SHIFTS_PER_DAY
MAX_HOURS = [FT_MAX_HOURS if t == "FT" else PT_MAX_HOURS for t in EMPLOYEE_TYPES]


def _setup_deap_types() -> None:
    """Register DEAP creator types (safe for re-import)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def build_fitness_function(
    num_employees: int,
    shifts_per_day: int,
    shift_lengths: list[int],
    max_hours: list[int],
    num_shifts: int,
) -> callable:
    """Build a fitness function for the scheduling problem."""

    def fitness(individual: list[int]) -> tuple[float]:
        hours_assigned = np.zeros(num_employees)
        for i, emp_idx in enumerate(individual):
            shift_type = i % shifts_per_day
            hours_assigned[emp_idx] += shift_lengths[shift_type]

        # Penalty for uneven distribution
        target = sum(shift_lengths) * (num_shifts // shifts_per_day) / num_employees
        balance_penalty = float(np.sum(np.abs(hours_assigned - target)))

        # Penalty for exceeding max hours
        exceed_penalty = sum(
            max(0, hours_assigned[i] - max_hours[i]) for i in range(num_employees)
        )

        # Penalty for back-to-back shifts by same employee
        back_to_back = sum(
            1 for i in range(num_shifts - 1) if individual[i] == individual[i + 1]
        )

        return (-(balance_penalty + exceed_penalty * 2 + back_to_back),)

    return fitness


def train_ga(
    generations: int = 200,
    pop_size: int = 100,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    tournsize: int = 3,
    indpb: float = 0.05,
    output_dir: str = "checkpoints",
) -> None:
    _setup_deap_types()

    fitness_fn = build_fitness_function(
        num_employees=NUM_EMPLOYEES,
        shifts_per_day=SHIFTS_PER_DAY,
        shift_lengths=SHIFT_LENGTHS,
        max_hours=MAX_HOURS,
        num_shifts=NUM_SHIFTS,
    )

    toolbox = base.Toolbox()
    toolbox.register("attr_emp", random.randint, 0, NUM_EMPLOYEES - 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_emp,
        n=NUM_SHIFTS,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=0,
        up=NUM_EMPLOYEES - 1,
        indpb=indpb,
    )
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(5)

    population = toolbox.population(n=pop_size)

    print(f"Running GA: {generations} generations, pop_size={pop_size}")
    print(f"  cxpb={cxpb}, mutpb={mutpb}, tournsize={tournsize}, indpb={indpb}")
    print(f"  Problem: {NUM_EMPLOYEES} employees, {NUM_SHIFTS} shifts")
    print()

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best = hof[0]
    best_fitness = best.fitness.values[0]

    # Save best schedule
    result = {
        "schedule": list(best),
        "fitness": best_fitness,
        "config": {
            "num_employees": NUM_EMPLOYEES,
            "employee_types": EMPLOYEE_TYPES,
            "days": DAYS,
            "shifts_per_day": SHIFTS_PER_DAY,
            "shift_lengths": SHIFT_LENGTHS,
            "generations": generations,
            "pop_size": pop_size,
            "cxpb": cxpb,
            "mutpb": mutpb,
            "tournsize": tournsize,
            "indpb": indpb,
        },
    }

    result_path = output_path / "ga_best_schedule.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save logbook for analysis
    logbook_path = output_path / "ga_logbook.pkl"
    with open(logbook_path, "wb") as f:
        pickle.dump(logbook, f)

    # Print summary
    print(f"\nBest fitness: {best_fitness:.2f}")

    hours_assigned = np.zeros(NUM_EMPLOYEES)
    for i, emp_idx in enumerate(best):
        shift_type = i % SHIFTS_PER_DAY
        hours_assigned[emp_idx] += SHIFT_LENGTHS[shift_type]

    print("\nHours distribution:")
    for i, hours in enumerate(hours_assigned):
        emp_type = EMPLOYEE_TYPES[i]
        max_h = MAX_HOURS[i]
        status = " OVER" if hours > max_h else ""
        print(f"  Employee {i} ({emp_type}): {hours:.0f}h / {max_h}h{status}")

    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GA for shift scheduling with DEAP")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--cxpb", type=float, default=0.7, help="Crossover probability")
    parser.add_argument("--mutpb", type=float, default=0.2, help="Mutation probability")
    parser.add_argument("--tournsize", type=int, default=3, help="Tournament size")
    parser.add_argument("--indpb", type=float, default=0.05, help="Per-gene mutation probability")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")

    args = parser.parse_args()

    train_ga(
        generations=args.generations,
        pop_size=args.pop_size,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        tournsize=args.tournsize,
        indpb=args.indpb,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
