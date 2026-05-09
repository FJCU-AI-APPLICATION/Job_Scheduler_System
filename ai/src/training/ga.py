"""Train a multi-objective Genetic Algorithm for shift scheduling using DEAP + NSGA-II.

Objectives (all minimized):
  1. Hours imbalance (1 - Jain's fairness index)
  2. Constraint violations (max hours exceeded + unavailability)
  3. Back-to-back shift count

Usage:
    python -m training.ga --generations 200 --pop-size 100
    python -m training.ga --generations 500 --pop-size 200 --cxpb 0.8 --mutpb 0.3
"""

import argparse
import json
import pickle
from pathlib import Path

from agents.environment import EnvironmentConfig
from domain.problem import SchedulingProblem, jain_fairness_index
from domain.schemas import GAConfigSnapshot, GAFitnessResult, GATrainResult
from optimizers.ga import GAConfig, GAOptimizer


def train_ga(
    generations: int = 200,
    pop_size: int = 100,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    indpb: float = 0.05,
    output_dir: str = "checkpoints",
    unavail_set: set[tuple[int, int]] | None = None,
) -> None:
    config = EnvironmentConfig(unavailability=unavail_set or set())
    problem = SchedulingProblem.from_config(config)

    ga_config = GAConfig(
        generations=generations,
        pop_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        indpb=indpb,
    )

    optimizer = GAOptimizer(problem)

    print(f"Running NSGA-II GA: {generations} generations, pop_size={pop_size}")
    print(f"  cxpb={cxpb}, mutpb={mutpb}, indpb={indpb}")
    print(f"  Problem: {problem.num_employees} employees, {problem.num_shifts} shifts")
    print("  Objectives: imbalance, constraint_violations, back_to_back")
    print()

    result = optimizer.run(ga_config, verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_fitness = result.best_fitness
    ga_result = GATrainResult(
        schedule=result.best_schedule,
        fitness=GAFitnessResult(
            imbalance=best_fitness[0],
            constraint_violations=best_fitness[1],
            back_to_back=best_fitness[2],
        ),
        pareto_front_size=len(result.pareto_front),
        config=GAConfigSnapshot(
            num_employees=problem.num_employees,
            employee_types=list(problem.employee_types),
            days=problem.days,
            shifts_per_day=problem.shifts_per_day,
            shift_lengths=list(problem.shift_lengths),
            generations=generations,
            pop_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            indpb=indpb,
        ),
    )

    result_path = output_path / "ga_best_schedule.json"
    with open(result_path, "w") as f:
        json.dump(ga_result.model_dump(), f, indent=2)

    if result.logbook is not None:
        logbook_path = output_path / "ga_logbook.pkl"
        with open(logbook_path, "wb") as f:
            pickle.dump(result.logbook, f)

    print(f"\nPareto front size: {len(result.pareto_front)} solutions")
    print(
        f"Best (lowest total): imbalance={best_fitness[0]:.4f}, "
        f"violations={best_fitness[1]:.1f}, b2b={best_fitness[2]:.0f}"
    )

    hours = problem.compute_hours(result.best_schedule)
    jain = jain_fairness_index(hours)
    print(f"Jain's fairness index: {jain:.4f}")

    print("\nHours distribution:")
    for i, h in enumerate(hours):
        emp_type = problem.employee_types[i]
        max_h = problem.max_hours[i]
        status = " OVER" if h > max_h else ""
        print(f"  Employee {i} ({emp_type}): {h:.0f}h / {max_h}h{status}")

    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NSGA-II GA for shift scheduling with DEAP"
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument("--indpb", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="checkpoints")

    args = parser.parse_args()

    train_ga(
        generations=args.generations,
        pop_size=args.pop_size,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        indpb=args.indpb,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
