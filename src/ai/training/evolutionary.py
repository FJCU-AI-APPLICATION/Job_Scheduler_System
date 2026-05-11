"""Train an evolutionary optimizer for shift scheduling.

Usage:
    python -m ai.training.evolutionary --algorithm nsga2 --generations 200 --pop-size 100
    python -m ai.training.evolutionary --algorithm ccmo  --generations 200 --pop-size 100 --device cuda
"""

import argparse
import json
from pathlib import Path

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem, jain_fairness_index
from ai.optimizers.base import Optimizer
from ai.optimizers.result import CCMOResult


def train_evolutionary(
    algorithm: str,
    generations: int,
    pop_size: int,
    cxpb: float,
    mutpb: float,
    indpb: float,
    seed: int | None,
    device: str,
    output_dir: str,
    fairness_alpha: float = 2.0,
) -> None:
    config_environment = EnvironmentConfig()
    problem = SchedulingProblem.from_config(config_environment)

    optimizer = Optimizer.create(algorithm, problem)
    config = optimizer.config_class(
        generations=generations,
        pop_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        indpb=indpb,
        seed=seed,
        device=device,
        fairness_alpha=fairness_alpha,
    )

    print(f"Running {algorithm}: {generations} generations, pop_size={pop_size}")
    print(f"  cxpb={cxpb}, mutpb={mutpb}, indpb={indpb}, device={device}, fairness_alpha={fairness_alpha}")
    print(f"  Problem: {problem.num_employees} employees, {problem.num_shifts} shifts")
    print()

    result = optimizer.run(config, verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schedule_path = output_path / f"{algorithm}_best_schedule.json"
    history_path = output_path / f"{algorithm}_step_history.json"

    schedule_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "schedule": result.best_schedule,
                "best_fitness": list(result.best_fitness),
                "pareto_front_size": len(result.pareto_front),
                "config": config.model_dump(),
            },
            indent=2,
        )
    )
    history_path.write_text(
        json.dumps([s.model_dump() for s in result.step_history], indent=2)
    )

    # Export feasible Pareto front for downstream RL warm-start (#17).
    if isinstance(result, CCMOResult):
        # CCMO explicitly separates the feasible front from the auxiliary front.
        # In degenerate runs CCMO may report fell_back_to_auxiliary=True (its
        # "feasible" front then contains a constraint-violating fallback); filter
        # defensively so the downstream consumer's feasibility contract holds.
        src_front = result.feasible_pareto_front
        src_fits = result.feasible_pareto_fitnesses
    else:
        # NSGA-II's pareto_front includes infeasible entries — filter them out
        src_front = result.pareto_front
        src_fits = result.pareto_fitnesses
    feasible_front = [s for s, f in zip(src_front, src_fits) if f[1] == 0.0]
    feasible_fits = [f for s, f in zip(src_front, src_fits) if f[1] == 0.0]

    pareto_path = output_path / f"{algorithm}_pareto_front.json"
    pareto_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "fairness_alpha": config.fairness_alpha,
                "hv_reference_point": [2.0, 1000.0, 100.0],
                "points": [list(f) for f in feasible_fits],
                "schedules": feasible_front,
            },
            indent=2,
        )
    )
    print(f"Pareto front written to {pareto_path} ({len(feasible_front)} feasible solutions)")

    hours = problem.compute_hours(result.best_schedule)
    jain = jain_fairness_index(hours)
    print(
        f"\nBest: unfairness={result.best_fitness[0]:.4f}, "
        f"violations={result.best_fitness[1]:.1f}, b2b={result.best_fitness[2]:.0f}"
    )
    print(f"Jain's fairness index: {jain:.4f}")
    print(f"Pareto front size: {len(result.pareto_front)} solutions")

    print("\nHours distribution:")
    for i, h in enumerate(hours):
        emp_type = problem.employee_types[i]
        max_h = problem.max_hours[i]
        marker = " OVER" if h > max_h else ""
        print(f"  Employee {i} ({emp_type}): {h:.0f}h / {max_h}h{marker}")

    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an evolutionary optimizer for shift scheduling"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=Optimizer.list_available(),
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument("--indpb", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--fairness-alpha",
        type=float,
        default=2.0,
        help="α-fairness parameter (0=utilitarian, 1=Nash, 2=Jain (default), large value≈max-min).",
    )
    args = parser.parse_args()
    train_evolutionary(**vars(args))


if __name__ == "__main__":
    main()
