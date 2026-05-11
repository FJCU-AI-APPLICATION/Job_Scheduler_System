"""Training CLI for MatheuristicOptimizer.

Runs the matheuristic against the canonical EnvironmentConfig
(7 employees × 30 days × 3 shifts/day) and writes the result to
checkpoints/matheuristic_best_schedule.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem
from ai.domain.schemas import MatheuristicConfigSnapshot, MatheuristicTrainResult
from ai.optimizers.matheuristic import MatheuristicOptimizer
from ai.optimizers.result import MatheuristicConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the IP + VNS / SA matheuristic (#18)."
    )
    parser.add_argument("--acceptance", choices=("vns", "sa"), default="vns")
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--stagnation-limit", type=int, default=20)
    parser.add_argument("--time-budget-s", type=float, default=300.0)
    parser.add_argument("--inner-ip-time-budget-s", type=float, default=5.0)
    parser.add_argument("--inner-ip-workers", type=int, default=4)
    parser.add_argument("--sa-initial-temperature", type=float, default=100.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.95)
    parser.add_argument("--sa-lex-weight-b2b", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Also write step_history to <output-dir>/matheuristic_trajectory.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = MatheuristicConfig(
        acceptance=args.acceptance,
        k_max=args.k_max,
        max_iterations=args.max_iterations,
        stagnation_limit=args.stagnation_limit,
        time_budget_s=args.time_budget_s,
        inner_ip_time_budget_s=args.inner_ip_time_budget_s,
        inner_ip_workers=args.inner_ip_workers,
        sa_initial_temperature=args.sa_initial_temperature,
        sa_cooling_rate=args.sa_cooling_rate,
        sa_lex_weight_b2b=args.sa_lex_weight_b2b,
        seed=args.seed,
    )

    env = EnvironmentConfig()
    problem = SchedulingProblem.from_config(env)

    optimizer = MatheuristicOptimizer(problem)
    result = optimizer.run(config, verbose=args.verbose)

    snapshot = MatheuristicConfigSnapshot(
        num_employees=problem.num_employees,
        employee_types=list(problem.employee_types),
        days=problem.days,
        shifts_per_day=problem.shifts_per_day,
        shift_lengths=list(problem.shift_lengths),
        acceptance=config.acceptance,
        k_max=config.k_max,
        max_iterations=config.max_iterations,
        stagnation_limit=config.stagnation_limit,
        time_budget_s=config.time_budget_s,
        inner_ip_time_budget_s=config.inner_ip_time_budget_s,
        inner_ip_workers=config.inner_ip_workers,
        sa_initial_temperature=config.sa_initial_temperature,
        sa_cooling_rate=config.sa_cooling_rate,
        sa_lex_weight_b2b=config.sa_lex_weight_b2b,
        fairness_alpha=config.fairness_alpha,
        seed=config.seed,
    )

    train_result = MatheuristicTrainResult(
        schedule=result.best_schedule,
        b2b_count=result.b2b_count,
        fairness_gap=result.fairness_gap,
        fairness_metric=result.fairness_metric,
        fairness_alpha=result.fairness_alpha,
        jain_index=result.jain_index,
        total_iterations=result.total_iterations,
        total_accepted=result.total_accepted,
        total_inner_ip_calls=result.total_inner_ip_calls,
        total_inner_ip_failures=result.total_inner_ip_failures,
        neighborhood_usage=result.neighborhood_usage,
        termination_reason=result.termination_reason,
        total_wall_clock_s=result.total_wall_clock_s,
        config=snapshot,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "matheuristic_best_schedule.json"
    out_path.write_text(json.dumps(train_result.model_dump(), indent=2))
    print(f"Wrote {out_path}")

    if args.save_trajectory:
        traj_path = output_dir / "matheuristic_trajectory.json"
        traj_path.write_text(
            json.dumps([s.model_dump() for s in result.step_history], indent=2)
        )
        print(f"Wrote {traj_path}")

    print(
        f"  acceptance={config.acceptance} b2b={result.b2b_count} "
        f"fairness_gap={result.fairness_gap} jain={result.jain_index:.4f} "
        f"wall_clock={result.total_wall_clock_s:.2f}s "
        f"termination={result.termination_reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
