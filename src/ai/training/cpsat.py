"""Training CLI for CPSATOptimizer.

Runs the exact-baseline solver against the canonical EnvironmentConfig
(7 employees × 30 days × 3 shifts/day) and writes the result to
checkpoints/cpsat_best_schedule.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.agents.environment import EnvironmentConfig
from ai.domain.problem import SchedulingProblem
from ai.domain.schemas import CPSATConfigSnapshot, CPSATTrainResult
from ai.optimizers.cpsat import CPSATOptimizer
from ai.optimizers.result import CPSATConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the CP-SAT exact-baseline solver."
    )
    parser.add_argument("--timeout-s-per-stage", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--objective-priority",
        default="b2b,spread",
        help="Comma-separated lex priority. Default: 'b2b,spread'.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    priority = [p.strip() for p in args.objective_priority.split(",") if p.strip()]
    config = CPSATConfig(
        timeout_s_per_stage=args.timeout_s_per_stage,
        num_workers=args.num_workers,
        objective_priority=priority,
        seed=args.seed,
    )

    env = EnvironmentConfig()
    problem = SchedulingProblem.from_config(env)

    optimizer = CPSATOptimizer(problem)
    result = optimizer.run(config, verbose=args.verbose)

    snapshot = CPSATConfigSnapshot(
        num_employees=problem.num_employees,
        employee_types=list(problem.employee_types),
        days=problem.days,
        shifts_per_day=problem.shifts_per_day,
        shift_lengths=list(problem.shift_lengths),
        timeout_s_per_stage=config.timeout_s_per_stage,
        num_workers=config.num_workers,
        objective_priority=config.objective_priority,
        seed=config.seed,
    )

    train_result = CPSATTrainResult(
        schedule=result.best_schedule,
        b2b_count=result.b2b_count,
        spread=result.spread,
        jain_index=result.jain_index,
        stages=result.stages,
        config=snapshot,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "cpsat_best_schedule.json"
    out_path.write_text(json.dumps(train_result.model_dump(), indent=2))
    print(f"Wrote {out_path}")
    print(
        f"  b2b={result.b2b_count} spread={result.spread} "
        f"jain={result.jain_index:.4f} wall_clock={result.total_wall_clock_s:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
