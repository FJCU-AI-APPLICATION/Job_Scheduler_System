"""Benchmark CLI: A/B compare optimizers on the INRC-I sprint track.

Examples:
    python -m ai.training.benchmark --algorithm nsga2 --track sprint --seeds 10
    python -m ai.training.benchmark --algorithm nsga2,ccmo --seeds 10 --report a_b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.benchmarks.inrc1.loader import list_instances
from ai.benchmarks.runner import run_benchmark
from ai.optimizers.base import Optimizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an A/B benchmark across the INRC-I sprint track."
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Comma-separated optimizer names (e.g. 'nsga2' or 'nsga2,ccmo').",
    )
    parser.add_argument(
        "--track",
        default="sprint",
        choices=["sprint"],
        help="INRC-I track. Only 'sprint' is supported in this PR.",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write a JSON report. Prints to stdout when omitted.",
    )
    args = parser.parse_args(argv)

    algorithms = [a.strip() for a in args.algorithm.split(",") if a.strip()]
    available = Optimizer.list_available()
    for a in algorithms:
        if a not in available:
            parser.error(f"Unknown algorithm '{a}'. Available: {available}")

    instances = list_instances(args.track)
    seeds = list(range(args.seeds))

    report = run_benchmark(
        algorithms=algorithms,
        instance_names=instances,
        seeds=seeds,
        config_overrides={
            "generations": args.generations,
            "pop_size": args.pop_size,
            "device": args.device,
        },
    )

    output = json.dumps(report.model_dump(), indent=2)
    if args.report:
        Path(args.report).write_text(output)
        print(f"Report saved to {args.report}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
