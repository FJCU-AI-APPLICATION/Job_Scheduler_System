"""A/B benchmark runner: hypervolume + Wilcoxon signed-rank.

Runs registered optimizers across a list of INRC-I instances × seeds and
emits a BenchmarkReport. Hypervolume is computed only on the feasible
front (violations == 0) so a large infeasible Pareto front cannot inflate
the score.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import wilcoxon

from ai.benchmarks.inrc1.loader import load_instance
from ai.domain.schemas import (
    BenchmarkAggregate,
    BenchmarkReport,
    BenchmarkRunRecord,
)
from ai.optimizers.base import Optimizer

REFERENCE_POINT: tuple[float, float, float] = (2.0, 1000.0, 100.0)
"""Hypervolume reference point. Dominates all plausible (unfairness,
violations, b2b) tuples for any α in the supported range. The unfairness
upper bound is 2.0 (not 1.0) as a safety margin for α=1 (Nash) where
unfairness can exceed 1 in adversarial cases. Tighten if expanding to
medium/long tracks.
"""

# Wilcoxon signed-rank requires at least this many paired samples per
# instance; below this we report wilcoxon_p=None instead of guessing.
_WILCOXON_MIN_SEEDS = 6


def run_benchmark(
    algorithms: list[str],
    instance_names: list[str],
    seeds: list[int],
    config_overrides: dict[str, Any] | None = None,
    reference_point: tuple[float, float, float] = REFERENCE_POINT,
) -> BenchmarkReport:
    """Run each algorithm × instance × seed, returning a structured report."""
    records: list[BenchmarkRunRecord] = []
    config_overrides = config_overrides or {}

    for algo in algorithms:
        for instance_name in instance_names:
            problem = load_instance(instance_name)
            for seed in seeds:
                optimizer = Optimizer.create(algo, problem)
                config = optimizer.config_class(seed=seed, **config_overrides)

                t0 = time.perf_counter()
                result = optimizer.run(config)
                wall_clock = time.perf_counter() - t0

                feasible_fits = [
                    f for f in result.pareto_fitnesses if f[1] <= 0.0
                ]
                hv = _compute_hypervolume(feasible_fits, reference_point)

                records.append(
                    BenchmarkRunRecord(
                        instance=instance_name,
                        algorithm=algo,
                        seed=seed,
                        hypervolume=hv,
                        feasible_front_size=len(feasible_fits),
                        best_unfairness=result.best_fitness[0],
                        best_violations=result.best_fitness[1],
                        best_b2b=int(result.best_fitness[2]),
                        wall_clock_s=wall_clock,
                    )
                )

    aggregate = _aggregate(records)
    return BenchmarkReport(
        config_summary={
            "algorithms": algorithms,
            "instance_count": len(instance_names),
            "seeds": seeds,
            "config_overrides": config_overrides,
            "reference_point": list(reference_point),
        },
        per_run=records,
        aggregate=aggregate,
    )


def _compute_hypervolume(
    fits: list[tuple[float, float, float]],
    ref: tuple[float, float, float],
) -> float:
    if not fits:
        return 0.0
    indicator = HV(ref_point=np.array(ref))
    return float(indicator(np.array(fits)))


def _aggregate(records: list[BenchmarkRunRecord]) -> list[BenchmarkAggregate]:
    by_instance: dict[str, dict[str, list[float]]] = {}
    for r in records:
        by_instance.setdefault(r.instance, {}).setdefault(
            r.algorithm, []
        ).append(r.hypervolume)

    out: list[BenchmarkAggregate] = []
    for instance, by_algo in by_instance.items():
        nsga = by_algo.get("nsga2", [])
        ccmo = by_algo.get("ccmo", [])
        wp: float | None = None
        if (
            len(nsga) == len(ccmo)
            and len(nsga) >= _WILCOXON_MIN_SEEDS
            and any(a != b for a, b in zip(nsga, ccmo))
        ):
            try:
                wp = float(wilcoxon(nsga, ccmo).pvalue)
            except ValueError:
                wp = None

        out.append(
            BenchmarkAggregate(
                instance=instance,
                nsga2_hv_mean=statistics.mean(nsga) if nsga else None,
                nsga2_hv_std=statistics.stdev(nsga) if len(nsga) > 1 else None,
                nsga2_n_seeds=len(nsga),
                ccmo_hv_mean=statistics.mean(ccmo) if ccmo else None,
                ccmo_hv_std=statistics.stdev(ccmo) if len(ccmo) > 1 else None,
                ccmo_n_seeds=len(ccmo),
                wilcoxon_p=wp,
            )
        )
    return out
