"""End-to-end smoke + isolated aggregate tests for the benchmark runner."""

import shutil
from pathlib import Path

import pytest

from ai.benchmarks.runner import _aggregate, _compute_hypervolume
from ai.domain.schemas import BenchmarkRunRecord

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "inrc1"
    / "sprint01.xml"
)


@pytest.fixture
def stub_inrc1_data(tmp_path, monkeypatch):
    target_dir = tmp_path / "data" / "benchmarks" / "inrc1"
    target_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_PATH, target_dir / "sprint01.xml")
    monkeypatch.setattr("ai.benchmarks.inrc1.loader.DATA_DIR", target_dir)
    return target_dir


def test_compute_hypervolume_empty_returns_zero():
    assert _compute_hypervolume([], (1.0, 1000.0, 100.0)) == 0.0


def test_compute_hypervolume_dominated_point_is_positive():
    """A point strictly inside the reference box yields HV > 0."""
    hv = _compute_hypervolume([(0.5, 100.0, 50.0)], (1.0, 1000.0, 100.0))
    assert hv > 0.0


def test_aggregate_skips_wilcoxon_below_min_seeds():
    """With < 6 paired seeds we report wilcoxon_p=None instead of a small-N p-value."""
    records = [
        BenchmarkRunRecord(
            instance="sprint01",
            algorithm=algo,
            seed=s,
            hypervolume=hv,
            feasible_front_size=1,
            best_imbalance=0.1,
            best_violations=0.0,
            best_b2b=5,
            wall_clock_s=0.1,
        )
        for algo, hvs in [("nsga2", [10.0, 11.0]), ("ccmo", [12.0, 13.0])]
        for s, hv in enumerate(hvs)
    ]

    aggs = _aggregate(records)

    assert len(aggs) == 1
    agg = aggs[0]
    assert agg.nsga2_n_seeds == 2
    assert agg.ccmo_n_seeds == 2
    assert agg.wilcoxon_p is None


def test_aggregate_runs_wilcoxon_at_or_above_min_seeds():
    """With ≥ 6 paired non-equal seeds we get a real p-value."""
    nsga_hvs = [10.0, 11.0, 9.5, 10.5, 12.0, 11.5]
    ccmo_hvs = [9.0, 10.0, 8.5, 9.5, 11.0, 10.5]
    records = [
        BenchmarkRunRecord(
            instance="sprint01",
            algorithm=algo,
            seed=s,
            hypervolume=hv,
            feasible_front_size=1,
            best_imbalance=0.1,
            best_violations=0.0,
            best_b2b=5,
            wall_clock_s=0.1,
        )
        for algo, hvs in [("nsga2", nsga_hvs), ("ccmo", ccmo_hvs)]
        for s, hv in enumerate(hvs)
    ]

    aggs = _aggregate(records)

    assert aggs[0].wilcoxon_p is not None
    assert 0.0 <= aggs[0].wilcoxon_p <= 1.0


@pytest.mark.benchmark
def test_run_nsga2_on_sprint01_one_seed_smoke(stub_inrc1_data):
    """End-to-end: nsga2 × sprint01 × 1 seed × 5 gens completes under 30s."""
    from ai.benchmarks.runner import run_benchmark

    report = run_benchmark(
        algorithms=["nsga2"],
        instance_names=["sprint01"],
        seeds=[42],
        config_overrides={"generations": 5, "pop_size": 20},
    )

    assert len(report.per_run) == 1
    record = report.per_run[0]
    assert record.algorithm == "nsga2"
    assert record.instance == "sprint01"
    assert record.hypervolume >= 0.0
    assert record.wall_clock_s < 30.0
    assert len(report.aggregate) == 1
