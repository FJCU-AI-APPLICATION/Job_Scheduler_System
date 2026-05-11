"""Regression tests for Pydantic one-release backwards-compatibility aliases.

After Task 3 of the α-fairness rollout (#16), three Pydantic models switched
field names but kept the old names as Field(alias=...) for one release:
  NSGAIIFitnessResult.imbalance      → unfairness
  CCMOFitnessResult.imbalance        → unfairness
  BenchmarkRunRecord.best_imbalance  → best_unfairness

These tests pin the alias behavior so a Pydantic upgrade or accidental
edit can't silently break checkpoint loading.
"""

import pytest


def test_nsga2_fitness_legacy_alias_loads():
    """NSGAIIFitnessResult accepts legacy `imbalance=` kwarg."""
    from ai.domain.schemas import NSGAIIFitnessResult

    legacy = NSGAIIFitnessResult(imbalance=0.5, constraint_violations=0.0, back_to_back=0.0)
    assert legacy.unfairness == 0.5

    canonical = NSGAIIFitnessResult(unfairness=0.5, constraint_violations=0.0, back_to_back=0.0)
    assert canonical.unfairness == 0.5


def test_ccmo_fitness_legacy_alias_loads():
    """CCMOFitnessResult accepts legacy `imbalance=` kwarg."""
    from ai.domain.schemas import CCMOFitnessResult

    legacy = CCMOFitnessResult(imbalance=0.5, constraint_violations=0.0, back_to_back=0.0)
    assert legacy.unfairness == 0.5

    canonical = CCMOFitnessResult(unfairness=0.5, constraint_violations=0.0, back_to_back=0.0)
    assert canonical.unfairness == 0.5


def test_benchmark_record_legacy_alias_loads():
    """BenchmarkRunRecord accepts legacy `best_imbalance=` kwarg."""
    from ai.domain.schemas import BenchmarkRunRecord

    legacy = BenchmarkRunRecord(
        instance="sprint01",
        algorithm="nsga2",
        seed=0,
        hypervolume=0.0,
        feasible_front_size=0,
        best_imbalance=0.1,
        best_violations=0.0,
        best_b2b=0,
        wall_clock_s=0.0,
    )
    assert legacy.best_unfairness == 0.1

    canonical = BenchmarkRunRecord(
        instance="sprint01",
        algorithm="nsga2",
        seed=0,
        hypervolume=0.0,
        feasible_front_size=0,
        best_unfairness=0.1,
        best_violations=0.0,
        best_b2b=0,
        wall_clock_s=0.0,
    )
    assert canonical.best_unfairness == 0.1


def test_aliases_round_trip_via_model_validate():
    """Loading from a dict with legacy keys produces the new field values."""
    from ai.domain.schemas import NSGAIIFitnessResult

    legacy_dict = {"imbalance": 0.7, "constraint_violations": 1.0, "back_to_back": 2.0}
    m = NSGAIIFitnessResult.model_validate(legacy_dict)
    assert m.unfairness == 0.7
    assert m.constraint_violations == 1.0
    assert m.back_to_back == 2.0
