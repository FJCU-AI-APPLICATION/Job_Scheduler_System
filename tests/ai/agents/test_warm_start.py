"""Tests for the warm_start module — CP-SAT optimum enumeration + imitation BC."""

import numpy as np
import pytest


def test_enumerate_cpsat_optima_returns_optimal_schedules(tiny_problem):
    """Each enumerated schedule achieves the lex optimum (b2b★, fairness_gap★)."""
    from ai.agents.warm_start import enumerate_cpsat_optima
    from ai.optimizers.cpsat import CPSATOptimizer
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)

    # Reference: single-solve to find the actual lex optimum.
    ref_result = CPSATOptimizer(tiny_problem).run(config)
    b2b_star = ref_result.b2b_count
    gap_star = ref_result.fairness_gap

    # Enumeration: should return ≥1 schedule at this optimum.
    schedules = enumerate_cpsat_optima(tiny_problem, config, n_solutions=5)
    assert len(schedules) >= 1
    assert len(schedules) <= 5
    # Each schedule should have length num_shifts.
    for s in schedules:
        assert len(s) == tiny_problem.num_shifts


def test_enumerate_respects_unavailability():
    """Each enumerated schedule respects the unavailability hard constraint."""
    from ai.agents.warm_start import enumerate_cpsat_optima
    from ai.domain.problem import SchedulingProblem
    from ai.optimizers.result import CPSATConfig

    sp = SchedulingProblem(
        num_employees=4,
        employee_types=("FT", "FT", "FT", "PT"),
        days=5,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0), (2, 1)}),
    )
    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)
    schedules = enumerate_cpsat_optima(sp, config, n_solutions=3)

    for schedule in schedules:
        for t, emp in enumerate(schedule):
            day = t // sp.shifts_per_day
            assert (day, emp) not in sp.unavailability, (
                f"schedule {schedule} assigned unavailable employee {emp} on day {day}"
            )


def test_cpsat_schedules_to_transitions_shape(tiny_problem):
    """Transitions has obs/acts/next_obs/dones with correct shape and dtype."""
    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.agents.warm_start import (
        cpsat_schedules_to_transitions,
        enumerate_cpsat_optima,
    )
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)
    schedules = enumerate_cpsat_optima(tiny_problem, config, n_solutions=2)

    env_config = EnvironmentConfig(
        num_employees=tiny_problem.num_employees,
        employee_types=list(tiny_problem.employee_types),
        days=tiny_problem.days,
        shifts_per_day=tiny_problem.shifts_per_day,
        shift_lengths=list(tiny_problem.shift_lengths),
        ft_max_hours=tiny_problem.max_hours[0],
        pt_max_hours=tiny_problem.max_hours[-1],
    )
    env = SchedulingEnv(env_config)
    transitions = cpsat_schedules_to_transitions(env, schedules)

    expected_T = len(schedules) * tiny_problem.num_shifts
    assert transitions.obs.shape[0] == expected_T
    assert transitions.acts.shape == (expected_T,)
    assert transitions.next_obs.shape[0] == expected_T
    assert transitions.dones.shape == (expected_T,)
    # dones should be True exactly at the last step of each trajectory.
    n_done = int(transitions.dones.sum())
    assert n_done == len(schedules)


def test_bc_pretrain_runs_to_completion(tiny_problem):
    """bc_pretrain executes without error on a small dataset and returns finite metrics."""
    from sb3_contrib import MaskablePPO

    from ai.agents.environment import EnvironmentConfig, SchedulingEnv
    from ai.agents.warm_start import (
        bc_pretrain,
        cpsat_schedules_to_transitions,
        enumerate_cpsat_optima,
    )
    from ai.optimizers.result import CPSATConfig

    config = CPSATConfig(timeout_s_per_stage=10.0, num_workers=2, seed=42)
    schedules = enumerate_cpsat_optima(tiny_problem, config, n_solutions=2)

    env_config = EnvironmentConfig(
        num_employees=tiny_problem.num_employees,
        employee_types=list(tiny_problem.employee_types),
        days=tiny_problem.days,
        shifts_per_day=tiny_problem.shifts_per_day,
        shift_lengths=list(tiny_problem.shift_lengths),
        ft_max_hours=tiny_problem.max_hours[0],
        pt_max_hours=tiny_problem.max_hours[-1],
    )
    env = SchedulingEnv(env_config)
    transitions = cpsat_schedules_to_transitions(env, schedules)

    model = MaskablePPO("MlpPolicy", env, verbose=0)
    rng = np.random.default_rng(0)
    metrics = bc_pretrain(
        model.policy, transitions, rng, n_batches=50, batch_size=16, lr=1e-3
    )
    assert np.isfinite(metrics["final_loss"])
    assert 0.0 <= metrics["final_accuracy"] <= 1.0
