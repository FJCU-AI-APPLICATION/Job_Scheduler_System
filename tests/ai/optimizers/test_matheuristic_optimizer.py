"""Tests for MatheuristicOptimizer — issue #18 acceptance criteria."""

import pytest
from pydantic import ValidationError


def test_alpha_inf_validator():
    """MatheuristicConfig(fairness_alpha=2.0) raises ValidationError."""
    from ai.optimizers.result import MatheuristicConfig

    with pytest.raises(ValidationError):
        MatheuristicConfig(fairness_alpha=2.0)
    with pytest.raises(ValidationError):
        MatheuristicConfig(fairness_alpha=0.0)

    # inf is fine
    MatheuristicConfig(fairness_alpha=float("inf"))


def test_acceptance_validator():
    """MatheuristicConfig.acceptance must be 'vns' or 'sa'."""
    from ai.optimizers.result import MatheuristicConfig

    with pytest.raises(ValidationError):
        MatheuristicConfig(acceptance="bogus")

    MatheuristicConfig(acceptance="vns")
    MatheuristicConfig(acceptance="sa")


def test_config_defaults():
    """MatheuristicConfig() has the documented defaults."""
    from ai.optimizers.result import MatheuristicConfig

    c = MatheuristicConfig()
    assert c.acceptance == "vns"
    assert c.k_max == 3
    assert c.max_iterations == 100
    assert c.stagnation_limit == 20
    assert c.time_budget_s == 300.0
    assert c.inner_ip_time_budget_s == 5.0
    assert c.inner_ip_workers == 4
    assert c.sa_initial_temperature == 100.0
    assert c.sa_cooling_rate == 0.95
    assert c.sa_lex_weight_b2b == 1000.0
    assert c.fairness_alpha == float("inf")


def test_result_schemas_import():
    """Schemas import without circular-dep issues."""
    from ai.domain.schemas import MatheuristicConfigSnapshot, MatheuristicTrainResult
    from ai.optimizers.result import MatheuristicResult, MatheuristicStepStatus

    # Sanity-check default-construct of StepStatus
    s = MatheuristicStepStatus(
        iteration=0,
        neighborhood="swap_day",
        size_k=1,
        accepted=False,
        candidate_b2b=None,
        candidate_fairness_gap=None,
        incumbent_b2b=5,
        incumbent_fairness_gap=10,
        best_b2b=5,
        best_fairness_gap=10,
        temperature=100.0,
        inner_ip_wall_clock_s=0.3,
        cumulative_wall_clock_s=0.3,
    )
    assert s.iteration == 0


def test_runs_via_create(tiny_problem):
    """Optimizer.create('matheuristic', sp) returns a MatheuristicOptimizer."""
    from ai.optimizers.base import Optimizer
    from ai.optimizers.matheuristic import MatheuristicOptimizer

    optimizer = Optimizer.create("matheuristic", tiny_problem)
    assert isinstance(optimizer, MatheuristicOptimizer)


def test_init_random_feasible_respects_unavailability():
    """_init_random_feasible never picks an unavailable (day, employee)."""
    import numpy as np

    from ai.domain.problem import SchedulingProblem
    from ai.optimizers.matheuristic import _init_random_feasible

    sp = SchedulingProblem(
        num_employees=4,
        employee_types=("FT", "FT", "FT", "PT"),
        days=5,
        shifts_per_day=2,
        shift_lengths=(8, 8),
        max_hours=(50, 50, 50, 20),
        unavailability=frozenset({(0, 0), (2, 1)}),
    )
    rng = np.random.default_rng(0)
    schedule = _init_random_feasible(sp, rng)

    for t, emp in enumerate(schedule):
        day = t // sp.shifts_per_day
        assert (day, emp) not in sp.unavailability


def test_init_random_feasible_raises_on_no_availability():
    """If every employee is unavailable on day 0, raise MatheuristicError."""
    import numpy as np

    from ai.domain.problem import SchedulingProblem
    from ai.optimizers.matheuristic import MatheuristicError, _init_random_feasible

    sp = SchedulingProblem(
        num_employees=2,
        employee_types=("FT", "PT"),
        days=3,
        shifts_per_day=1,
        shift_lengths=(8,),
        max_hours=(50, 20),
        unavailability=frozenset({(0, 0), (0, 1)}),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(MatheuristicError):
        _init_random_feasible(sp, rng)


def test_inner_ip_solve_returns_schedule(tiny_problem):
    """_inner_ip_solve produces a feasible full schedule respecting frozen vars."""
    import numpy as np

    from ai.optimizers.matheuristic import _init_random_feasible, _inner_ip_solve

    rng = np.random.default_rng(0)
    incumbent = _init_random_feasible(tiny_problem, rng)

    frozen = list(range(6))
    optimize = list(range(6, tiny_problem.num_shifts))

    result = _inner_ip_solve(
        sp=tiny_problem,
        incumbent=incumbent,
        frozen=frozen,
        optimize=optimize,
        time_budget_s=10.0,
        workers=2,
    )
    assert result is not None
    assert len(result) == tiny_problem.num_shifts
    for t in frozen:
        assert result[t] == incumbent[t]
    for t in optimize:
        assert 0 <= result[t] < tiny_problem.num_employees


def test_inner_ip_solve_returns_none_on_timeout(tiny_problem, mocker):
    """Mocking CP-SAT to return UNKNOWN → _inner_ip_solve returns None."""
    import numpy as np
    from ortools.sat.python import cp_model

    from ai.optimizers.matheuristic import _init_random_feasible, _inner_ip_solve

    mocker.patch.object(cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN)
    mocker.patch.object(cp_model.CpSolver, "WallTime", return_value=0.1)

    rng = np.random.default_rng(0)
    incumbent = _init_random_feasible(tiny_problem, rng)
    result = _inner_ip_solve(
        sp=tiny_problem,
        incumbent=incumbent,
        frozen=[],
        optimize=list(range(tiny_problem.num_shifts)),
        time_budget_s=1.0,
        workers=1,
    )
    assert result is None


def test_result_shape(tiny_problem):
    """MatheuristicResult carries all advertised fields; best_fitness is 3-tuple."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        acceptance="vns",
        max_iterations=5,
        stagnation_limit=5,
        time_budget_s=30.0,
        inner_ip_time_budget_s=2.0,
        inner_ip_workers=2,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    assert len(result.best_schedule) == tiny_problem.num_shifts
    assert len(result.best_fitness) == 3
    unfairness, violations, b2b = result.best_fitness
    assert violations == 0.0
    assert 0.0 <= unfairness <= 1.0
    assert b2b == float(result.b2b_count)
    assert result.fairness_alpha == float("inf")
    assert result.termination_reason in {"time_budget", "max_iterations", "stagnation"}
    assert len(result.step_history) > 0


def test_returns_feasible_schedule(tiny_problem):
    """Best schedule respects unavailability + max_hours."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=3, time_budget_s=30.0, inner_ip_time_budget_s=2.0, seed=42
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    for t, emp in enumerate(result.best_schedule):
        day = t // tiny_problem.shifts_per_day
        assert (day, emp) not in tiny_problem.unavailability

    hours = [0] * tiny_problem.num_employees
    for t, emp in enumerate(result.best_schedule):
        shift_idx = t % tiny_problem.shifts_per_day
        hours[emp] += tiny_problem.shift_lengths[shift_idx]
    for e, h in enumerate(hours):
        assert h <= tiny_problem.max_hours[e]


def test_neighborhood_usage_logged(tiny_problem):
    """All three neighborhoods appear in result.neighborhood_usage."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=10, time_budget_s=60.0, inner_ip_time_budget_s=2.0, seed=42
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    expected = {"swap_day", "swap_shift_block", "swap_employee"}
    assert expected == set(result.neighborhood_usage.keys())
    assert sum(result.neighborhood_usage.values()) == result.total_inner_ip_calls


def test_vns_acceptance_is_strictly_lex(tiny_problem):
    """Every accepted=True step in VNS history is strictly lex-better."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        acceptance="vns",
        max_iterations=10,
        time_budget_s=60.0,
        inner_ip_time_budget_s=2.0,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    prev_b2b = None
    prev_fair = None
    for step in result.step_history:
        if step.accepted:
            if prev_b2b is not None:
                assert (step.candidate_b2b, step.candidate_fairness_gap) < (
                    prev_b2b,
                    prev_fair,
                ), f"VNS accepted non-lex-dominating candidate at iter {step.iteration}"
        prev_b2b = step.incumbent_b2b
        prev_fair = step.incumbent_fairness_gap


def test_sa_accept_improving():
    """At any T, SA always accepts strictly improving candidates."""
    import numpy as np

    from ai.optimizers.matheuristic import _accept_sa

    rng = np.random.default_rng(0)
    assert _accept_sa(
        cand_b2b=4, cand_fair=9, inc_b2b=5, inc_fair=10,
        temperature=1.0, weight=1000.0, rng=rng,
    ) is True


def test_sa_can_reject_at_low_T():
    """At T→0, SA rejects worse candidates almost surely."""
    import numpy as np

    from ai.optimizers.matheuristic import _accept_sa

    rng = np.random.default_rng(0)
    decisions = [
        _accept_sa(
            cand_b2b=6, cand_fair=10, inc_b2b=5, inc_fair=10,
            temperature=1e-6, weight=1000.0, rng=rng,
        )
        for _ in range(20)
    ]
    assert not any(decisions)


def test_sa_can_accept_worse_at_high_T(tiny_problem):
    """At T=1e9, SA's run accepts at least one lex-worse candidate."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        acceptance="sa",
        sa_initial_temperature=1e9,
        sa_cooling_rate=1.0,
        max_iterations=20,
        time_budget_s=120.0,
        inner_ip_time_budget_s=2.0,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    accept_not_dominating_best = [
        s for s in result.step_history
        if s.accepted
        and s.candidate_b2b is not None
        and not (
            (s.candidate_b2b, s.candidate_fairness_gap)
            < (s.best_b2b, s.best_fairness_gap)
        )
    ]
    assert len(accept_not_dominating_best) >= 1


def test_termination_reason_max_iterations(tiny_problem):
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=2,
        stagnation_limit=1000,
        time_budget_s=3600.0,
        inner_ip_time_budget_s=2.0,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)
    assert result.termination_reason == "max_iterations"


def test_termination_reason_stagnation(tiny_problem):
    """stagnation_limit=2 trips before max_iterations=20 on tiny_problem with seed 42.

    Catches off-by-one and reset-on-uphill bugs in the stagnation counter.
    """
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=20,
        stagnation_limit=2,
        time_budget_s=3600.0,
        inner_ip_time_budget_s=2.0,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)
    assert result.termination_reason == "stagnation"


def test_termination_reason_time_budget(tiny_problem):
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=1000,
        stagnation_limit=1000,
        time_budget_s=0.5,
        inner_ip_time_budget_s=1.0,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)
    assert result.termination_reason == "time_budget"


def test_inner_ip_failure_does_not_crash(tiny_problem):
    """inner_ip_time_budget_s=0.001 forces timeouts; run completes."""
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=5,
        time_budget_s=120.0,
        inner_ip_time_budget_s=0.001,
        seed=42,
    )
    result = MatheuristicOptimizer(tiny_problem).run(config)

    assert result.total_inner_ip_failures > 0
    for t, emp in enumerate(result.best_schedule):
        day = t // tiny_problem.shifts_per_day
        assert (day, emp) not in tiny_problem.unavailability


def test_improves_or_matches_random_init(tiny_problem):
    """`best` is monotone: final ≤ initial in lex order."""
    import numpy as np

    from ai.optimizers.matheuristic import (
        MatheuristicOptimizer,
        _compute_b2b,
        _compute_fairness_gap,
        _init_random_feasible,
    )
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=10, time_budget_s=60.0, inner_ip_time_budget_s=2.0, seed=42
    )
    rng = np.random.default_rng(42)
    init = _init_random_feasible(tiny_problem, rng)
    init_b2b = _compute_b2b(init)
    init_fair = _compute_fairness_gap(init, tiny_problem)

    result = MatheuristicOptimizer(tiny_problem).run(config)
    assert (result.b2b_count, result.fairness_gap) <= (init_b2b, init_fair)


def test_seed_reproducibility(tiny_problem):
    """Same seed → bit-identical best_schedule and step_history.

    `inner_ip_workers=1` is required: CP-SAT's multi-worker search is
    non-deterministic by design (workers race on subproblems). Do NOT bump
    this for speed without also reworking the test's invariants.
    """
    from ai.optimizers.matheuristic import MatheuristicOptimizer
    from ai.optimizers.result import MatheuristicConfig

    config = MatheuristicConfig(
        max_iterations=5, time_budget_s=60.0, inner_ip_time_budget_s=2.0,
        inner_ip_workers=1,
        seed=123,
    )
    a = MatheuristicOptimizer(tiny_problem).run(config)
    b = MatheuristicOptimizer(tiny_problem).run(config)

    assert a.best_schedule == b.best_schedule
    assert len(a.step_history) == len(b.step_history)
    for sa_step, sb_step in zip(a.step_history, b.step_history):
        assert sa_step.neighborhood == sb_step.neighborhood
        assert sa_step.size_k == sb_step.size_k


def test_inference_service_dispatch():
    """run_optimizer_inference('matheuristic', request, ...) returns a response."""
    from ai.domain.schemas import (
        EmployeeInfo,
        SchedulingRequest,
        ShiftInfo,
        UnavailabilityInfo,
    )
    from ai.services.optimizer_inference import run_optimizer_inference

    request = SchedulingRequest(
        employees=[
            EmployeeInfo(id=1, employee_type="FT", max_hours=40),
            EmployeeInfo(id=2, employee_type="FT", max_hours=40),
            EmployeeInfo(id=3, employee_type="PT", max_hours=20),
        ],
        shifts=[
            ShiftInfo(start_time="08:00", end_time="16:00", length_hours=8),
            ShiftInfo(start_time="16:00", end_time="00:00", length_hours=8),
        ],
        days=5,
        unavailability=[UnavailabilityInfo(employee_id=1, day=0)],
    )

    response = run_optimizer_inference(
        "matheuristic",
        request,
        config_overrides={
            "max_iterations": 3,
            "time_budget_s": 30.0,
            "inner_ip_time_budget_s": 2.0,
            "seed": 42,
        },
    )
    assert len(response.schedule) == 5 * 2  # 5 days × 2 shifts
    assert response.metrics is not None

    # Verify the dispatch layer actually honors the request's unavailability:
    # employee 1 is blocked on day 0, so neither shift on day 0 should be theirs.
    day0_assignments = [a.employee_id for a in response.schedule if a.day == 0]
    assert 1 not in day0_assignments, (
        f"Employee 1 was unavailable on day 0 but got assigned: {day0_assignments}"
    )
