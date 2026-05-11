"""Tests for the α-fairness primitive — issue #16 acceptance criteria + invariants."""

import math

import numpy as np
import pytest
import torch


# === Canonical α values ===


def test_utilitarian_alpha_zero():
    """α=0 → welfare = Σ x_i."""
    from ai.domain.fairness import alpha_fairness

    assert alpha_fairness([1, 2, 3, 4], alpha=0.0) == pytest.approx(10.0)
    assert alpha_fairness([0, 0, 5], alpha=0.0) == pytest.approx(5.0)


def test_nash_welfare_alpha_one():
    """α=1 → welfare = Σ log(max(x_i, ε)). For x_i = e, n equal employees → n."""
    from ai.domain.fairness import alpha_fairness

    e = math.e
    assert alpha_fairness([e, e, e], alpha=1.0) == pytest.approx(3.0, abs=1e-9)


def test_jain_equivalent_alpha_two_uses_jain_formula():
    """α=2 → (Σx)² / (n · Σx²). Test against the formula directly for float parity."""
    from ai.domain.fairness import alpha_fairness

    values = [10, 20, 30, 40, 50]
    n = len(values)
    expected_jain = sum(values) ** 2 / (n * sum(v * v for v in values))
    assert alpha_fairness(values, alpha=2.0) == pytest.approx(expected_jain, abs=1e-12)


def test_rawlsian_alpha_inf():
    """α→∞ → welfare = min(x_i)."""
    from ai.domain.fairness import alpha_fairness

    assert alpha_fairness([10, 20, 5, 100], alpha=float("inf")) == 5.0


def test_general_alpha_three():
    """α=3 → (1/(1−α)) · Σ x_i^(1−α) = -1/2 · Σ 1/x_i²."""
    from ai.domain.fairness import alpha_fairness

    values = [1.0, 2.0, 4.0]
    expected = -0.5 * (1.0 + 0.25 + 0.0625)  # = -0.5 * 1.3125
    assert alpha_fairness(values, alpha=3.0) == pytest.approx(expected, abs=1e-12)


# === welfare_uniform ===


def test_welfare_uniform_alpha_two_is_one():
    """At uniform distribution, α=2 welfare = 1 (the Jain max)."""
    from ai.domain.fairness import welfare_uniform

    assert welfare_uniform(alpha=2.0, total=100.0, n=5) == pytest.approx(1.0)


def test_welfare_uniform_alpha_inf_is_mean():
    """At uniform distribution, α=∞ welfare = total/n."""
    from ai.domain.fairness import welfare_uniform

    assert welfare_uniform(alpha=float("inf"), total=100.0, n=5) == pytest.approx(20.0)


def test_welfare_uniform_alpha_zero_is_total():
    """At uniform distribution, α=0 welfare = total."""
    from ai.domain.fairness import welfare_uniform

    assert welfare_uniform(alpha=0.0, total=100.0, n=5) == pytest.approx(100.0)


# === aggregate_fairness ===


def test_aggregate_welfare_dispatches_to_alpha_fairness():
    """kind='welfare' returns alpha_fairness directly."""
    from ai.domain.fairness import aggregate_fairness, alpha_fairness

    values = [3, 6, 9]
    assert aggregate_fairness(values, alpha=2.0, kind="welfare") == pytest.approx(
        alpha_fairness(values, alpha=2.0)
    )


def test_aggregate_unfairness_alpha_two_equals_one_minus_jain():
    """kind='unfairness' at α=2 ≡ 1 − jain."""
    from ai.domain.fairness import aggregate_fairness, alpha_fairness

    values = [10, 20, 30, 40, 50]
    expected = 1.0 - alpha_fairness(values, alpha=2.0)
    assert aggregate_fairness(values, alpha=2.0, kind="unfairness") == pytest.approx(
        expected, abs=1e-12
    )


def test_aggregate_unfairness_at_uniform_is_zero():
    """At uniform distribution, unfairness = 0 at α∈{2, ∞}."""
    from ai.domain.fairness import aggregate_fairness

    uniform = [20.0, 20.0, 20.0, 20.0, 20.0]
    assert aggregate_fairness(uniform, alpha=2.0, kind="unfairness") == pytest.approx(0.0, abs=1e-12)
    assert aggregate_fairness(uniform, alpha=float("inf"), kind="unfairness") == pytest.approx(0.0, abs=1e-12)


def test_aggregate_unfairness_alpha_inf_is_one_minus_n_min_over_total():
    """At α=∞, unfairness = 1 − n·min/total."""
    from ai.domain.fairness import aggregate_fairness

    values = [10.0, 20.0, 30.0]   # total=60, n=3, min=10
    expected = 1.0 - 3 * 10.0 / 60.0   # = 0.5
    assert aggregate_fairness(values, alpha=float("inf"), kind="unfairness") == pytest.approx(
        expected, abs=1e-12
    )


def test_aggregate_unfairness_bad_kind_raises():
    """Invalid kind raises ValueError."""
    from ai.domain.fairness import aggregate_fairness

    with pytest.raises(ValueError):
        aggregate_fairness([1, 2, 3], alpha=2.0, kind="nope")


# === Edge cases ===


def test_empty_input_returns_zero_welfare():
    """n=0 → welfare = 0 for α∈{0,1,∞}; α=2 returns 1.0 for bit-identical
    Jain regression."""
    from ai.domain.fairness import alpha_fairness

    assert alpha_fairness([], alpha=2.0) == pytest.approx(1.0)
    assert alpha_fairness([], alpha=0.0) == pytest.approx(0.0)
    assert alpha_fairness([], alpha=1.0) == pytest.approx(0.0)
    assert alpha_fairness([], alpha=float("inf")) == pytest.approx(0.0)


def test_single_value_alpha_two_returns_one():
    """n=1 → Jain = 1 (perfectly fair to one person)."""
    from ai.domain.fairness import alpha_fairness

    assert alpha_fairness([42.0], alpha=2.0) == pytest.approx(1.0)


def test_all_zeros_alpha_two_returns_one():
    """All zeros → Jain returns 1 by convention (no inequality among zero values)."""
    from ai.domain.fairness import alpha_fairness

    assert alpha_fairness([0, 0, 0], alpha=2.0) == pytest.approx(1.0)


def test_zero_value_alpha_one_does_not_nan():
    """log(0) clamped via EPSILON; result is finite (large negative)."""
    from ai.domain.fairness import EPSILON, alpha_fairness

    result = alpha_fairness([1.0, 0.0, 1.0], alpha=1.0)
    assert math.isfinite(result)
    expected = 2 * math.log(1.0) + math.log(EPSILON)
    assert result == pytest.approx(expected, abs=1e-9)


def test_zero_value_alpha_three_does_not_nan():
    """1/x clamped via EPSILON for α>1; finite result."""
    from ai.domain.fairness import alpha_fairness

    result = alpha_fairness([1.0, 0.0, 1.0], alpha=3.0)
    assert math.isfinite(result)


# === Vectorized parity ===


def test_alpha_fairness_batch_matches_scalar():
    """Batched torch impl matches scalar per row across α values."""
    from ai.domain.fairness import alpha_fairness, alpha_fairness_batch

    rows = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [50.0, 50.0, 50.0],
            [100.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    for alpha in (0.0, 1.0, 2.0, 3.0, float("inf")):
        batched = alpha_fairness_batch(rows, alpha=alpha).tolist()
        for i, row in enumerate(rows.tolist()):
            assert batched[i] == pytest.approx(
                alpha_fairness(row, alpha=alpha), abs=1e-9
            ), f"row {i} α={alpha} batched {batched[i]} scalar {alpha_fairness(row, alpha=alpha)}"


def test_unfairness_batch_matches_scalar():
    """Batched unfairness matches scalar aggregate_fairness(kind='unfairness')."""
    from ai.domain.fairness import aggregate_fairness, unfairness_batch

    rows = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [50.0, 50.0, 50.0],
            [60.0, 0.0, 0.0],   # max inequality at α=∞
        ],
        dtype=torch.float64,
    )
    for alpha in (2.0, float("inf")):
        batched = unfairness_batch(rows, alpha=alpha).tolist()
        for i, row in enumerate(rows.tolist()):
            assert batched[i] == pytest.approx(
                aggregate_fairness(row, alpha=alpha, kind="unfairness"), abs=1e-9
            )


# === Jain regression ===


def test_alpha_two_matches_legacy_jain_bit_identical():
    """At α=2, alpha_fairness ≡ jain_fairness_index within 1e-12 on representative hours."""
    from ai.domain.fairness import alpha_fairness
    from ai.domain.problem import jain_fairness_index

    hours_examples = [
        [],                                       # empty — degenerate Jain (1.0)
        [160, 160, 160, 160, 40, 40, 40],         # default canonical FT/PT mix
        [100, 50, 200, 75, 125, 30, 90],          # uneven
        [120, 120, 120, 120, 120, 120, 120],      # uniform
        [0, 0, 0, 0, 0, 0, 720],                  # extreme
    ]
    for hours in hours_examples:
        a = alpha_fairness(hours, alpha=2.0)
        j = jain_fairness_index(hours)
        assert a == pytest.approx(j, abs=1e-12), f"{hours}: α=2 gave {a}, jain gave {j}"


def test_accepts_list_ndarray_tensor():
    """Accepts list / ndarray / tensor uniformly."""
    from ai.domain.fairness import alpha_fairness

    values_list = [1.0, 2.0, 3.0]
    expected = alpha_fairness(values_list, alpha=2.0)
    assert alpha_fairness(np.array(values_list), alpha=2.0) == pytest.approx(expected, abs=1e-12)
    assert alpha_fairness(torch.tensor(values_list), alpha=2.0) == pytest.approx(expected, abs=1e-12)
