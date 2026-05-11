"""Tests for the <algo>_pareto_front.json export from training/evolutionary.py."""

import json


def test_ccmo_pareto_front_written(tmp_path):
    """After CCMO training, <output_dir>/ccmo_pareto_front.json exists and parses."""
    from ai.training.evolutionary import train_evolutionary

    train_evolutionary(
        algorithm="ccmo",
        generations=5,
        pop_size=20,
        cxpb=0.7,
        mutpb=0.2,
        indpb=0.05,
        seed=42,
        device="cpu",
        output_dir=str(tmp_path),
        fairness_alpha=2.0,
    )
    path = tmp_path / "ccmo_pareto_front.json"
    assert path.exists(), f"Expected {path} to exist"

    payload = json.loads(path.read_text())
    assert payload["algorithm"] == "ccmo"
    assert payload["fairness_alpha"] == 2.0
    assert payload["hv_reference_point"] == [2.0, 1000.0, 100.0]
    assert "points" in payload
    assert "schedules" in payload
    assert len(payload["points"]) == len(payload["schedules"])
    # All points must be feasible (violations == 0.0).
    for p in payload["points"]:
        assert len(p) == 3, f"point must be (unfairness, violations, b2b), got {p}"
        assert p[1] == 0.0, f"point {p} has non-zero violations"


def test_nsga2_pareto_front_written(tmp_path):
    """NSGA-II export filters pareto_front to feasible-only (violations == 0)."""
    from ai.training.evolutionary import train_evolutionary

    train_evolutionary(
        algorithm="nsga2",
        generations=5,
        pop_size=20,
        cxpb=0.7,
        mutpb=0.2,
        indpb=0.05,
        seed=42,
        device="cpu",
        output_dir=str(tmp_path),
        fairness_alpha=2.0,
    )
    path = tmp_path / "nsga2_pareto_front.json"
    assert path.exists()

    payload = json.loads(path.read_text())
    for p in payload["points"]:
        assert p[1] == 0.0, f"NSGA-II export must filter to feasible-only; got point {p}"
