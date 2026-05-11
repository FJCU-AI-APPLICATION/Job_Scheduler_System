"""Tests for training/rl.py warm-start argparse guards."""

import json
import pytest


def test_warm_start_with_non_maskable_ppo_raises():
    """--warm-start cpsat with --algorithm dqn raises ValueError."""
    from ai.training.rl import train

    with pytest.raises(ValueError) as exc:
        train(
            algorithm="dqn",
            total_timesteps=10,
            warm_start="cpsat",
            warm_start_solutions=2,
            bc_steps=10,
        )
    assert "maskable_ppo" in str(exc.value)


def test_empty_pareto_ref_raises(tmp_path):
    """--pareto-ref pointing to a file with empty points raises ValueError."""
    from ai.training.rl import train

    empty_ref = tmp_path / "empty_pareto.json"
    empty_ref.write_text(
        json.dumps(
            {
                "algorithm": "ccmo",
                "fairness_alpha": 2.0,
                "hv_reference_point": [2.0, 1000.0, 100.0],
                "points": [],
                "schedules": [],
            }
        )
    )
    with pytest.raises(ValueError) as exc:
        train(
            algorithm="maskable_ppo",
            total_timesteps=10,
            pareto_ref=str(empty_ref),
        )
    assert "empty" in str(exc.value).lower()
