"""End-to-end smoke test for training/rl.py --warm-start cpsat."""

import json

import pytest


@pytest.mark.slow
def test_warm_start_cpsat_runs_end_to_end(tmp_path):
    """python -m ai.training.rl --algorithm maskable_ppo --warm-start cpsat
    --warm-start-solutions 2 --bc-steps 30 --total-timesteps 100 finishes
    and writes a metadata file with the warm_start config."""
    from ai.training.rl import train

    train(
        algorithm="maskable_ppo",
        total_timesteps=100,
        learning_rate=3e-4,
        checkpoint_dir=str(tmp_path),
        tb_log_dir=str(tmp_path / "tb"),
        eval_freq=200,
        checkpoint_freq=200,
        net_arch=[16],
        fairness_alpha=2.0,
        warm_start="cpsat",
        warm_start_solutions=2,
        bc_steps=30,
        bc_lr=1e-3,
        pareto_ref=None,
    )

    metadata_path = tmp_path / "model_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["warm_start"]["type"] == "cpsat"
    assert metadata["warm_start"]["n_solutions"] == 2
    assert metadata["warm_start"]["bc_steps"] == 30
