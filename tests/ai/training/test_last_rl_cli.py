"""End-to-end CLI test for python -m ai.training.last_rl."""

import json


def test_train_ours_cli_roundtrip(tmp_path):
    """train-ours writes a .npz; the JSON snapshot inside reloads cleanly."""
    from ai.domain.schemas import LastRLConfigSnapshot
    from ai.training.last_rl import main

    rc = main(
        [
            "train-ours",
            "--num-episodes", "2",
            "--episode-length", "10",
            "--ip-time-budget-s", "0.5",
            "--ip-workers", "1",
            "--seed", "42",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0

    ckpt = tmp_path / "last_rl_policy.npz"
    assert ckpt.exists()

    import numpy as np
    data = np.load(str(ckpt), allow_pickle=True)
    snapshot_json = str(data["config_snapshot"][0])
    snap = LastRLConfigSnapshot(**json.loads(snapshot_json))
    assert snap.num_episodes == 2
    assert snap.episode_length == 10
    assert snap.fairness_alpha == float("inf")


def test_train_ours_save_trajectory(tmp_path):
    """--save-trajectory writes a sibling .episodes.json parseable as a list."""
    from ai.optimizers.result import LastRLEpisodeStatus
    from ai.training.last_rl import main

    rc = main(
        [
            "train-ours",
            "--num-episodes", "2",
            "--episode-length", "10",
            "--ip-time-budget-s", "0.5",
            "--ip-workers", "1",
            "--seed", "42",
            "--output-dir", str(tmp_path),
            "--save-trajectory",
        ]
    )
    assert rc == 0

    traj_path = tmp_path / "last_rl_policy.episodes.json"
    assert traj_path.exists()
    episodes = json.loads(traj_path.read_text())
    assert isinstance(episodes, list)
    assert len(episodes) == 2
    for e in episodes:
        LastRLEpisodeStatus(**e)


def test_train_paper_missing_fixture_skips_cleanly(tmp_path):
    """train-paper with a non-existent instance file errors cleanly (rc != 0)."""
    from ai.training.last_rl import main

    rc = main(
        [
            "train-paper",
            "--instance-path", "/nonexistent/bcv.txt",
            "--num-episodes", "1",
            "--episode-length", "5",
            "--seed", "0",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc != 0
