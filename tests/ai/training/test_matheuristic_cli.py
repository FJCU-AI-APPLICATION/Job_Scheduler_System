"""End-to-end test for `python -m ai.training.matheuristic` CLI.

Verifies the CLI roundtrip: invoke → JSON checkpoint written → reloadable via
the Pydantic schema with all defaults preserved. Catches mismatches between
argparse defaults and MatheuristicConfig defaults (which are duplicated in
two places and could drift silently).
"""

from __future__ import annotations

import json


def test_matheuristic_cli_roundtrip(tmp_path):
    """CLI run → JSON written → reload via MatheuristicTrainResult validates."""
    from ai.domain.schemas import MatheuristicTrainResult
    from ai.training.matheuristic import main

    rc = main(
        [
            "--max-iterations", "2",
            "--stagnation-limit", "2",
            "--time-budget-s", "30",
            "--inner-ip-time-budget-s", "1.5",
            "--inner-ip-workers", "1",
            "--seed", "42",
            "--output-dir", str(tmp_path),
        ]
    )
    assert rc == 0

    out_path = tmp_path / "matheuristic_best_schedule.json"
    assert out_path.exists()

    # Pydantic reload — catches schema drift / type mismatches at the boundary
    payload = json.loads(out_path.read_text())
    reloaded = MatheuristicTrainResult(**payload)
    assert reloaded.schedule
    assert reloaded.termination_reason in {"stagnation", "max_iterations", "time_budget"}
    assert reloaded.config.acceptance == "vns"
    assert reloaded.config.fairness_alpha == float("inf")

    # Trajectory file should NOT exist without --save-trajectory
    assert not (tmp_path / "matheuristic_trajectory.json").exists()


def test_matheuristic_cli_save_trajectory(tmp_path):
    """--save-trajectory writes a list of step dicts alongside the main JSON."""
    from ai.optimizers.result import MatheuristicStepStatus
    from ai.training.matheuristic import main

    rc = main(
        [
            "--max-iterations", "2",
            "--stagnation-limit", "2",
            "--time-budget-s", "30",
            "--inner-ip-time-budget-s", "1.5",
            "--inner-ip-workers", "1",
            "--seed", "42",
            "--output-dir", str(tmp_path),
            "--save-trajectory",
        ]
    )
    assert rc == 0

    traj_path = tmp_path / "matheuristic_trajectory.json"
    assert traj_path.exists()

    steps = json.loads(traj_path.read_text())
    assert isinstance(steps, list)
    assert len(steps) > 0
    # Each entry must validate as MatheuristicStepStatus (catches field drift)
    for s in steps:
        MatheuristicStepStatus(**s)
