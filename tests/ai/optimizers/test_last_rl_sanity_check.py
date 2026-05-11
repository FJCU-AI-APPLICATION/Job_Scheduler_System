"""Sanity-check test for LAST-RL paper reproduction — issue #19 hard gate.

Marked @pytest.mark.slow @pytest.mark.benchmark. Skipped if the BCV-a.1
fixture is not present. Expected wall-clock: 30-60 minutes on a single thread.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "last_rl"
BCV_A1 = FIXTURE_DIR / "bcv-a.1.txt"


def test_paper_benchmark_within_10pct_of_published():
    """Train LAST-RL on bcv-a.1 for the paper's reported step count;
    assert achieved cost within ±10% of paper_target_cost.

    The issue's hard acceptance gate. Failure here means our LAST-RL port
    does NOT reproduce the paper and should not be considered to "work" yet.
    """
    if not BCV_A1.exists():
        pytest.skip(
            f"BCV-a.1 fixture not found at {BCV_A1}. "
            "Download from Kletzander's GitLab and place under tests/fixtures/last_rl/."
        )

    from ai.data.last_rl_benchmark import load_paper_instance
    from ai.optimizers.last_rl import SARSALambdaPolicy, train
    from ai.optimizers.last_rl_problem import PaperBenchmarkProblem
    from ai.optimizers.result import LastRLConfig

    instance = load_paper_instance(str(BCV_A1))
    problem = PaperBenchmarkProblem(instance)
    config = LastRLConfig(
        num_episodes=1000,
        episode_length=10000,
        alpha=0.1, gamma=0.99, lam=0.9,
        epsilon_start=0.5, epsilon_end=0.05,
        iht_size=4096, num_tilings=8,
        seed=42,
    )
    policy = SARSALambdaPolicy(
        iht_size=config.iht_size, num_tilings=config.num_tilings,
        num_actions=problem.num_actions,
    )
    episodes = train(problem, policy, config, np.random.default_rng(42))

    achieved = min(ep.best_cost for ep in episodes)
    target = instance.paper_target_cost
    rel_error = abs(achieved - target) / max(target, 1.0)
    assert rel_error <= 0.10, (
        f"LAST-RL sanity check FAILED: achieved={achieved:.1f}, "
        f"paper_target={target:.1f}, rel_error={rel_error:.1%} (>10%)"
    )
