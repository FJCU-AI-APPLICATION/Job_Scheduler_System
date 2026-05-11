"""Paper-benchmark loader tests for LAST-RL — issue #19.

Marked @pytest.mark.benchmark; skipped if the BCV-a.1 fixture is not present
under tests/fixtures/last_rl/. Download from Kletzander's GitLab.
"""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "last_rl"
BCV_A1 = FIXTURE_DIR / "bcv-a.1.txt"


pytestmark = pytest.mark.benchmark


def _require_fixture() -> Path:
    if not BCV_A1.exists():
        pytest.skip(
            f"BCV-a.1 fixture not found at {BCV_A1}. "
            "Download from Kletzander's GitLab and place under tests/fixtures/last_rl/."
        )
    return BCV_A1


def test_load_bcv_a1_shape():
    """Loaded PaperInstance has expected dimensions + paper target cost."""
    from ai.data.last_rl_benchmark import load_paper_instance

    instance = load_paper_instance(str(_require_fixture()))
    assert instance.name.startswith("bcv-a")
    assert instance.num_employees > 0
    assert instance.num_days > 0
    assert instance.paper_target_cost > 0


def test_paper_problem_implements_protocol():
    """PaperBenchmarkProblem exposes all LastRLProblem methods + 6-LLH library."""
    import numpy as np

    from ai.data.last_rl_benchmark import load_paper_instance
    from ai.optimizers.last_rl_problem import PaperBenchmarkProblem

    instance = load_paper_instance(str(_require_fixture()))
    problem = PaperBenchmarkProblem(instance)

    assert problem.num_actions == 6
    assert problem.name.startswith("paper:")
    rng = np.random.default_rng(0)
    sched = problem.initial_solution(rng)
    assert len(sched) == instance.num_days * instance.num_shift_types
    c = problem.cost(sched)
    assert c >= 0.0
    lib = problem.llh_library()
    assert len(lib) == 6
    assert len({h.name for h in lib}) == 6
