"""INRC-I lossy adapter tests."""

import shutil
from pathlib import Path

import pytest

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "fixtures"
    / "inrc1"
    / "sprint01.xml"
)


@pytest.fixture
def stub_inrc1_data(tmp_path, monkeypatch):
    """Mirror the bundled sprint01 fixture into a tmp data dir for the loader."""
    target_dir = tmp_path / "data" / "benchmarks" / "inrc1"
    target_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_PATH, target_dir / "sprint01.xml")
    monkeypatch.setattr("ai.benchmarks.inrc1.loader.DATA_DIR", target_dir)
    return target_dir


def test_loads_as_scheduling_problem(stub_inrc1_data):
    from ai.benchmarks.inrc1.loader import load_instance

    sp = load_instance("sprint01")

    assert sp.num_employees == 10
    assert sp.days == 28
    assert sp.shifts_per_day == 4
    assert len(sp.shift_lengths) == 4
    assert all(length == 8 for length in sp.shift_lengths)
    assert sp.num_shifts == 28 * 4


def test_max_hours_reflect_contract_assignments(stub_inrc1_data):
    """Per-nurse max_hours derives from contract.max_assignments × avg shift hours."""
    from ai.benchmarks.inrc1.loader import load_instance

    sp = load_instance("sprint01")

    # Sprint01 contracts: fulltime=16, 75=12, 50=9, night=8 assignments.
    # Avg shift duration = 8h, so max_hours ∈ {128, 96, 72, 64}.
    assert set(sp.max_hours) <= {128, 96, 72, 64}


def test_unavailability_populated(stub_inrc1_data):
    """DayOffRequests should produce some unavailability entries."""
    from ai.benchmarks.inrc1.loader import load_instance

    sp = load_instance("sprint01")
    assert len(sp.unavailability) > 0
    for day, emp in sp.unavailability:
        assert 0 <= day < sp.days
        assert 0 <= emp < sp.num_employees


def test_load_missing_raises(tmp_path, monkeypatch):
    from ai.benchmarks.inrc1.loader import load_instance

    monkeypatch.setattr(
        "ai.benchmarks.inrc1.loader.DATA_DIR", tmp_path / "nope"
    )
    with pytest.raises(FileNotFoundError) as exc:
        load_instance("sprint01")

    assert "fetch_inrc1" in str(exc.value)


def test_list_instances():
    from ai.benchmarks.inrc1.loader import list_instances

    names = list_instances("sprint")

    assert names == [f"sprint{i:02d}" for i in range(1, 11)]


def test_list_instances_unknown_track_returns_empty():
    from ai.benchmarks.inrc1.loader import list_instances

    assert list_instances("medium") == []
