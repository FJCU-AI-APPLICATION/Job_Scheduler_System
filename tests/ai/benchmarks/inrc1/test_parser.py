"""Tests for the INRC-I XML parser."""

from pathlib import Path

import pytest

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "fixtures"
    / "inrc1"
    / "sprint01.xml"
)


def test_parses_sprint01_fixture():
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    instance = parse_inrc1_instance(FIXTURE_PATH.read_text())

    assert instance.name == "sprint01"
    assert instance.num_nurses == 10
    assert instance.num_days == 28
    assert len(instance.shift_types) == 4
    assert len(instance.contracts) == 4


def test_parses_shift_durations_in_minutes():
    """E shift is 06:30→14:30 (8h); N shift is 22:30→06:30 next day (8h)."""
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    instance = parse_inrc1_instance(FIXTURE_PATH.read_text())
    by_id = {s.name: s for s in instance.shift_types}

    assert by_id["E"].duration_minutes == 8 * 60
    assert by_id["N"].duration_minutes == 8 * 60  # wraps to next day


def test_day_offs_are_zero_indexed_from_start_date():
    """Sprint01 StartDate is 2010-01-01, so a 2010-01-02 day-off has day=1."""
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    instance = parse_inrc1_instance(FIXTURE_PATH.read_text())

    assert any(
        req.nurse == "0" and req.day == 1 for req in instance.day_off_requests
    )
    assert all(0 <= req.day < instance.num_days for req in instance.day_off_requests)


def test_contracts_carry_assignment_bounds():
    """The full-time contract in sprint01 has Max=16, Min=9 assignments."""
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    instance = parse_inrc1_instance(FIXTURE_PATH.read_text())
    fulltime = next(c for c in instance.contracts if c.description == "fulltime")

    assert fulltime.max_assignments == 16
    assert fulltime.min_assignments == 9


def test_rejects_non_inrc1_xml():
    """Wrong root element should raise ValueError, not silently parse junk."""
    from ai.benchmarks.inrc1.parser import parse_inrc1_instance

    with pytest.raises(ValueError):
        parse_inrc1_instance("<NotAnInstance/>")
