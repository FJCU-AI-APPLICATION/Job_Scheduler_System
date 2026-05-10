"""INRC-I instance parser.

The canonical INRC-I corpus distributes sprint instances as XML
(per Haspeslagh et al. 2014 and the official PATAT mirror), not the
KEYWORD={...} text format an earlier draft of the spec assumed. This
parser uses stdlib xml.etree to avoid a new dependency and exposes a
structured InrcInstance preserving the full INRC-I shape so the lossy
adapter to SchedulingProblem doesn't have to re-parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class InrcShiftType:
    """One <Shift> entry from <ShiftTypes>."""

    name: str
    start_time: str
    end_time: str
    duration_minutes: int


@dataclass(frozen=True)
class InrcContract:
    """One <Contract> entry from <Contracts>. Most fields dropped — only the
    counts the lossy adapter consumes are retained.
    """

    name: str
    description: str
    max_assignments: int
    min_assignments: int


@dataclass(frozen=True)
class InrcNurse:
    """One <Employee> entry from <Employees>."""

    name: str
    contract: str
    skills: tuple[str, ...]


@dataclass(frozen=True)
class InrcDayOffRequest:
    """One <DayOff> entry from <DayOffRequests>. `day` is a 0-indexed offset
    from the scheduling period's StartDate.
    """

    nurse: str
    day: int


@dataclass(frozen=True)
class InrcInstance:
    """Structured representation of a parsed INRC-I instance file."""

    name: str
    num_nurses: int
    num_days: int
    shift_types: tuple[InrcShiftType, ...]
    contracts: tuple[InrcContract, ...]
    nurses: tuple[InrcNurse, ...]
    day_off_requests: tuple[InrcDayOffRequest, ...]


def parse_inrc1_instance(text: str) -> InrcInstance:
    """Parse an INRC-I XML instance string into an InrcInstance."""
    root = ET.fromstring(text)
    if root.tag != "SchedulingPeriod":
        raise ValueError(
            f"Expected root element 'SchedulingPeriod', got '{root.tag}'"
        )

    name = root.attrib.get("ID", "unknown")
    start_date = _parse_date(_text(root, "StartDate"))
    end_date = _parse_date(_text(root, "EndDate"))
    num_days = (end_date - start_date).days + 1

    shift_types = tuple(_parse_shift_types(root.find("ShiftTypes")))
    contracts = tuple(_parse_contracts(root.find("Contracts")))
    nurses = tuple(_parse_nurses(root.find("Employees")))
    day_off_requests = tuple(
        _parse_day_offs(root.find("DayOffRequests"), start_date)
    )

    return InrcInstance(
        name=name,
        num_nurses=len(nurses),
        num_days=num_days,
        shift_types=shift_types,
        contracts=contracts,
        nurses=nurses,
        day_off_requests=day_off_requests,
    )


def _text(elem: ET.Element, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_time(s: str) -> time:
    return datetime.strptime(s, "%H:%M:%S").time()


def _shift_duration_minutes(start: str, end: str) -> int:
    s = _parse_time(start)
    e = _parse_time(end)
    start_min = s.hour * 60 + s.minute
    end_min = e.hour * 60 + e.minute
    delta = end_min - start_min
    if delta <= 0:
        delta += 24 * 60
    return delta


def _parse_shift_types(elem: ET.Element | None):
    if elem is None:
        return
    for shift in elem.findall("Shift"):
        start = _text(shift, "StartTime")
        end = _text(shift, "EndTime")
        if not start or not end:
            continue
        yield InrcShiftType(
            name=shift.attrib.get("ID", _text(shift, "Description") or "?"),
            start_time=start,
            end_time=end,
            duration_minutes=_shift_duration_minutes(start, end),
        )


def _parse_contracts(elem: ET.Element | None):
    if elem is None:
        return
    for contract in elem.findall("Contract"):
        cid = contract.attrib.get("ID", "")
        try:
            max_assign = int(_text(contract, "MaxNumAssignments", "0"))
            min_assign = int(_text(contract, "MinNumAssignments", "0"))
        except ValueError:
            continue
        yield InrcContract(
            name=cid,
            description=_text(contract, "Description"),
            max_assignments=max_assign,
            min_assignments=min_assign,
        )


def _parse_nurses(elem: ET.Element | None):
    if elem is None:
        return
    for emp in elem.findall("Employee"):
        skills_elem = emp.find("Skills")
        skills = (
            tuple(
                s.text.strip()
                for s in skills_elem.findall("Skill")
                if s.text is not None
            )
            if skills_elem is not None
            else ()
        )
        yield InrcNurse(
            name=emp.attrib.get("ID", _text(emp, "Name")),
            contract=_text(emp, "ContractID"),
            skills=skills,
        )


def _parse_day_offs(elem: ET.Element | None, start_date: date):
    if elem is None:
        return
    for dayoff in elem.findall("DayOff"):
        nurse = _text(dayoff, "EmployeeID")
        date_str = _text(dayoff, "Date")
        if not nurse or not date_str:
            continue
        try:
            d = _parse_date(date_str)
        except ValueError:
            continue
        yield InrcDayOffRequest(nurse=nurse, day=(d - start_date).days)
