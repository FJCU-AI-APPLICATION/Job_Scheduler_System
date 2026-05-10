"""INRC-I → SchedulingProblem lossy adapter.

Maps a parsed InrcInstance to our SchedulingProblem domain model. Drops:
  - Nurse skills/grades
  - Contract patterns
  - Weekend rotation constraints
  - Shift-type sequencing constraints
  - Soft-constraint weights from the original INRC-I scoring

Mapped:
  - num_nurses → num_employees
  - StartDate→EndDate inclusive → days
  - shift_types (count) → shifts_per_day
  - per-shift duration (minutes → hours) → shift_lengths
  - Contract.max_assignments × average shift hours → max_hours per nurse
  - DayOffRequests → unavailability
"""

from __future__ import annotations

import json
from pathlib import Path

from ai.benchmarks.inrc1.parser import InrcInstance, parse_inrc1_instance
from ai.domain.problem import SchedulingProblem

DATA_DIR = Path("data/benchmarks/inrc1")
MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def list_instances(track: str = "sprint") -> list[str]:
    """Return ordered instance names for a track from the manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    return [item["name"] for item in manifest.get(track, [])]


def load_instance(name: str) -> SchedulingProblem:
    """Load an INRC-I instance by name as a SchedulingProblem (lossy)."""
    path = DATA_DIR / f"{name}.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"INRC-I instance '{name}' not found at {path}. "
            "Run `python scripts/fetch_inrc1.py` to download the corpus."
        )
    instance = parse_inrc1_instance(path.read_text())
    return _to_scheduling_problem(instance)


def _to_scheduling_problem(instance: InrcInstance) -> SchedulingProblem:
    num_employees = instance.num_nurses
    days = instance.num_days
    shift_types = instance.shift_types
    shifts_per_day = len(shift_types)

    # SchedulingProblem expects integer hours per shift; round duration to nearest hour.
    shift_lengths = tuple(round(s.duration_minutes / 60) for s in shift_types)
    avg_shift_hours = (
        sum(s.duration_minutes for s in shift_types) / max(len(shift_types), 1)
    ) / 60

    contract_by_name = {c.name: c for c in instance.contracts}
    max_hours_list: list[int] = []
    employee_types_list: list[str] = []
    for nurse in instance.nurses:
        contract = contract_by_name.get(nurse.contract)
        if contract is None:
            max_hours_list.append(int(days * 8))
            employee_types_list.append("FT")
        else:
            max_hours_list.append(int(contract.max_assignments * avg_shift_hours))
            employee_types_list.append(
                "FT" if contract.max_assignments >= days * 0.5 else "PT"
            )

    nurse_idx = {n.name: i for i, n in enumerate(instance.nurses)}
    unavailability = frozenset(
        (req.day, nurse_idx[req.nurse])
        for req in instance.day_off_requests
        if req.nurse in nurse_idx and 0 <= req.day < days
    )

    return SchedulingProblem(
        num_employees=num_employees,
        employee_types=tuple(employee_types_list),
        days=days,
        shifts_per_day=shifts_per_day,
        shift_lengths=shift_lengths,
        max_hours=tuple(max_hours_list),
        unavailability=unavailability,
    )
