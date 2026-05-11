"""LAST-RL paper benchmark loader — issue #19.

Parses one BCV-format text instance from Kletzander's GitLab. The format is
documented at https://www.dbai.tuwien.ac.at/staff/musliu/research/

Public API:
  PaperInstance               — dataclass with parsed fields
  load_paper_instance(path)   — parser entry point
  paper_cost(instance, sched) — exact paper weighted-sum cost

The BCV format parser (load_paper_instance) and exact cost (paper_cost) are
stubbed with NotImplementedError because they require consulting the paper's
GitLab format spec. They are guarded by the sanity-check test's
skip-if-fixture-missing pattern so this module can still ship as scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PaperInstance:
    """A LAST-RL paper benchmark instance.

    `paper_target_cost` is the paper's published mean cost for this instance
    after their reported number of training steps — used by the sanity-check
    test as the ±10% reference.
    """

    name: str
    num_employees: int
    num_days: int
    num_shift_types: int
    shift_lengths_hours: list[int]
    demand: np.ndarray                       # (num_days, num_shift_types)
    days_off: dict[int, set[int]]            # employee_idx → set of blocked day indices
    contract_min_hours: dict[int, int]
    contract_max_hours: dict[int, int]
    contract_max_consec: dict[int, int]
    weights: dict[str, float] = field(default_factory=dict)
    paper_target_cost: float = 0.0


def load_paper_instance(path: str | Path) -> PaperInstance:
    """Parse a BCV-format text file from Kletzander's LAST-RL GitLab.

    Stubbed — see module docstring for the rationale. When implementing,
    walk SECTION_<NAME> headers and parse CSV-style records inside.
    """
    raise NotImplementedError(
        f"BCV-format parser not yet implemented for {path}. "
        "See https://www.dbai.tuwien.ac.at/staff/musliu/research/ for the spec."
    )


def paper_cost(instance: PaperInstance, solution: list[int]) -> float:
    """The paper's exact weighted-sum cost on this instance.

    Stubbed; see paper §3.1 for the formula. Components:
      coverage + sequence + preference + contract violations, weighted by
      instance.weights['cover'|'seq'|'pref'|'contract'].
    """
    raise NotImplementedError(
        "paper_cost not yet implemented. See paper §3.1 for the cost formula."
    )
