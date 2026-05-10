from pydantic import BaseModel


class EmployeeInfo(BaseModel):
    id: int
    employee_type: str
    max_hours: int


class ShiftInfo(BaseModel):
    start_time: str
    end_time: str
    length_hours: int


class UnavailabilityInfo(BaseModel):
    employee_id: int
    day: int


class SchedulingRequest(BaseModel):
    employees: list[EmployeeInfo]
    shifts: list[ShiftInfo]
    days: int
    unavailability: list[UnavailabilityInfo] = []


class ShiftAssignment(BaseModel):
    day: int
    shift_index: int
    employee_id: int


class ConstraintViolations(BaseModel):
    unavailability_violations: int
    max_hours_violations: int
    total_violations: int


class ScheduleMetrics(BaseModel):
    fairness_score: float
    jain_fairness_index: float
    total_hours_by_employee: dict[int, int]
    constraint_violations: ConstraintViolations
    back_to_back_rate: float
    coverage_rate: float
    shift_type_distribution: dict[int, dict[int, int]]


class SchedulingResponse(BaseModel):
    schedule: list[ShiftAssignment]
    metrics: ScheduleMetrics


# === NSGA-II (renamed from GA*) ===

class NSGAIIFitnessResult(BaseModel):
    imbalance: float
    constraint_violations: float
    back_to_back: float


class NSGAIIConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    generations: int
    pop_size: int
    cxpb: float
    mutpb: float
    indpb: float
    tournament_size: int
    elitist: bool
    seed: int | None
    device: str


class NSGAIITrainResult(BaseModel):
    schedule: list[int]
    fitness: NSGAIIFitnessResult
    pareto_front_size: int
    config: NSGAIIConfigSnapshot


# === CCMO ===

class CCMOFitnessResult(BaseModel):
    imbalance: float
    constraint_violations: float
    back_to_back: float


class CCMOConfigSnapshot(BaseModel):
    num_employees: int
    employee_types: list[str]
    days: int
    shifts_per_day: int
    shift_lengths: list[int]
    generations: int
    pop_size: int
    cxpb: float
    mutpb: float
    indpb: float
    tournament_size: int
    seed: int | None
    device: str


class CCMOTrainResult(BaseModel):
    schedule: list[int]
    fitness: CCMOFitnessResult
    feasible_front_size: int
    auxiliary_front_size: int
    fell_back_to_auxiliary: bool
    config: CCMOConfigSnapshot


# === Benchmark ===

class BenchmarkRunRecord(BaseModel):
    instance: str
    algorithm: str
    seed: int
    hypervolume: float
    feasible_front_size: int
    best_imbalance: float
    best_violations: float
    best_b2b: int
    wall_clock_s: float


class BenchmarkAggregate(BaseModel):
    instance: str
    nsga2_hv_mean: float | None = None
    nsga2_hv_std: float | None = None
    nsga2_n_seeds: int = 0
    ccmo_hv_mean: float | None = None
    ccmo_hv_std: float | None = None
    ccmo_n_seeds: int = 0
    wilcoxon_p: float | None = None


class BenchmarkReport(BaseModel):
    config_summary: dict
    per_run: list[BenchmarkRunRecord]
    aggregate: list[BenchmarkAggregate]


# === One-release deprecation aliases ===

import warnings


class GAFitnessResult(NSGAIIFitnessResult):
    """Deprecated alias for NSGAIIFitnessResult."""

    def __init__(self, **data):
        warnings.warn(
            "GAFitnessResult is deprecated; use NSGAIIFitnessResult",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class GAConfigSnapshot(NSGAIIConfigSnapshot):
    """Deprecated alias for NSGAIIConfigSnapshot."""

    def __init__(self, **data):
        warnings.warn(
            "GAConfigSnapshot is deprecated; use NSGAIIConfigSnapshot",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class GATrainResult(NSGAIITrainResult):
    """Deprecated alias for NSGAIITrainResult."""

    def __init__(self, **data):
        warnings.warn(
            "GATrainResult is deprecated; use NSGAIITrainResult",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
