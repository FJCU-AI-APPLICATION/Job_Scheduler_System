from pydantic import BaseModel


class EmployeeInfo(BaseModel):
    id: int
    employee_type: str  # "FT" or "PT"
    max_hours: int


class ShiftInfo(BaseModel):
    start_time: str
    end_time: str
    length_hours: int


class UnavailabilityInfo(BaseModel):
    employee_id: int
    day: int  # day index (0-based)


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
    shift_type_distribution: dict[int, dict[int, int]]  # employee_id -> {shift_idx: count}


class SchedulingResponse(BaseModel):
    schedule: list[ShiftAssignment]
    metrics: ScheduleMetrics


class GAFitnessResult(BaseModel):
    imbalance: float
    constraint_violations: float
    back_to_back: float


class GAConfigSnapshot(BaseModel):
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


class GATrainResult(BaseModel):
    schedule: list[int]
    fitness: GAFitnessResult
    pareto_front_size: int
    config: GAConfigSnapshot
