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


class ScheduleMetrics(BaseModel):
    fairness_score: float
    total_hours_by_employee: dict[int, int]


class SchedulingResponse(BaseModel):
    schedule: list[ShiftAssignment]
    metrics: ScheduleMetrics
