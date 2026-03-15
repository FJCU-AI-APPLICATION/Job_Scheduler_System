from datetime import datetime, time
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ScheduleCreate(BaseModel):
    name: str = Field(max_length=200)
    description: str
    start_date: datetime
    end_date: datetime
    start_time: Optional[time] = None
    end_time: Optional[time] = None

    @model_validator(mode="after")
    def validate_dates(self) -> "ScheduleCreate":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        return self


class ScheduleUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None


class ScheduleResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    start_time: Optional[time]
    end_time: Optional[time]


# --- Compute / Confirm ---


class ComputeScheduleRequest(BaseModel):
    policy_id: int
    employee_ids: list[int] = Field(min_length=1)
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD


class ShiftAssignmentSchema(BaseModel):
    start_time: str
    end_time: str
    assigned_employees: list[int]


class DayScheduleSchema(BaseModel):
    date: str
    day_of_week: int
    shift_assignments: list[ShiftAssignmentSchema]
    unavailable_employees: list[int]


class ComputedScheduleResponse(BaseModel):
    policy_id: int
    start_date: str
    end_date: str
    shift_details: list[dict]
    schedule: list[DayScheduleSchema]


class ConfirmScheduleRequest(BaseModel):
    policy_id: int
    start_date: str
    end_date: str
    schedule: list[DayScheduleSchema]
