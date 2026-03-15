from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class IdentityEnum(str, Enum):
    FULL = "FULL"
    PART = "PART"


class SalaryTypeEnum(str, Enum):
    MONTH = "MONTH"
    HOUR = "HOUR"


class UnavailabilityTypeEnum(str, Enum):
    DAY_OF_WEEK = "DAY_OF_WEEK"
    DATE_RANGE = "DATE_RANGE"


# --- Employee ---


class EmployeeCreate(BaseModel):
    name: str = Field(max_length=100)
    age: int = Field(ge=0)
    phone: str = Field(max_length=20)
    identity: IdentityEnum = IdentityEnum.FULL
    salary_type: SalaryTypeEnum = SalaryTypeEnum.MONTH


class EmployeeUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    age: Optional[int] = Field(None, ge=0)
    phone: Optional[str] = Field(None, max_length=20)
    identity: Optional[IdentityEnum] = None
    salary_type: Optional[SalaryTypeEnum] = None


class EmployeeResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    age: int
    phone: str
    identity: str
    salary_type: str
    insert_date: date
    update_date: date


# --- EmployeeUnavailability ---


class EmployeeUnavailabilityCreate(BaseModel):
    employee_id: int
    unavailability_type: UnavailabilityTypeEnum
    day_of_week: Optional[int] = Field(None, ge=0, le=7)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_type_fields(self) -> "EmployeeUnavailabilityCreate":
        if self.unavailability_type == UnavailabilityTypeEnum.DAY_OF_WEEK:
            if self.day_of_week is None:
                raise ValueError("day_of_week is required for DAY_OF_WEEK type")
            if self.start_date is not None or self.end_date is not None:
                raise ValueError("start_date/end_date must be null for DAY_OF_WEEK type")
        elif self.unavailability_type == UnavailabilityTypeEnum.DATE_RANGE:
            if self.start_date is None or self.end_date is None:
                raise ValueError("start_date and end_date are required for DATE_RANGE type")
            if self.day_of_week is not None:
                raise ValueError("day_of_week must be null for DATE_RANGE type")
            if self.end_date < self.start_date:
                raise ValueError("end_date must be >= start_date")
        return self


class EmployeeUnavailabilityUpdate(BaseModel):
    unavailability_type: Optional[UnavailabilityTypeEnum] = None
    day_of_week: Optional[int] = Field(None, ge=0, le=7)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    reason: Optional[str] = None


class EmployeeUnavailabilityResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    employee_id: int
    unavailability_type: str
    day_of_week: Optional[int]
    start_date: Optional[date]
    end_date: Optional[date]
    reason: str


# --- Aggregated views ---


class ShiftAssignmentDetail(BaseModel):
    schedule_name: str
    description: str
    schedule_date: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    assigned_date: str


class EmployeeFullDetail(BaseModel):
    employee: EmployeeResponse
    unavailability: list[EmployeeUnavailabilityResponse]
    shift_assignments: list[ShiftAssignmentDetail]
