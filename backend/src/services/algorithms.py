from dataclasses import dataclass, field


@dataclass
class ShiftInfo:
    start_time: str
    end_time: str


@dataclass
class DayAvailability:
    date: str
    day_of_week: int
    available_employees: list[int]
    unavailable_employees: list[int]


@dataclass
class ScheduleInput:
    policy_id: int
    start_date: str
    end_date: str
    shift_info: list[ShiftInfo]
    daily_availability: list[DayAvailability]


@dataclass
class ShiftAssignment:
    start_time: str
    end_time: str
    assigned_employees: list[int]


@dataclass
class DaySchedule:
    date: str
    day_of_week: int
    shift_assignments: list[ShiftAssignment] = field(default_factory=list)
    unavailable_employees: list[int] = field(default_factory=list)


@dataclass
class ComputedSchedule:
    policy_id: int
    start_date: str
    end_date: str
    shift_details: list[ShiftInfo]
    schedule: list[DaySchedule] = field(default_factory=list)


def process_schedule(data: ScheduleInput) -> ComputedSchedule:
    """For each day, assign all available employees to each shift."""
    computed = ComputedSchedule(
        policy_id=data.policy_id,
        start_date=data.start_date,
        end_date=data.end_date,
        shift_details=data.shift_info,
    )

    for day in data.daily_availability:
        day_schedule = DaySchedule(
            date=day.date,
            day_of_week=day.day_of_week,
            unavailable_employees=day.unavailable_employees,
        )
        for shift in data.shift_info:
            day_schedule.shift_assignments.append(
                ShiftAssignment(
                    start_time=shift.start_time,
                    end_time=shift.end_time,
                    assigned_employees=list(day.available_employees),
                )
            )
        computed.schedule.append(day_schedule)

    return computed
