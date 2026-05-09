from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.domain.employee import EmployeeUnavailability
from backend.domain.policy import ShiftPolicy
from backend.domain.schedule import Schedule, ScheduleEmployee
from backend.schemas.schedule import ComputeScheduleRequest, ConfirmScheduleRequest
from backend.services.algorithms import (
    ComputedSchedule,
    DayAvailability,
    ScheduleInput,
    ShiftInfo,
    process_schedule,
)


async def compute_schedule(
    request: ComputeScheduleRequest,
    db: AsyncSession,
) -> ComputedSchedule:
    """Query shift policies and employee availability, then compute a schedule."""
    stmt = select(ShiftPolicy).where(ShiftPolicy.policy_id == request.policy_id)
    result = await db.execute(stmt)
    shift_policies = result.scalars().all()

    if not shift_policies:
        raise ValueError(f"No shift policies found for policy_id={request.policy_id}")

    shift_info = [
        ShiftInfo(
            start_time=sp.start_time.strftime("%H:%M:%S"),
            end_time=sp.end_time.strftime("%H:%M:%S"),
        )
        for sp in shift_policies
    ]

    stmt = select(EmployeeUnavailability).where(
        EmployeeUnavailability.employee_id.in_(request.employee_ids)
    )
    result = await db.execute(stmt)
    all_unavailabilities = result.scalars().all()

    day_of_week_unavail: dict[int, set[int]] = {}
    date_range_unavail: list[EmployeeUnavailability] = []

    for u in all_unavailabilities:
        if u.unavailability_type == "DAY_OF_WEEK" and u.day_of_week is not None:
            day_of_week_unavail.setdefault(u.day_of_week, set()).add(u.employee_id)
        elif u.unavailability_type == "DATE_RANGE":
            date_range_unavail.append(u)

    start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    daily_availability: list[DayAvailability] = []

    current = start
    while current <= end:
        dow = current.isoweekday()
        unavailable: set[int] = set()

        if dow in day_of_week_unavail:
            unavailable.update(day_of_week_unavail[dow])

        for u in date_range_unavail:
            if u.start_date and u.end_date and u.start_date <= current <= u.end_date:
                unavailable.add(u.employee_id)

        available = [eid for eid in request.employee_ids if eid not in unavailable]

        daily_availability.append(
            DayAvailability(
                date=current.strftime("%Y-%m-%d"),
                day_of_week=dow,
                available_employees=available,
                unavailable_employees=list(unavailable),
            )
        )
        current += timedelta(days=1)

    schedule_input = ScheduleInput(
        policy_id=request.policy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        shift_info=shift_info,
        daily_availability=daily_availability,
    )

    return process_schedule(schedule_input)


async def confirm_schedule(
    request: ConfirmScheduleRequest,
    db: AsyncSession,
) -> list[int]:
    """Persist a computed schedule. Returns created schedule IDs."""
    created_ids: list[int] = []

    for day in request.schedule:
        for shift in day.shift_assignments:
            schedule_record = Schedule(
                name=f"Schedule {day.date}",
                description=f"Policy {request.policy_id} - {day.date}",
                start_date=datetime.strptime(f"{day.date} {shift.start_time}", "%Y-%m-%d %H:%M:%S"),
                end_date=datetime.strptime(f"{day.date} {shift.end_time}", "%Y-%m-%d %H:%M:%S"),
                start_time=datetime.strptime(shift.start_time, "%H:%M:%S").time(),
                end_time=datetime.strptime(shift.end_time, "%H:%M:%S").time(),
            )
            db.add(schedule_record)
            await db.flush()

            for emp_id in shift.assigned_employees:
                assignment = ScheduleEmployee(
                    schedule_id=schedule_record.id,
                    employee_id=emp_id,
                )
                db.add(assignment)

            created_ids.append(schedule_record.id)

    await db.flush()
    return created_ids
