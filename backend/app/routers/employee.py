from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.employee import Employee, EmployeeUnavailability
from app.models.schedule import Schedule, ScheduleEmployee
from app.schemas.common import PaginatedResponse
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeFullDetail,
    EmployeeResponse,
    EmployeeUnavailabilityCreate,
    EmployeeUnavailabilityResponse,
    EmployeeUnavailabilityUpdate,
    EmployeeUpdate,
    ShiftAssignmentDetail,
)

router = APIRouter(prefix="/api/employee", tags=["employee"])


# --- Employee CRUD ---


@router.get("/", response_model=PaginatedResponse[EmployeeResponse])
async def list_employees(
    page: int = Query(1, ge=1),
    page_size: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    count_stmt = select(func.count()).select_from(Employee)
    total = (await db.execute(count_stmt)).scalar() or 0

    offset = (page - 1) * page_size
    stmt = select(Employee).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    employees = result.scalars().all()

    return PaginatedResponse(
        count=total,
        page=page,
        page_size=page_size,
        results=employees,
    )


@router.post("/", response_model=EmployeeResponse, status_code=201)
async def create_employee(
    payload: EmployeeCreate,
    db: AsyncSession = Depends(get_db),
):
    employee = Employee(**payload.model_dump())
    db.add(employee)
    await db.flush()
    await db.refresh(employee)
    return employee


@router.get("/{employee_id}/", response_model=EmployeeResponse)
async def get_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    employee = await db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee


@router.put("/{employee_id}/", response_model=EmployeeResponse)
async def update_employee(
    employee_id: int,
    payload: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
):
    employee = await db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(employee, key, value)

    await db.flush()
    await db.refresh(employee)
    return employee


@router.delete("/{employee_id}/", status_code=204)
async def delete_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    employee = await db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    await db.delete(employee)


# --- EmployeeUnavailability CRUD ---


@router.get("/unavailabilities/", response_model=list[EmployeeUnavailabilityResponse])
async def list_unavailabilities(
    employee_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(EmployeeUnavailability).where(
        EmployeeUnavailability.employee_id == employee_id
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/unavailabilities/", response_model=EmployeeUnavailabilityResponse, status_code=201)
async def create_unavailability(
    payload: EmployeeUnavailabilityCreate,
    db: AsyncSession = Depends(get_db),
):
    # Verify employee exists
    employee = await db.get(Employee, payload.employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    unavailability = EmployeeUnavailability(**payload.model_dump())
    db.add(unavailability)
    await db.flush()
    await db.refresh(unavailability)
    return unavailability


@router.get("/unavailabilities/{unavailability_id}/", response_model=EmployeeUnavailabilityResponse)
async def get_unavailability(unavailability_id: int, db: AsyncSession = Depends(get_db)):
    unavailability = await db.get(EmployeeUnavailability, unavailability_id)
    if not unavailability:
        raise HTTPException(status_code=404, detail="Unavailability not found")
    return unavailability


@router.put("/unavailabilities/{unavailability_id}/", response_model=EmployeeUnavailabilityResponse)
async def update_unavailability(
    unavailability_id: int,
    payload: EmployeeUnavailabilityUpdate,
    db: AsyncSession = Depends(get_db),
):
    unavailability = await db.get(EmployeeUnavailability, unavailability_id)
    if not unavailability:
        raise HTTPException(status_code=404, detail="Unavailability not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(unavailability, key, value)

    await db.flush()
    await db.refresh(unavailability)
    return unavailability


@router.delete("/unavailabilities/{unavailability_id}/", status_code=204)
async def delete_unavailability(unavailability_id: int, db: AsyncSession = Depends(get_db)):
    unavailability = await db.get(EmployeeUnavailability, unavailability_id)
    if not unavailability:
        raise HTTPException(status_code=404, detail="Unavailability not found")
    await db.delete(unavailability)


# --- Employee Shifts ---


@router.get("/employee_shifts/")
async def employee_shifts(
    employee_id: int = Query(...),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(ScheduleEmployee)
        .where(ScheduleEmployee.employee_id == employee_id)
        .options(selectinload(ScheduleEmployee.schedule))
    )

    if start_date:
        stmt = stmt.join(Schedule).where(Schedule.start_date >= start_date)
    if end_date:
        stmt = stmt.join(Schedule, isouter=True).where(Schedule.start_date <= end_date)

    result = await db.execute(stmt)
    assignments = result.scalars().all()

    return [
        {
            "schedule_name": a.schedule.name,
            "description": a.schedule.description,
            "schedule_date": a.schedule.start_date.strftime("%Y-%m-%d"),
            "start_time": a.schedule.start_time.strftime("%H:%M:%S") if a.schedule.start_time else None,
            "end_time": a.schedule.end_time.strftime("%H:%M:%S") if a.schedule.end_time else None,
            "assigned_date": a.assigned_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for a in assignments
    ]


# --- Employee Full Detail ---


@router.get("/full_detail/", response_model=EmployeeFullDetail)
async def employee_full_detail(
    employee_id: int = Query(...),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    employee = await db.get(Employee, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Unavailabilities
    stmt = select(EmployeeUnavailability).where(
        EmployeeUnavailability.employee_id == employee_id
    )
    result = await db.execute(stmt)
    unavailabilities = result.scalars().all()

    # Shift assignments
    shift_stmt = (
        select(ScheduleEmployee)
        .where(ScheduleEmployee.employee_id == employee_id)
        .options(selectinload(ScheduleEmployee.schedule))
    )

    if start_date:
        shift_stmt = shift_stmt.join(Schedule).where(Schedule.start_date >= start_date)
    if end_date:
        shift_stmt = shift_stmt.join(Schedule, isouter=True).where(
            Schedule.start_date <= end_date
        )

    result = await db.execute(shift_stmt)
    assignments = result.scalars().all()

    shift_details = [
        ShiftAssignmentDetail(
            schedule_name=a.schedule.name,
            description=a.schedule.description,
            schedule_date=a.schedule.start_date.strftime("%Y-%m-%d"),
            start_time=a.schedule.start_time.strftime("%H:%M:%S") if a.schedule.start_time else None,
            end_time=a.schedule.end_time.strftime("%H:%M:%S") if a.schedule.end_time else None,
            assigned_date=a.assigned_date.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for a in assignments
    ]

    return EmployeeFullDetail(
        employee=employee,
        unavailability=unavailabilities,
        shift_assignments=shift_details,
    )
