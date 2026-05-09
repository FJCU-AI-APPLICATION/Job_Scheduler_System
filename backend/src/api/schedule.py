from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from domain.schedule import Schedule
from schemas.common import PaginatedResponse
from schemas.schedule import (
    ComputeScheduleRequest,
    ConfirmScheduleRequest,
    ScheduleCreate,
    ScheduleResponse,
    ScheduleUpdate,
)
from services import scheduling

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.get("/", response_model=PaginatedResponse[ScheduleResponse])
async def list_schedules(
    page: int = Query(1, ge=1),
    page_size: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    count_stmt = select(func.count()).select_from(Schedule)
    total = (await db.execute(count_stmt)).scalar() or 0

    offset = (page - 1) * page_size
    stmt = select(Schedule).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    schedules = result.scalars().all()

    return PaginatedResponse(
        count=total,
        page=page,
        page_size=page_size,
        results=schedules,
    )


@router.post("/", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    payload: ScheduleCreate,
    db: AsyncSession = Depends(get_db),
):
    schedule = Schedule(**payload.model_dump())
    db.add(schedule)
    await db.flush()
    await db.refresh(schedule)
    return schedule


@router.get("/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(schedule_id: int, db: AsyncSession = Depends(get_db)):
    schedule = await db.get(Schedule, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.put("/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: int,
    payload: ScheduleUpdate,
    db: AsyncSession = Depends(get_db),
):
    schedule = await db.get(Schedule, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(schedule, key, value)

    await db.flush()
    await db.refresh(schedule)
    return schedule


@router.delete("/{schedule_id}", status_code=204)
async def delete_schedule(schedule_id: int, db: AsyncSession = Depends(get_db)):
    schedule = await db.get(Schedule, schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    await db.delete(schedule)


@router.post("/compute/")
async def compute_schedule_endpoint(
    payload: ComputeScheduleRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        computed = await scheduling.compute_schedule(payload, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return asdict(computed)


@router.post("/confirm/", status_code=201)
async def confirm_schedule_endpoint(
    payload: ConfirmScheduleRequest,
    db: AsyncSession = Depends(get_db),
):
    created_ids = await scheduling.confirm_schedule(payload, db)
    return {"created_schedule_ids": created_ids}
