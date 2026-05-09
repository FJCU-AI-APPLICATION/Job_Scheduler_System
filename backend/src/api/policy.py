from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from domain.policy import AiModel, Policy, ShiftPolicy
from schemas.policy import (
    AiModelCreate,
    AiModelResponse,
    AiModelUpdate,
    PolicyCreate,
    PolicyResponse,
    PolicyUpdate,
    ShiftPolicyCreate,
    ShiftPolicyResponse,
    ShiftPolicyUpdate,
)

router = APIRouter(prefix="/api/policy", tags=["policy"])


@router.get("/", response_model=list[PolicyResponse])
async def list_policies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Policy))
    return result.scalars().all()


@router.post("/", response_model=PolicyResponse, status_code=201)
async def create_policy(payload: PolicyCreate, db: AsyncSession = Depends(get_db)):
    policy = Policy(**payload.model_dump())
    db.add(policy)
    await db.flush()
    await db.refresh(policy)
    return policy


@router.get("/{policy_id}/", response_model=PolicyResponse)
async def get_policy(policy_id: int, db: AsyncSession = Depends(get_db)):
    policy = await db.get(Policy, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@router.put("/{policy_id}/", response_model=PolicyResponse)
async def update_policy(
    policy_id: int,
    payload: PolicyUpdate,
    db: AsyncSession = Depends(get_db),
):
    policy = await db.get(Policy, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(policy, key, value)

    await db.flush()
    await db.refresh(policy)
    return policy


@router.delete("/{policy_id}/", status_code=204)
async def delete_policy(policy_id: int, db: AsyncSession = Depends(get_db)):
    policy = await db.get(Policy, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    await db.delete(policy)


@router.get("/shiftpolicy/", response_model=list[ShiftPolicyResponse])
async def list_shift_policies(
    policy_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(ShiftPolicy)
        .where(ShiftPolicy.policy_id == policy_id)
        .order_by(ShiftPolicy.start_time)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/shiftpolicy/", response_model=ShiftPolicyResponse, status_code=201)
async def create_shift_policy(
    payload: ShiftPolicyCreate,
    db: AsyncSession = Depends(get_db),
):
    policy = await db.get(Policy, payload.policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    shift_policy = ShiftPolicy(**payload.model_dump())
    db.add(shift_policy)
    await db.flush()
    await db.refresh(shift_policy)
    return shift_policy


@router.get("/shiftpolicy/{shift_policy_id}/", response_model=ShiftPolicyResponse)
async def get_shift_policy(shift_policy_id: int, db: AsyncSession = Depends(get_db)):
    shift_policy = await db.get(ShiftPolicy, shift_policy_id)
    if not shift_policy:
        raise HTTPException(status_code=404, detail="ShiftPolicy not found")
    return shift_policy


@router.put("/shiftpolicy/{shift_policy_id}/", response_model=ShiftPolicyResponse)
async def update_shift_policy(
    shift_policy_id: int,
    payload: ShiftPolicyUpdate,
    db: AsyncSession = Depends(get_db),
):
    shift_policy = await db.get(ShiftPolicy, shift_policy_id)
    if not shift_policy:
        raise HTTPException(status_code=404, detail="ShiftPolicy not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(shift_policy, key, value)

    await db.flush()
    await db.refresh(shift_policy)
    return shift_policy


@router.delete("/shiftpolicy/{shift_policy_id}/", status_code=204)
async def delete_shift_policy(shift_policy_id: int, db: AsyncSession = Depends(get_db)):
    shift_policy = await db.get(ShiftPolicy, shift_policy_id)
    if not shift_policy:
        raise HTTPException(status_code=404, detail="ShiftPolicy not found")
    await db.delete(shift_policy)


@router.get("/aimodels/", response_model=list[AiModelResponse])
async def list_ai_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AiModel))
    return result.scalars().all()


@router.post("/aimodels/", response_model=AiModelResponse, status_code=201)
async def create_ai_model(payload: AiModelCreate, db: AsyncSession = Depends(get_db)):
    ai_model = AiModel(**payload.model_dump())
    db.add(ai_model)
    await db.flush()
    await db.refresh(ai_model)
    return ai_model


@router.get("/aimodels/{ai_model_id}/", response_model=AiModelResponse)
async def get_ai_model(ai_model_id: int, db: AsyncSession = Depends(get_db)):
    ai_model = await db.get(AiModel, ai_model_id)
    if not ai_model:
        raise HTTPException(status_code=404, detail="AiModel not found")
    return ai_model


@router.put("/aimodels/{ai_model_id}/", response_model=AiModelResponse)
async def update_ai_model(
    ai_model_id: int,
    payload: AiModelUpdate,
    db: AsyncSession = Depends(get_db),
):
    ai_model = await db.get(AiModel, ai_model_id)
    if not ai_model:
        raise HTTPException(status_code=404, detail="AiModel not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(ai_model, key, value)

    await db.flush()
    await db.refresh(ai_model)
    return ai_model


@router.delete("/aimodels/{ai_model_id}/", status_code=204)
async def delete_ai_model(ai_model_id: int, db: AsyncSession = Depends(get_db)):
    ai_model = await db.get(AiModel, ai_model_id)
    if not ai_model:
        raise HTTPException(status_code=404, detail="AiModel not found")
    await db.delete(ai_model)
