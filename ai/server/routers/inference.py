from fastapi import APIRouter, Query

from server.schemas import SchedulingRequest, SchedulingResponse
from server.services.ga_service import run_ga_inference
from server.services.rl_service import run_rl_inference

router = APIRouter(prefix="/predict", tags=["inference"])


@router.post("/rl", response_model=SchedulingResponse)
async def predict_rl(
    request: SchedulingRequest,
    checkpoint: str = Query("best_model.zip"),
) -> SchedulingResponse:
    """Run SB3 model inference for schedule optimization."""
    return run_rl_inference(request, checkpoint=checkpoint)


@router.post("/ga", response_model=SchedulingResponse)
async def predict_ga(
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
) -> SchedulingResponse:
    """Run Genetic Algorithm optimization for schedule."""
    return run_ga_inference(request, generations=generations, pop_size=pop_size)
