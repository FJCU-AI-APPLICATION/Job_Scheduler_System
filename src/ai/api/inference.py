"""FastAPI inference routes."""

from enum import Enum

from fastapi import APIRouter, Query

from ai.domain.schemas import SchedulingRequest, SchedulingResponse
from ai.optimizers.base import Optimizer
from ai.services.optimizer_inference import run_optimizer_inference
from ai.services.rl_inference import run_rl_inference

router = APIRouter(prefix="/predict", tags=["inference"])

# Built dynamically so adding to the registry auto-extends the API enum.
EvolutionaryAlgorithm = Enum(
    "EvolutionaryAlgorithm",
    {n.upper(): n for n in Optimizer.list_available()},
)


@router.post("/rl", response_model=SchedulingResponse)
async def predict_rl(
    request: SchedulingRequest,
    checkpoint: str = Query("best_model.zip"),
) -> SchedulingResponse:
    """Run SB3 model inference for schedule optimization."""
    return run_rl_inference(request, checkpoint=checkpoint)


@router.post("/evolutionary/{algorithm}", response_model=SchedulingResponse)
async def predict_evolutionary(
    algorithm: EvolutionaryAlgorithm,
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
    device: str = Query("cpu", pattern=r"^(cpu|cuda)$"),
) -> SchedulingResponse:
    """Run an evolutionary multi-objective optimizer ('nsga2' | 'ccmo')."""
    return run_optimizer_inference(
        algorithm.value,
        request,
        generations=generations,
        pop_size=pop_size,
        device=device,
    )


@router.post("/ga", response_model=SchedulingResponse, deprecated=True)
async def predict_ga(
    request: SchedulingRequest,
    generations: int = Query(100, ge=1, le=1000),
    pop_size: int = Query(50, ge=10, le=500),
) -> SchedulingResponse:
    """DEPRECATED: use /predict/evolutionary/nsga2."""
    return run_optimizer_inference(
        "nsga2", request, generations=generations, pop_size=pop_size
    )
