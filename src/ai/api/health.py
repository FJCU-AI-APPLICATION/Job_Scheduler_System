from fastapi import APIRouter

from ai.agents.registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    registry = get_registry()
    return {
        "status": "ok",
        "available_checkpoints": registry.list_checkpoints(),
    }
