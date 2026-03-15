from fastapi import APIRouter

from server.services.model_registry import get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    registry = get_registry()
    checkpoints = registry.list_checkpoints()
    return {
        "status": "ok",
        "available_checkpoints": checkpoints,
    }
