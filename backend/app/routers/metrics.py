from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app import observability
from app.deps import get_session
from app.services.recommender import get_active_model

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(session: AsyncSession = Depends(get_session)) -> dict:
    stats = observability.snapshot()
    model = await get_active_model(session)
    stats["active_model"] = (
        {
            "name": model.name,
            "version": model.version,
            "trained_at": model.trained_at.isoformat(),
            "offline_metrics": model.metrics,
        }
        if model
        else None
    )
    return stats
