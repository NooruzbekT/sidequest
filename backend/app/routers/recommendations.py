import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app import observability
from app.deps import get_session
from app.models import User
from app.schemas import GameOut, RecommendationItem, RecommendationsOut
from app.services.recommender import get_active_model, recommend_for_user

logger = logging.getLogger("sidequest.recommendations")

router = APIRouter(tags=["recommendations"])


@router.get("/users/{user_id}/recommendations", response_model=RecommendationsOut)
async def get_recommendations(user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    model = await get_active_model(session)
    if model is None:
        observability.record_error()
        raise HTTPException(status_code=503, detail="Нет активной модели — выполните импорт данных")

    started = time.perf_counter()
    try:
        served_model, items = await recommend_for_user(session, user, model)
    except Exception:
        observability.record_error()
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    observability.record_recommendation(served_model.name, served_model.version, latency_ms)
    logger.info(
        "user_id=%s model=%s_%s items=%d duration_ms=%.1f",
        user.id, served_model.name, served_model.version, len(items), latency_ms,
    )
    return RecommendationsOut(
        user_id=user.id,
        model_name=served_model.name,
        model_version=served_model.version,
        items=[
            RecommendationItem(
                game=GameOut.model_validate(game), rank=rank, score=score, reason=reason
            )
            for game, rank, score, reason in items
        ],
    )
