import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.models import User
from app.schemas import GameOut, RecommendationItem, RecommendationsOut
from app.services.recommender import get_active_model, recommend_for_user

router = APIRouter(tags=["recommendations"])


@router.get("/users/{user_id}/recommendations", response_model=RecommendationsOut)
async def get_recommendations(user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    model = await get_active_model(session)
    if model is None:
        raise HTTPException(status_code=503, detail="Нет активной модели — выполните импорт данных")

    items = await recommend_for_user(session, user, model)
    return RecommendationsOut(
        user_id=user.id,
        model_name=model.name,
        model_version=model.version,
        items=[
            RecommendationItem(
                game=GameOut.model_validate(game), rank=rank, score=score, reason=reason
            )
            for game, rank, score, reason in items
        ],
    )
