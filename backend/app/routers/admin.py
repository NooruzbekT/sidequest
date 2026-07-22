from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.schemas import ModelVersionOut
from app.services.recommender import get_active_model

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models/current", response_model=ModelVersionOut)
async def current_model(session: AsyncSession = Depends(get_session)):
    model = await get_active_model(session)
    if model is None:
        raise HTTPException(status_code=404, detail="Активная модель не найдена")
    return model


@router.post("/retrain", status_code=202)
async def retrain():
    # выполнение — фоновой задачей через очередь; здесь фиксируется только контракт
    return {"status": "queued", "detail": "Переобучение будет выполнено фоновой задачей"}
