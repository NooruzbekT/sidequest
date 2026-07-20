from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.models import Game
from app.schemas import GameOut

router = APIRouter(prefix="/games", tags=["games"])


@router.get("", response_model=list[GameOut])
async def search_games(
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Game).order_by(Game.user_reviews.desc()).limit(limit)
    if query:
        stmt = stmt.where(Game.title.ilike(f"%{query}%"))
    return (await session.execute(stmt)).scalars().all()
