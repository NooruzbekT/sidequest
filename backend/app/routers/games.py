from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.models import Game
from app.schemas import GameOut

router = APIRouter(prefix="/games", tags=["games"])

# устойчивые игровые сокращения → подстрока официального названия
SEARCH_ALIASES = {
    "cs": "counter-strike",
    "cs2": "counter-strike",
    "csgo": "counter-strike: global offensive",
    "cs go": "counter-strike: global offensive",
    "gta": "grand theft auto",
    "gta5": "grand theft auto v",
    "gta v": "grand theft auto v",
    "rdr2": "red dead redemption 2",
    "tf2": "team fortress 2",
    "l4d2": "left 4 dead 2",
    "dbd": "dead by daylight",
    "bg3": "baldur's gate 3",
    "r6": "rainbow six",
    "cod": "call of duty",
    "ets2": "euro truck",
    "wow": "world of warcraft",
}


def resolve_query(query: str) -> str:
    return SEARCH_ALIASES.get(query.strip().lower(), query)


@router.get("", response_model=list[GameOut])
async def search_games(
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Game).order_by(Game.user_reviews.desc()).limit(limit)
    if query:
        stmt = stmt.where(Game.title.ilike(f"%{resolve_query(query)}%"))
    return (await session.execute(stmt)).scalars().all()
