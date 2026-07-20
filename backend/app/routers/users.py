import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.models import Game, Interaction, User
from app.schemas import InteractionIn, InteractionOut, PreferencesIn, UserCreate, UserOut

router = APIRouter(prefix="/users", tags=["users"])


async def _get_user(session: AsyncSession, user_id: uuid.UUID) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return user


@router.get("/demo", response_model=list[UserOut])
async def demo_users(session: AsyncSession = Depends(get_session)):
    rows = await session.execute(select(User).where(User.name.like("Demo:%")).order_by(User.name))
    return rows.scalars().all()


@router.post("", response_model=UserOut, status_code=201)
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    user = User(name=payload.name)
    session.add(user)
    await session.commit()
    return user


@router.get("/{user_id}", response_model=UserOut)
async def get_user(user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    return await _get_user(session, user_id)


@router.post("/{user_id}/preferences", response_model=UserOut)
async def set_preferences(
    user_id: uuid.UUID, payload: PreferencesIn, session: AsyncSession = Depends(get_session)
):
    user = await _get_user(session, user_id)
    user.genres = payload.genres
    user.max_price = payload.max_price
    user.blocked_tags = payload.blocked_tags
    await session.commit()
    return user


@router.post("/{user_id}/interactions", response_model=InteractionOut, status_code=201)
async def add_interaction(
    user_id: uuid.UUID, payload: InteractionIn, session: AsyncSession = Depends(get_session)
):
    user = await _get_user(session, user_id)
    game = await session.get(Game, payload.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Игра не найдена")

    stmt = (
        pg_insert(Interaction)
        .values(user_id=user.id, game_id=game.id, kind=payload.kind)
        .on_conflict_do_nothing(constraint="uq_interaction_user_game_kind")
        .returning(Interaction.id)
    )
    inserted = (await session.execute(stmt)).scalar()
    await session.commit()
    return InteractionOut(game_id=game.id, kind=payload.kind, duplicate=inserted is None)


@router.get("/{user_id}/interactions", response_model=list[InteractionOut])
async def list_interactions(user_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    user = await _get_user(session, user_id)
    rows = (
        (await session.execute(select(Interaction).where(Interaction.user_id == user.id)))
        .scalars()
        .all()
    )
    return [InteractionOut(game_id=r.game_id, kind=r.kind, duplicate=False) for r in rows]
