"""Оркестратор выдачи: собирает кандидатов из БД, зовёт активную модель, сохраняет результат."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Game, Interaction, ModelVersion, Recommendation, User
from app.services.baseline import Candidate, rank_popular

# пул кандидатов ограничен самыми обсуждаемыми играми: baseline всё равно ранжирует
# по популярности, а полный каталог в память не тянем
CANDIDATE_POOL = 500


async def get_active_model(session: AsyncSession) -> ModelVersion | None:
    result = await session.execute(
        select(ModelVersion).where(ModelVersion.is_active).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


async def recommend_for_user(
    session: AsyncSession, user: User, model: ModelVersion
) -> list[tuple[Game, int, float, str]]:
    interacted = (
        (
            await session.execute(
                select(Interaction.game_id).where(Interaction.user_id == user.id)
            )
        )
        .scalars()
        .all()
    )

    games = (
        (
            await session.execute(
                select(Game).order_by(Game.user_reviews.desc()).limit(CANDIDATE_POOL)
            )
        )
        .scalars()
        .all()
    )
    by_id = {g.id: g for g in games}

    candidates = [
        Candidate(
            game_id=g.id,
            title=g.title,
            tags=list(g.tags or []),
            price=g.price,
            positive_ratio=g.positive_ratio,
            user_reviews=g.user_reviews,
        )
        for g in games
    ]

    ranked = rank_popular(
        candidates,
        max_price=user.max_price,
        blocked_tags=list(user.blocked_tags or []),
        preferred_genres=list(user.genres or []),
        exclude_game_ids=set(interacted),
    )

    items: list[tuple[Game, int, float, str]] = []
    for rank, s in enumerate(ranked, start=1):
        game = by_id[s.game_id]
        items.append((game, rank, s.score, s.reason))
        session.add(
            Recommendation(
                user_id=user.id,
                game_id=game.id,
                rank=rank,
                score=s.score,
                reason=s.reason,
                model_version_id=model.id,
            )
        )
    await session.commit()
    return items
