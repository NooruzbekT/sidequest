"""Оркестратор выдачи: активная модель + fallback.

Активная модель hybrid → скоринг по артефакту (item-item похожесть + популярность).
Артефакт недоступен/повреждён или модель неизвестна → fallback: популярностный
baseline с теми же фильтрами; в ответе честно указывается фактически отработавшая
версия модели.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Game, Interaction, ModelVersion, Recommendation, User
from app.services import hybrid
from app.services.baseline import Candidate, rank_popular
from app.services.hybrid import ServingError

logger = logging.getLogger("sidequest.recommender")

# пул кандидатов ограничен самыми обсуждаемыми играми: полный каталог в память не тянем
CANDIDATE_POOL = 500
TOP_N = 10


async def get_active_model(session: AsyncSession) -> ModelVersion | None:
    result = await session.execute(
        select(ModelVersion).where(ModelVersion.is_active).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


async def _get_model_by_name(session: AsyncSession, name: str) -> ModelVersion | None:
    result = await session.execute(
        select(ModelVersion).where(ModelVersion.name == name).order_by(ModelVersion.id.desc())
    )
    return result.scalars().first()


def _passes_filters(game: Game, max_price: float | None, blocked: set[str]) -> bool:
    if max_price is not None and game.price > max_price:
        return False
    return not blocked & {t.lower() for t in (game.tags or [])}


async def _load_context(session: AsyncSession, user: User):
    interactions = (
        (await session.execute(select(Interaction).where(Interaction.user_id == user.id)))
        .scalars()
        .all()
    )
    interacted = {i.game_id for i in interactions}
    liked = [i.game_id for i in interactions if i.kind == "like"]

    games = (
        (
            await session.execute(
                select(Game).order_by(Game.user_reviews.desc()).limit(CANDIDATE_POOL)
            )
        )
        .scalars()
        .all()
    )
    return games, interacted, liked


def _recommend_hybrid(
    games: list[Game], interacted: set[int], liked: list[int], user: User
) -> list[tuple[Game, float, str]]:
    artifact = hybrid.get_artifact(settings.model_artifact_path)
    blocked = {t.lower() for t in (user.blocked_tags or [])}
    candidates = [
        g for g in games if g.id not in interacted and _passes_filters(g, user.max_price, blocked)
    ]
    scored = hybrid.score_candidates(artifact, liked, [g.id for g in candidates])
    by_id = {g.id: g for g in candidates}
    titles = {g.id: g.title for g in games}

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[0]))[:TOP_N]
    items = []
    for gid, (score, source_id) in ranked:
        game = by_id[gid]
        if source_id is not None and source_id in titles:
            reason = (
                f"Похожа на «{titles[source_id]}», которая вам понравилась "
                f"(совпадение вкусов игроков); {game.positive_ratio}% положительных отзывов"
            )
        else:
            reason = (
                f"Популярная игра: {game.positive_ratio}% из "
                f"{game.user_reviews:,} отзывов положительные".replace(",", " ")
            )
        genre_overlap = sorted(
            {t.lower() for t in (game.tags or [])} & {g.lower() for g in (user.genres or [])}
        )
        if genre_overlap:
            reason += "; ваши жанры: " + ", ".join(genre_overlap)
        items.append((game, score, reason))
    return items


def _recommend_baseline(
    games: list[Game], interacted: set[int], user: User
) -> list[tuple[Game, float, str]]:
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
        exclude_game_ids=interacted,
        top_n=TOP_N,
    )
    by_id = {g.id: g for g in games}
    return [(by_id[s.game_id], s.score, s.reason) for s in ranked]


async def recommend_for_user(
    session: AsyncSession, user: User, model: ModelVersion
) -> tuple[ModelVersion, list[tuple[Game, int, float, str]]]:
    games, interacted, liked = await _load_context(session, user)

    served_model = model
    if model.name == "hybrid":
        try:
            scored = _recommend_hybrid(games, interacted, liked, user)
        except ServingError as e:
            logger.warning("fallback на baseline: %s", e)
            scored = _recommend_baseline(games, interacted, user)
            served_model = await _get_model_by_name(session, "baseline") or model
    else:
        scored = _recommend_baseline(games, interacted, user)

    items: list[tuple[Game, int, float, str]] = []
    for rank, (game, score, reason) in enumerate(scored, start=1):
        items.append((game, rank, score, reason))
        session.add(
            Recommendation(
                user_id=user.id,
                game_id=game.id,
                rank=rank,
                score=score,
                reason=reason,
                model_version_id=served_model.id,
            )
        )
    await session.commit()
    return served_model, items
