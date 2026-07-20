"""Популярностный baseline: score = positive_ratio * log1p(user_reviews).

Фильтры по бюджету и стоп-тегам применяются до ранжирования; уже известные
пользователю игры исключаются. Объяснение строится только из данных игры.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    game_id: int
    title: str
    tags: list[str]
    price: float
    positive_ratio: int
    user_reviews: int


@dataclass(frozen=True)
class Scored:
    game_id: int
    score: float
    reason: str


def rank_popular(
    candidates: list[Candidate],
    *,
    max_price: float | None,
    blocked_tags: list[str],
    preferred_genres: list[str],
    exclude_game_ids: set[int],
    top_n: int = 10,
) -> list[Scored]:
    blocked = {t.lower() for t in blocked_tags}
    preferred = {g.lower() for g in preferred_genres}

    scored: list[Scored] = []
    for c in candidates:
        if c.game_id in exclude_game_ids:
            continue
        if max_price is not None and c.price > max_price:
            continue
        tags_lower = {t.lower() for t in c.tags}
        if tags_lower & blocked:
            continue

        score = c.positive_ratio * math.log1p(c.user_reviews)
        genre_overlap = sorted(tags_lower & preferred)
        if genre_overlap:
            # совпадение жанров поднимает игру, но популярность остаётся основой
            score *= 1.0 + 0.25 * len(genre_overlap)

        scored.append(
            Scored(game_id=c.game_id, score=round(score, 2), reason=_reason(c, genre_overlap))
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_n]


def _reason(c: Candidate, genre_overlap: list[str]) -> str:
    parts = [f"{c.positive_ratio}% из {c.user_reviews:,} отзывов положительные".replace(",", " ")]
    if genre_overlap:
        parts.append("совпадение с вашими жанрами: " + ", ".join(genre_overlap))
    if c.price == 0:
        parts.append("бесплатная")
    return "Популярная игра: " + "; ".join(parts)
