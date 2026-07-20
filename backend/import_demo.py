"""Импорт demo-набора в БД: игры, baseline-версия модели, три demo-профиля.

Запуск: python import_demo.py (из backend/, БД должна быть поднята).
Идемпотентен: игры апсертятся, модель и профили создаются один раз.
"""

import asyncio
import json
import sys
from pathlib import Path

# psycopg async несовместим с ProactorEventLoop (дефолт Windows)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.deps import session_factory
from app.models import Game, Interaction, ModelVersion, User

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"

DEMO_PROFILES = [
    {
        "name": "Demo: инди-исследователь",
        "genres": ["Indie", "Adventure", "Puzzle"],
        "max_price": 15.0,
        "blocked_tags": ["Horror"],
    },
    {
        "name": "Demo: стратег",
        "genres": ["Strategy", "Simulation", "RPG"],
        "max_price": 30.0,
        "blocked_tags": ["Sports"],
    },
    {
        "name": "Demo: казуальный игрок",
        "genres": ["Casual", "Colorful", "Cute"],
        "max_price": 10.0,
        "blocked_tags": ["Violent", "Horror"],
    },
]
LIKES_PER_PROFILE = 5


async def import_games(session) -> int:
    df = pd.read_csv(DEMO_DIR / "games.csv")
    rows = [
        {
            "id": int(r.app_id),
            "title": r.title,
            "description": r.description if isinstance(r.description, str) else "",
            "tags": json.loads(r.tags),
            "price": float(r.price_final),
            "date_release": pd.Timestamp(r.date_release).date(),
            "rating": r.rating,
            "positive_ratio": int(r.positive_ratio),
            "user_reviews": int(r.user_reviews),
        }
        for r in df.itertuples()
    ]
    stmt = pg_insert(Game).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Game.id],
        set_={c: stmt.excluded[c] for c in rows[0] if c != "id"},
    )
    await session.execute(stmt)
    await session.commit()
    return len(rows)


async def ensure_model(session) -> None:
    existing = (
        (await session.execute(select(ModelVersion).where(ModelVersion.name == "baseline")))
        .scalars()
        .first()
    )
    if existing:
        return
    session.add(
        ModelVersion(
            name="baseline",
            version="v1",
            params={"strategy": "popularity", "score": "positive_ratio*log1p(reviews)"},
            metrics={},
            is_active=True,
        )
    )
    await session.commit()


async def ensure_demo_users(session) -> None:
    existing = (
        (await session.execute(select(User).where(User.name.like("Demo:%")))).scalars().all()
    )
    if existing:
        return

    for profile in DEMO_PROFILES:
        user = User(**profile)
        session.add(user)
        await session.flush()

        genre_set = {g.lower() for g in profile["genres"]}
        top = (
            (
                await session.execute(
                    select(Game).order_by(Game.user_reviews.desc()).limit(300)
                )
            )
            .scalars()
            .all()
        )
        liked = [
            g for g in top if genre_set & {t.lower() for t in (g.tags or [])}
        ][:LIKES_PER_PROFILE]
        for g in liked:
            session.add(Interaction(user_id=user.id, game_id=g.id, kind="like"))
    await session.commit()


async def main() -> None:
    async with session_factory() as session:
        n = await import_games(session)
        await ensure_model(session)
        await ensure_demo_users(session)
        print(f"imported games: {n}; model baseline v1 active; demo users ready")


if __name__ == "__main__":
    asyncio.run(main())
