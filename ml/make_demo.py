"""Формирует маленький demo-набор из сырого датасета (коммитится в репо).

Игры: топ по числу отзывов с непустыми тегами. Взаимодействия: пользователи
с >= MIN_REVIEWS отзывами внутри выбранных игр, не более MAX_USERS.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEMO_DIR = ROOT / "data" / "demo"

N_GAMES = 1500
MIN_REVIEWS = 5
MAX_USERS = 3000
CHUNK_SIZE = 5_000_000
SEED = 42


def build_games() -> pd.DataFrame:
    games = pd.read_csv(RAW_DIR / "games.csv")
    meta = pd.read_json(RAW_DIR / "games_metadata.json", lines=True)
    meta = meta[meta["tags"].map(len) > 0]

    df = games.merge(meta[["app_id", "description", "tags"]], on="app_id")
    df = df.sort_values("user_reviews", ascending=False).head(N_GAMES)
    df["tags"] = df["tags"].map(json.dumps)
    return df[
        [
            "app_id",
            "title",
            "description",
            "tags",
            "price_final",
            "date_release",
            "rating",
            "positive_ratio",
            "user_reviews",
        ]
    ]


def build_interactions(app_ids: set[int]) -> pd.DataFrame:
    parts = []
    reader = pd.read_csv(
        RAW_DIR / "recommendations.csv",
        usecols=["user_id", "app_id", "is_recommended", "hours", "date"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        parts.append(chunk[chunk["app_id"].isin(app_ids)])
    df = pd.concat(parts)

    counts = df["user_id"].value_counts()
    eligible = counts[counts >= MIN_REVIEWS].index
    sampled = pd.Series(eligible).sample(
        n=min(MAX_USERS, len(eligible)), random_state=SEED
    )
    return df[df["user_id"].isin(set(sampled))]


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    games = build_games()
    games.to_csv(DEMO_DIR / "games.csv", index=False)

    interactions = build_interactions(set(games["app_id"]))
    interactions.to_csv(DEMO_DIR / "interactions.csv", index=False)

    print(f"games: {len(games)}, interactions: {len(interactions)}, "
          f"users: {interactions['user_id'].nunique()}")


if __name__ == "__main__":
    main()
