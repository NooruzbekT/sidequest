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

# у самых известных игр Steam в games_metadata.json пустые теги (особенность датасета);
# без тегов они выпадали из demo-каталога — теги присвоены вручную
FAMOUS_TAGS: dict[int, list[str]] = {
    730: ["FPS", "Shooter", "Multiplayer", "Competitive", "Tactical", "Team-Based", "Action"],
    570: ["MOBA", "Strategy", "Multiplayer", "Team-Based", "Competitive", "Free to Play"],
    578080: ["Battle Royale", "Shooter", "Multiplayer", "Survival", "FPS", "Action"],
    271590: ["Open World", "Action", "Crime", "Multiplayer", "Third Person", "Mature"],
    359550: ["FPS", "Tactical", "Shooter", "Multiplayer", "Team-Based", "Competitive"],
    440: ["FPS", "Shooter", "Multiplayer", "Class-Based", "Free to Play", "Team-Based"],
    105600: ["Sandbox", "Survival", "2D", "Crafting", "Adventure", "Pixel Graphics"],
    4000: ["Sandbox", "Multiplayer", "Physics", "Comedy", "Moddable", "Building"],
    252490: ["Survival", "Multiplayer", "Open World", "Crafting", "PvP", "Building"],
    1172470: ["Battle Royale", "FPS", "Multiplayer", "Shooter", "Free to Play"],
    292030: ["RPG", "Open World", "Story Rich", "Fantasy", "Atmospheric", "Mature"],
    945360: ["Multiplayer", "Social Deduction", "Casual", "Party Game", "Space"],
    550: ["Zombies", "Co-op", "FPS", "Shooter", "Multiplayer", "Horror"],
    1085660: ["FPS", "Looter Shooter", "Multiplayer", "MMO", "Free to Play", "Sci-fi"],
    1091500: ["RPG", "Open World", "Cyberpunk", "Story Rich", "Sci-fi", "Mature"],
    230410: ["Action", "Free to Play", "Looter Shooter", "Co-op", "Sci-fi"],
    1245620: ["Souls-like", "RPG", "Open World", "Dark Fantasy", "Difficult", "Atmospheric"],
    413150: ["Farming Sim", "Life Sim", "Pixel Graphics", "Relaxing", "RPG", "Casual"],
    346110: ["Survival", "Open World", "Dinosaurs", "Multiplayer", "Crafting"],
    227300: ["Simulation", "Driving", "Automobile Sim", "Relaxing", "Open World"],
    381210: ["Horror", "Multiplayer", "Survival Horror", "PvP", "Co-op"],
    739630: ["Horror", "Co-op", "Multiplayer", "Psychological Horror", "Investigation"],
    1938090: ["FPS", "Shooter", "Multiplayer", "Action", "Military", "First-Person"],
    242760: ["Survival Horror", "Open World", "Crafting", "Co-op", "Horror", "Building"],
    218620: ["Co-op", "FPS", "Heist", "Action", "Multiplayer", "Crime"],
    1174180: ["Open World", "Western", "Story Rich", "Action", "Adventure", "Mature"],
    236390: ["Simulation", "War", "Military", "Multiplayer", "Free to Play", "Tanks"],
    892970: ["Survival", "Open World", "Crafting", "Co-op", "Building", "Multiplayer"],
    444090: ["Hero Shooter", "FPS", "Multiplayer", "Free to Play", "Team-Based"],
    291550: ["Fighting", "2D", "Free to Play", "Multiplayer", "Platformer"],
    221100: ["Survival", "Zombies", "Open World", "Multiplayer", "PvP"],
    620: ["Puzzle", "First-Person", "Story Rich", "Comedy", "Co-op", "Sci-fi"],
    239140: ["Zombies", "Parkour", "Open World", "Horror", "Co-op", "Action"],
    322330: ["Survival", "Co-op", "Multiplayer", "Crafting", "Atmospheric", "2D"],
    367520: ["Metroidvania", "Souls-like", "Platformer", "Atmospheric", "2D", "Difficult"],
    1086940: ["RPG", "Story Rich", "Turn-Based Combat", "Fantasy", "Choices Matter", "Co-op"],
    1172620: ["Pirates", "Open World", "Multiplayer", "Adventure", "Co-op"],
    648800: ["Survival", "Crafting", "Co-op", "Building", "Multiplayer"],
}


def build_games() -> pd.DataFrame:
    games = pd.read_csv(RAW_DIR / "games.csv")
    meta = pd.read_json(RAW_DIR / "games_metadata.json", lines=True)
    desc_map = dict(zip(meta["app_id"], meta["description"].fillna(""), strict=True))
    meta = meta[meta["tags"].map(len) > 0]

    df = games.merge(meta[["app_id", "description", "tags"]], on="app_id")
    df = df.sort_values("user_reviews", ascending=False).head(N_GAMES)

    famous = games[
        games["app_id"].isin(FAMOUS_TAGS) & ~games["app_id"].isin(set(df["app_id"]))
    ].copy()
    famous["description"] = famous["app_id"].map(desc_map).fillna("")
    famous["tags"] = famous["app_id"].map(FAMOUS_TAGS)

    df = pd.concat([df, famous], ignore_index=True)
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
