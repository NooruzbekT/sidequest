"""Воспроизводимый train/test-сплит для offline-evaluation.

Временной сплит без утечки: train — взаимодействия до CUTOFF, test — после.
В test остаются только пользователи, у которых есть история в train (warm-start;
cold start оценивается отдельно сценарием, а не метрикой). Параметры и хеши
фиксируются в data/processed/split_manifest.json.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

SEED = 42
N_GAMES = 5000
MIN_USER_REVIEWS = 5
MAX_USERS = 50_000
CUTOFF = "2022-01-01"
CHUNK_SIZE = 5_000_000


def train_period_counts() -> pd.Series:
    # каталог отбирается по train-периоду: all-time агрегаты содержат тестовое будущее
    parts = []
    reader = pd.read_csv(
        RAW_DIR / "recommendations.csv",
        usecols=["app_id", "date"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        parts.append(chunk.loc[chunk["date"] < CUTOFF, "app_id"].value_counts())
    return pd.concat(parts).groupby(level=0).sum()


def select_games(train_counts: pd.Series) -> pd.DataFrame:
    games = pd.read_csv(RAW_DIR / "games.csv")
    meta = pd.read_json(RAW_DIR / "games_metadata.json", lines=True)
    meta = meta[meta["tags"].map(len) > 0].drop_duplicates("app_id")
    df = games.merge(meta[["app_id", "description", "tags"]], on="app_id", validate="1:1")
    df["train_reviews"] = df["app_id"].map(train_counts).fillna(0).astype(int)
    df = df.sort_values(["train_reviews", "app_id"], ascending=[False, True]).head(N_GAMES)
    df["tags"] = df["tags"].map(list)
    return df.reset_index(drop=True)


def collect_interactions(app_ids: set[int]) -> pd.DataFrame:
    parts = []
    reader = pd.read_csv(
        RAW_DIR / "recommendations.csv",
        usecols=["user_id", "app_id", "is_recommended", "date"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        parts.append(chunk[chunk["app_id"].isin(app_ids)])
    df = pd.concat(parts, ignore_index=True)

    # порог активности и сэмпл — только по train-периоду, без опоры на будущее
    counts = df.loc[df["date"] < CUTOFF, "user_id"].value_counts()
    eligible = counts[counts >= MIN_USER_REVIEWS].index.to_series().sort_values()
    sampled = eligible.sample(n=min(MAX_USERS, len(eligible)), random_state=SEED)
    return df[df["user_id"].isin(set(sampled))].reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    games = select_games(train_period_counts())
    inter = collect_interactions(set(games["app_id"]))

    train = inter[inter["date"] < CUTOFF]
    test = inter[inter["date"] >= CUTOFF]

    warm_users = set(train["user_id"])
    dropped_cold = int(test.loc[~test["user_id"].isin(warm_users), "user_id"].nunique())
    test = test[test["user_id"].isin(warm_users)]

    games_out = games.copy()
    games_out["tags"] = games_out["tags"].map(json.dumps)
    games_out.to_csv(OUT_DIR / "games.csv", index=False)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    raw_manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "source_dataset": raw_manifest["dataset"],
        "source_kaggle_version": raw_manifest["kaggle_version"],
        "source_sha256": {k: v["sha256"] for k, v in raw_manifest["files"].items()},
        "params": {
            "seed": SEED,
            "n_games": N_GAMES,
            "min_user_reviews": MIN_USER_REVIEWS,
            "max_users": MAX_USERS,
            "cutoff": CUTOFF,
            "selection": "каталог и порог активности пользователей — по train-периоду",
        },
        "sizes": {
            "games": len(games_out),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_users": int(train["user_id"].nunique()),
            "test_users": int(test["user_id"].nunique()),
            "cold_test_users_dropped": dropped_cold,
        },
        "excluded_fields": {
            "hours": "часы наигранного времени растут и после момента рекомендации",
            "helpful/funny": "оценки отзыва появляются после публикации отзыва",
            "games.user_reviews/positive_ratio в eval": (
                "агрегаты каталога включают отзывы тестового периода; "
                "в evaluation популярность считается только по train"
            ),
        },
    }
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["sizes"], indent=2))


if __name__ == "__main__":
    main()
