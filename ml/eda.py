"""EDA по сырому датасету: sparsity, распределения взаимодействий, топ-теги.

Читает data/raw, пишет отчёт в docs/eda.md. recommendations.csv (41M строк)
обрабатывается чанками, чтобы не упираться в память.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "docs" / "eda.md"

CHUNK_SIZE = 5_000_000


def interactions_stats() -> dict:
    user_parts: list[pd.Series] = []
    game_parts: list[pd.Series] = []
    n_rows = 0
    n_positive = 0
    date_min, date_max = "9999", "0000"

    reader = pd.read_csv(
        RAW_DIR / "recommendations.csv",
        usecols=["user_id", "app_id", "is_recommended", "date"],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        n_rows += len(chunk)
        n_positive += int(chunk["is_recommended"].sum())
        user_parts.append(chunk["user_id"].value_counts())
        game_parts.append(chunk["app_id"].value_counts())
        # даты в ISO-формате, лексикографическое сравнение корректно
        date_min = min(date_min, chunk["date"].min())
        date_max = max(date_max, chunk["date"].max())

    per_user = pd.concat(user_parts).groupby(level=0).sum()
    per_game = pd.concat(game_parts).groupby(level=0).sum()

    def quantiles(s: pd.Series) -> dict:
        q = s.quantile([0.5, 0.9, 0.99])
        return {"median": q[0.5], "p90": q[0.9], "p99": q[0.99], "max": int(s.max())}

    return {
        "n_interactions": n_rows,
        "positive_share": n_positive / n_rows,
        "n_users_active": len(per_user),
        "n_games_reviewed": len(per_game),
        "per_user": quantiles(per_user),
        "per_game": quantiles(per_game),
        "date_min": date_min,
        "date_max": date_max,
    }


def games_stats() -> dict:
    games = pd.read_csv(RAW_DIR / "games.csv")
    price = games["price_final"]
    return {
        "n_games": len(games),
        "free_share": float((price == 0).mean()),
        "price_median_paid": float(price[price > 0].median()),
        "under_25_share": float((price <= 25).mean()),
    }


def top_tags(n: int = 20) -> pd.Series:
    meta = pd.read_json(RAW_DIR / "games_metadata.json", lines=True)
    no_tags_share = float(meta["tags"].map(len).eq(0).mean())
    tags = meta["tags"].explode().dropna()
    top = tags.value_counts().head(n)
    top.attrs["no_tags_share"] = no_tags_share
    return top


def main() -> None:
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    inter = interactions_stats()
    games = games_stats()
    tags = top_tags()

    users_total = sum(1 for _ in open(RAW_DIR / "users.csv", encoding="utf-8")) - 1
    sparsity = 1 - inter["n_interactions"] / (inter["n_users_active"] * games["n_games"])

    lines = [
        "# EDA: Game Recommendations on Steam",
        "",
        f"Датасет: `{manifest['dataset']}`, версия Kaggle {manifest['kaggle_version']}.",
        "",
        "## Объёмы",
        "",
        f"- Игры: **{games['n_games']:,}**",
        f"- Пользователи (всего в users.csv): **{users_total:,}**",
        f"- Пользователи с отзывами: **{inter['n_users_active']:,}**",
        f"- Взаимодействия (отзывы): **{inter['n_interactions']:,}**",
        f"- Период: {inter['date_min']} — {inter['date_max']}",
        f"- Доля положительных (is_recommended): **{inter['positive_share']:.1%}**",
        f"- Sparsity матрицы user×game: **{sparsity:.4%}** (заполнено {1 - sparsity:.4%})",
        "",
        "## Распределение взаимодействий",
        "",
        "| | медиана | p90 | p99 | max |",
        "|---|---|---|---|---|",
        f"| на пользователя | {inter['per_user']['median']:.0f} | {inter['per_user']['p90']:.0f} "
        f"| {inter['per_user']['p99']:.0f} | {inter['per_user']['max']:,} |",
        f"| на игру | {inter['per_game']['median']:.0f} | {inter['per_game']['p90']:.0f} "
        f"| {inter['per_game']['p99']:.0f} | {inter['per_game']['max']:,} |",
        "",
        f"Игр с отзывами: {inter['n_games_reviewed']:,} из {games['n_games']:,} — "
        "хвост без отзывов пойдёт в cold-start-сценарий.",
        "",
        "## Цены",
        "",
        f"- Бесплатных игр: {games['free_share']:.1%}",
        f"- Медианная цена платной игры: ${games['price_median_paid']:.2f}",
        f"- Игр с ценой ≤ $25: {games['under_25_share']:.1%}",
        "",
        "## Топ-20 тегов (по числу игр)",
        "",
        f"Игр без тегов: {tags.attrs['no_tags_share']:.1%} — жанры выводим из тегов.",
        "",
        "| Тег | Игр |",
        "|---|---|",
    ]
    lines += [f"| {tag} | {count:,} |" for tag, count in tags.items()]
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
