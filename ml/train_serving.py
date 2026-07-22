"""Обучение serving-артефакта гибридной модели.

Считает по train-сплиту item-item соседей (top-N по косинусу со-встречаемости лайков)
и decay-popularity, пишет ml/artifacts/hybrid_v2.json. Регистрация версии в БД —
backend/register_model.py (переключение serving на новую версию).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import DecayPopularityModel, ItemItemModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

VERSION = "v2"
TOP_NEIGHBORS = 20


def main() -> None:
    games = pd.read_csv(DATA_DIR / "games.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    manifest = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))

    item_item = ItemItemModel().fit(games, train)
    decay = DecayPopularityModel().fit(games, train)

    game_ids = item_item._game_ids
    sim = item_item._sim
    neighbors: dict[str, list[list[float]]] = {}
    for i, gid in enumerate(game_ids):
        row = sim[i].tocoo()
        if row.nnz == 0:
            continue
        top = sorted(zip(row.col, row.data, strict=True), key=lambda t: -t[1])[:TOP_NEIGHBORS]
        neighbors[str(int(gid))] = [[int(game_ids[j]), round(float(s), 4)] for j, s in top]

    ranked = decay._ranked
    # нормированный к [0,1] скор популярности по позиции в ранге
    popularity = {
        str(int(gid)): round(1 - rank / len(ranked), 6) for rank, gid in enumerate(ranked)
    }

    artifact = {
        "model": "hybrid",
        "version": VERSION,
        "params": {
            "top_neighbors": TOP_NEIGHBORS,
            "decay_half_life_days": decay.half_life_days,
            "weights": {"similarity": 0.6, "popularity": 0.4},
        },
        "split": manifest["params"],
        "dataset_kaggle_version": manifest["source_kaggle_version"],
        "popularity": popularity,
        "neighbors": neighbors,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / f"hybrid_{VERSION}.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"artifact -> {out} ({size_mb:.1f} MB, {len(neighbors)} games with neighbors)")
    assert np.isfinite(size_mb)


if __name__ == "__main__":
    main()
