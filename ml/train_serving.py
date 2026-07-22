"""Обучение serving-артефакта гибридной модели.

`build_artifact()` переиспользуется фоновой задачей переобучения (backend worker).
Файл артефакта один — hybrid_current.json, версия записана внутри; переключение
версий выполняет backend/register_model.py или promotion gate в задаче retrain.
"""

import json
from pathlib import Path

import pandas as pd

from models import DecayPopularityModel, ItemItemModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

VERSION = "v2"
TOP_NEIGHBORS = 20
WEIGHTS = {"similarity": 0.6, "popularity": 0.4}


def build_artifact(
    games: pd.DataFrame, train: pd.DataFrame, version: str, split_params: dict
) -> dict:
    item_item = ItemItemModel().fit(games, train)
    decay = DecayPopularityModel().fit(games, train)

    game_ids = item_item._game_ids
    sim = item_item._sim
    sim.eliminate_zeros()
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

    return {
        "model": "hybrid",
        "version": version,
        "params": {
            "top_neighbors": TOP_NEIGHBORS,
            "decay_half_life_days": decay.half_life_days,
            "weights": WEIGHTS,
        },
        "split": split_params,
        "popularity": popularity,
        "neighbors": neighbors,
    }


def main() -> None:
    games = pd.read_csv(DATA_DIR / "games.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    manifest = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))

    artifact = build_artifact(
        games,
        train,
        VERSION,
        manifest["params"] | {"dataset_kaggle_version": manifest["source_kaggle_version"]},
    )
    artifact["dataset_kaggle_version"] = manifest["source_kaggle_version"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "hybrid_current.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(f"artifact -> {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(artifact['neighbors'])} games with neighbors, version {VERSION})")


if __name__ == "__main__":
    main()
