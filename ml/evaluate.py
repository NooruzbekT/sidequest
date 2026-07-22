"""Offline-evaluation моделей на временном сплите.

Запуск: python ml/evaluate.py [--users N]
Артефакты: ml/results/<model>.json (параметры сплита, метрики, время обучения).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import catalog_coverage, intra_list_diversity, precision_at_k, recall_at_k
from models import (
    ALSModel,
    ArtifactHybridModel,
    ContentBasedModel,
    DecayPopularityModel,
    HybridModel,
    ItemItemModel,
    PopularityModel,
    RandomModel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "ml" / "results"

SEED = 42
TOP_K = 10
EVAL_USERS = 2000


def slice_of(n_likes: int) -> str:
    if n_likes == 0:
        return "0_likes"
    if n_likes <= 2:
        return "1-2_likes"
    return "3+_likes"


def evaluate(
    model, games, train, test_pos_by_user, train_by_user, eval_users, div_model, likes_by_user
):
    t0 = time.perf_counter()
    model.fit(games, train)
    fit_seconds = time.perf_counter() - t0

    precisions, recalls, diversities, latencies = [], [], [], []
    slice_precisions: dict[str, list[float]] = {}
    all_recommended: set[int] = set()
    n_empty = 0

    for user_id in eval_users:
        exclude = train_by_user.get(user_id, set())
        t0 = time.perf_counter()
        recs = model.recommend(user_id, TOP_K, exclude)
        latencies.append(time.perf_counter() - t0)

        # повторный отзыв на ту же игру исключён из выдачи — не считаем его недостижимым хитом
        relevant = test_pos_by_user[user_id] - exclude
        p = precision_at_k(recs, relevant, TOP_K)
        precisions.append(p)
        slice_precisions.setdefault(slice_of(likes_by_user.get(user_id, 0)), []).append(p)
        recalls.append(recall_at_k(recs, relevant, TOP_K))
        if recs:
            diversities.append(intra_list_diversity(div_model.item_vectors(recs)))
        else:
            n_empty += 1
        all_recommended.update(recs)

    lat_ms = np.array(latencies) * 1000
    return {
        "model": model.name,
        "fit_seconds": round(fit_seconds, 2),
        "eval_users": len(eval_users),
        "eval_seed": SEED,
        "top_k": TOP_K,
        "metrics": {
            "precision_at_10": round(float(np.mean(precisions)), 4),
            "recall_at_10": round(float(np.mean(recalls)), 4),
            "coverage": round(catalog_coverage(all_recommended, len(games)), 4),
            "diversity": round(float(np.mean(diversities)), 4) if diversities else None,
            "latency_ms_mean": round(float(lat_ms.mean()), 2),
            "latency_ms_p95": round(float(np.percentile(lat_ms, 95)), 2),
            "empty_recs_share": round(n_empty / len(eval_users), 4),
        },
        "precision_at_10_by_slice": {
            k: {"p_at_10": round(float(np.mean(v)), 4), "n_users": len(v)}
            for k, v in sorted(slice_precisions.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=EVAL_USERS)
    args = parser.parse_args()

    games = pd.read_csv(DATA_DIR / "games.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    manifest = json.loads((DATA_DIR / "split_manifest.json").read_text(encoding="utf-8"))

    test_pos = test[test["is_recommended"]]
    test_pos_by_user = test_pos.groupby("user_id")["app_id"].agg(set).to_dict()
    train_by_user = train.groupby("user_id")["app_id"].agg(set).to_dict()
    likes_by_user = train[train["is_recommended"]].groupby("user_id").size().to_dict()

    rng = np.random.default_rng(SEED)
    candidates = sorted(test_pos_by_user)
    eval_users = list(rng.choice(candidates, size=min(args.users, len(candidates)), replace=False))

    # диверсити считаем в одном признаковом пространстве (TF-IDF) для всех моделей
    div_model = ContentBasedModel().fit(games, train)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    models = (
        RandomModel(),
        PopularityModel(),
        DecayPopularityModel(),
        ContentBasedModel(),
        ItemItemModel(),
        ALSModel(),
        HybridModel(),
        ArtifactHybridModel(),
    )
    for model in models:
        result = evaluate(
            model, games, train, test_pos_by_user, train_by_user, eval_users, div_model,
            likes_by_user,
        )
        result["split"] = manifest["params"] | {
            "dataset": manifest["source_dataset"],
            "kaggle_version": manifest["source_kaggle_version"],
        }
        out = RESULTS_DIR / f"{model.name}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(result)
        print(f"{model.name}: {result['metrics']}")

    print("\n| Модель | P@10 | R@10 | Coverage | Diversity | Latency mean | Latency p95 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["metrics"]
        print(
            f"| {r['model']} | {m['precision_at_10']:.4f} | {m['recall_at_10']:.4f} "
            f"| {m['coverage']:.1%} | {m['diversity']:.3f} "
            f"| {m['latency_ms_mean']:.1f} ms | {m['latency_ms_p95']:.1f} ms |"
        )


if __name__ == "__main__":
    main()
