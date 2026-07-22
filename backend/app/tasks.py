"""RQ-задачи: импорт данных и переобучение с promotion gate.

Выполняются в worker-контейнере (Linux). Переобучение: пересборка артефакта из
data/processed, быстрый offline-замер P@10 нового артефакта и gate — новая версия
активируется только если бьёт популярностный baseline; иначе остаётся текущая
(проверочный инцидент «новая модель хуже baseline»).
"""

import json
import logging
import random
import sys

import pandas as pd
from sqlalchemy import select, update

from app.config import settings
from app.models import Game, ModelVersion
from app.sync_db import SyncSession

logger = logging.getLogger("sidequest.tasks")

GATE_EVAL_USERS = 300
GATE_SEED = 42


def _ml_import(name: str):
    ml_dir = str(settings.ml_path)
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    return __import__(name)


def import_demo_data() -> dict:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    demo_dir = settings.data_path / "demo"
    df = pd.read_csv(demo_dir / "games.csv")
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
    with SyncSession() as session:
        stmt = pg_insert(Game).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Game.id],
            set_={c: stmt.excluded[c] for c in rows[0] if c != "id"},
        )
        session.execute(stmt)
        session.commit()
    logger.info("import_demo_data: upserted %d games", len(rows))
    return {"status": "ok", "games_upserted": len(rows)}


def _gate_precision(artifact: dict, train: pd.DataFrame, test: pd.DataFrame) -> float:
    """P@10 артефакта на подвыборке test-пользователей — той же формулой, что serving."""
    neighbors = {
        int(k): {int(g): float(s) for g, s in pairs}
        for k, pairs in artifact["neighbors"].items()
    }
    pop = {int(k): float(v) for k, v in artifact["popularity"].items()}
    w = artifact["params"]["weights"]

    likes = train[train["is_recommended"]].groupby("user_id")["app_id"].agg(list).to_dict()
    seen = train.groupby("user_id")["app_id"].agg(set).to_dict()
    test_pos = (
        test[test["is_recommended"]].groupby("user_id")["app_id"].agg(set).to_dict()
    )

    rng = random.Random(GATE_SEED)
    users = rng.sample(sorted(test_pos), min(GATE_EVAL_USERS, len(test_pos)))
    game_ids = [int(g) for g in pop]

    hits_total = 0
    for user in users:
        sims: dict[int, float] = {}
        for liked in likes.get(user, []):
            for gid, s in neighbors.get(liked, {}).items():
                sims[gid] = sims.get(gid, 0.0) + s
        max_sim = max(sims.values()) if sims else 0.0
        exclude = seen.get(user, set())
        scored = sorted(
            (
                (
                    -(
                        w["similarity"] * (sims.get(g, 0.0) / max_sim if max_sim else 0.0)
                        + w["popularity"] * pop.get(g, 0.0)
                    ),
                    g,
                )
                for g in game_ids
                if g not in exclude
            ),
        )[:10]
        relevant = test_pos[user] - exclude
        hits_total += sum(1 for _, g in scored if g in relevant)
    return hits_total / (len(users) * 10)


def retrain(simulate_degraded: bool = False) -> dict:
    train_serving = _ml_import("train_serving")

    processed = settings.data_path / "processed"
    games = pd.read_csv(processed / "games.csv")
    train = pd.read_csv(processed / "train.csv")
    test = pd.read_csv(processed / "test.csv")
    manifest = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))

    with SyncSession() as session:
        max_id = session.execute(select(ModelVersion.id).order_by(ModelVersion.id.desc())).scalar()
    version = f"v{(max_id or 0) + 1}"

    artifact = train_serving.build_artifact(
        games, train, version,
        manifest["params"] | {"dataset_kaggle_version": manifest["source_kaggle_version"]},
    )
    artifact["dataset_kaggle_version"] = manifest["source_kaggle_version"]

    if simulate_degraded:
        # инцидент №6: «обучение» на перемешанных соседях даёт заведомо слабую модель
        rng = random.Random(GATE_SEED)
        gids = [int(g) for g in artifact["popularity"]]
        artifact["neighbors"] = {
            k: [[rng.choice(gids), 0.01] for _ in range(5)] for k in artifact["neighbors"]
        }
        artifact["popularity"] = {k: rng.random() for k in artifact["popularity"]}

    new_p10 = _gate_precision(artifact, train, test)
    baseline_path = settings.ml_path / "results" / "popularity.json"
    baseline_p10 = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"][
        "precision_at_10"
    ]

    with SyncSession() as session:
        mv = ModelVersion(
            name="hybrid",
            version=version,
            params=artifact["params"] | {"gate_eval_users": GATE_EVAL_USERS},
            metrics={"gate_precision_at_10": round(new_p10, 4), "baseline_p10": baseline_p10},
            dataset_hash=manifest["source_kaggle_version"],
        )
        session.add(mv)

        if new_p10 <= baseline_p10:
            mv.is_active = False
            session.commit()
            logger.warning(
                "promotion gate: %s отклонена (P@10 %.4f <= baseline %.4f) — активная не меняется",
                version, new_p10, baseline_p10,
            )
            return {
                "status": "rejected",
                "version": version,
                "gate_precision_at_10": round(new_p10, 4),
                "baseline_p10": baseline_p10,
            }

        session.execute(update(ModelVersion).values(is_active=False))
        mv.is_active = True
        session.commit()
        # запись после коммита и атомарно: обрыв между коммитом и записью ловится
        # сверкой версий в serving (fallback), а не тихой отдачей битого файла
        out = settings.ml_path / "artifacts" / "hybrid_current.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        logger.info("promotion gate: %s активирована (P@10 %.4f)", version, new_p10)
        return {
            "status": "promoted",
            "version": version,
            "gate_precision_at_10": round(new_p10, 4),
            "baseline_p10": baseline_p10,
        }
