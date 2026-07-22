"""Регистрация версии модели и переключение serving.

Запуск из backend/: python register_model.py --name hybrid --version v2
Читает параметры из артефакта и метрики из ml/results/<name>.json (если есть),
создаёт запись ModelVersion и делает её активной. Идемпотентен по (name, version).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select, update

from app.models import ModelVersion

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.deps import session_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main(name: str, version: str) -> None:
    params, dataset_hash = {}, None
    artifact_path = ROOT / "ml" / "artifacts" / f"{name}_current.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("version") != version:
            raise SystemExit(
                f"в артефакте версия {artifact.get('version')}, а регистрируется {version} — "
                "сначала пересоберите артефакт"
            )
        params = artifact.get("params", {})
        dataset_hash = artifact.get("dataset_kaggle_version")

    metrics = {}
    # метрика served-артефакта честнее метрики полной модели, если она замерена
    for results_name in (f"{name}-artifact.json", f"{name}.json"):
        results_path = ROOT / "ml" / "results" / results_name
        if results_path.exists():
            metrics = json.loads(results_path.read_text(encoding="utf-8")).get("metrics", {})
            break

    async with session_factory() as session:
        existing = (
            (
                await session.execute(
                    select(ModelVersion).where(
                        ModelVersion.name == name, ModelVersion.version == version
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing is None:
            existing = ModelVersion(name=name, version=version)
            session.add(existing)
        existing.params = params
        existing.metrics = metrics
        existing.dataset_hash = dataset_hash
        await session.flush()

        await session.execute(update(ModelVersion).values(is_active=False))
        existing.is_active = True
        await session.commit()
        print(f"active model: {name} {version} (id={existing.id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.name, args.version))
