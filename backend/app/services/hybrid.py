"""Serving гибридной модели из артефакта.

Артефакт: item-item соседи + decay-popularity (см. ml/train_serving.py).
final = w_sim * sim_norm + w_pop * pop; без лайков у пользователя — чистая
популярность (cold start). Ошибка загрузки артефакта → ServingError, выше по
стеку срабатывает fallback на baseline.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class ServingError(Exception):
    pass


@dataclass
class HybridArtifact:
    version: str
    weights: dict
    popularity: dict[int, float]
    neighbors: dict[int, dict[int, float]]

    @classmethod
    def load(cls, path: str | Path) -> "HybridArtifact":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(
                version=raw["version"],
                weights=raw["params"]["weights"],
                popularity={int(k): float(v) for k, v in raw["popularity"].items()},
                neighbors={
                    int(k): {int(g): float(s) for g, s in pairs}
                    for k, pairs in raw["neighbors"].items()
                },
            )
        except (OSError, KeyError, ValueError, TypeError) as e:
            raise ServingError(f"артефакт гибридной модели недоступен: {e}") from e


_cache: dict[str, HybridArtifact] = {}


def get_artifact(path: str) -> HybridArtifact:
    if path not in _cache:
        _cache[path] = HybridArtifact.load(path)
    return _cache[path]


def score_candidates(
    artifact: HybridArtifact, liked_ids: list[int], candidate_ids: list[int]
) -> dict[int, tuple[float, int | None]]:
    """Для каждого кандидата: (score, id самой похожей лайкнутой игры или None)."""
    sims: dict[int, float] = {}
    best_sim: dict[int, float] = {}
    best_source: dict[int, int] = {}
    for liked in liked_ids:
        for gid, s in artifact.neighbors.get(liked, {}).items():
            sims[gid] = sims.get(gid, 0.0) + s
            if s > best_sim.get(gid, 0.0):
                best_sim[gid] = s
                best_source[gid] = liked

    max_sim = max(sims.values()) if sims else 0.0
    w_sim = artifact.weights["similarity"]
    w_pop = artifact.weights["popularity"]

    result: dict[int, tuple[float, int | None]] = {}
    for gid in candidate_ids:
        sim_norm = sims.get(gid, 0.0) / max_sim if max_sim > 0 else 0.0
        score = w_sim * sim_norm + w_pop * artifact.popularity.get(gid, 0.0)
        result[gid] = (round(score, 6), best_source.get(gid) if sim_norm > 0 else None)
    return result
