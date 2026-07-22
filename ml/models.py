"""Модели рекомендаций для offline-evaluation.

Обе модели обучаются строго на train-данных: popularity считает счётчики по train
(агрегаты каталога содержат отзывы тестового периода — утечка), content-based строит
TF-IDF по тегам и описанию и профиль пользователя как средний вектор его лайков.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TAG_WEIGHT = 3  # теги информативнее свободного текста описания


@dataclass
class PopularityModel:
    name: str = "popularity"
    _scores: pd.Series = field(default=None, repr=False)
    _ranked: list[int] = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "PopularityModel":
        grp = train.groupby("app_id")["is_recommended"]
        pos_share = grp.mean()
        n = grp.size()
        self._scores = pos_share * np.log1p(n)
        self._ranked = self._scores.sort_values(ascending=False).index.tolist()
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        result = []
        for app_id in self._ranked:
            if app_id not in exclude:
                result.append(app_id)
                if len(result) == n:
                    break
        return result


@dataclass
class ContentBasedModel:
    name: str = "content-based"
    _matrix: sparse.csr_matrix = field(default=None, repr=False)
    _game_ids: np.ndarray = field(default=None, repr=False)
    _game_pos: dict[int, int] = field(default=None, repr=False)
    _profiles: dict[int, np.ndarray] = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "ContentBasedModel":
        tags = games["tags"].map(lambda t: json.loads(t) if isinstance(t, str) else t)
        docs = (
            tags.map(lambda ts: (" ".join(t.replace(" ", "_") for t in ts) + " ") * TAG_WEIGHT)
            + games["description"].fillna("")
        )
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        self._matrix = normalize(vectorizer.fit_transform(docs))
        self._game_ids = games["app_id"].to_numpy()
        self._game_pos = {gid: i for i, gid in enumerate(self._game_ids)}

        liked = train[train["is_recommended"]]
        self._profiles = {}
        for user_id, group in liked.groupby("user_id"):
            rows = [self._game_pos[g] for g in group["app_id"] if g in self._game_pos]
            if rows:
                profile = np.asarray(self._matrix[rows].mean(axis=0)).ravel()
                norm = np.linalg.norm(profile)
                if norm > 0:
                    self._profiles[user_id] = profile / norm
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        profile = self._profiles.get(user_id)
        if profile is None:
            return []
        scores = self._matrix @ profile
        order = np.argsort(-scores)
        result = []
        for idx in order:
            gid = int(self._game_ids[idx])
            if gid not in exclude:
                result.append(gid)
                if len(result) == n:
                    break
        return result

    def item_vectors(self, game_ids: list[int]) -> sparse.csr_matrix:
        rows = [self._game_pos[g] for g in game_ids if g in self._game_pos]
        return self._matrix[rows]
