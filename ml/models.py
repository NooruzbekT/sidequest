"""Модели рекомендаций для offline-evaluation.

Все модели обучаются строго на train-данных (агрегаты каталога содержат отзывы
тестового периода — утечка). Popularity — счётчики по train; content-based — TF-IDF
по тегам и описанию; item-item — косинус со-встречаемости лайков; ALS — матричная
факторизация implicit; hybrid — взвешенная смесь item-item и decay-популярности;
hybrid-artifact — тот же гибрид, но по усечённому serving-артефакту.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TAG_WEIGHT = 3  # теги информативнее свободного текста описания
DECAY_HALF_LIFE_DAYS = 180


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


@dataclass
class RandomModel:
    name: str = "random"
    seed: int = 42
    _game_ids: np.ndarray = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "RandomModel":
        self._game_ids = games["app_id"].to_numpy()
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        rng = np.random.default_rng(self.seed + int(user_id))
        order = rng.permutation(len(self._game_ids))
        result = []
        for idx in order:
            gid = int(self._game_ids[idx])
            if gid not in exclude:
                result.append(gid)
                if len(result) == n:
                    break
        return result


@dataclass
class DecayPopularityModel:
    """Популярность с экспоненциальным затуханием: свежие train-отзывы весят больше.

    w = 0.5 ** (возраст_дней / half_life), возраст считается от cutoff сплита.
    """

    name: str = "popularity-decay"
    half_life_days: int = DECAY_HALF_LIFE_DAYS
    _ranked: list[int] = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "DecayPopularityModel":
        cutoff = pd.Timestamp(train["date"].max()) + pd.Timedelta(days=1)
        age_days = (cutoff - pd.to_datetime(train["date"])).dt.days
        w = np.power(0.5, age_days / self.half_life_days)
        df = pd.DataFrame(
            {"app_id": train["app_id"], "w": w, "w_pos": w * train["is_recommended"]}
        )
        grp = df.groupby("app_id").sum()
        scores = (grp["w_pos"] / grp["w"]) * np.log1p(grp["w"])
        self._ranked = scores.sort_values(ascending=False).index.tolist()
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        result = []
        for app_id in self._ranked:
            if app_id not in exclude:
                result.append(app_id)
                if len(result) == n:
                    break
        return result


def build_user_item(train: pd.DataFrame, game_ids: np.ndarray):
    """Бинарная user×item матрица по train-лайкам + индексы."""
    liked = train[train["is_recommended"]]
    users = np.sort(liked["user_id"].unique())
    user_pos = {u: i for i, u in enumerate(users)}
    game_pos = {g: i for i, g in enumerate(game_ids)}
    rows = liked["user_id"].map(user_pos).to_numpy()
    cols = liked["app_id"].map(game_pos).to_numpy()
    matrix = sparse.csr_matrix(
        (np.ones(len(liked)), (rows, cols)), shape=(len(users), len(game_ids))
    )
    return matrix, user_pos, game_pos


@dataclass
class ItemItemModel:
    """Item-item collaborative: косинус со-встречаемости лайков в train."""

    name: str = "item-item"
    _sim: sparse.csr_matrix = field(default=None, repr=False)
    _game_ids: np.ndarray = field(default=None, repr=False)
    _game_pos: dict[int, int] = field(default=None, repr=False)
    _user_items: dict[int, list[int]] = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "ItemItemModel":
        self._game_ids = games["app_id"].to_numpy()
        matrix, user_pos, self._game_pos = build_user_item(train, self._game_ids)
        item_norm = normalize(matrix.T.tocsr())
        self._sim = (item_norm @ item_norm.T).tocsr()
        self._sim.setdiag(0)
        liked = train[train["is_recommended"]]
        self._user_items = (
            liked.groupby("user_id")["app_id"]
            .agg(lambda s: [self._game_pos[g] for g in s])
            .to_dict()
        )
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        items = self._user_items.get(user_id)
        if not items:
            return []
        scores = np.asarray(self._sim[items].sum(axis=0)).ravel()
        order = np.argsort(-scores)
        result = []
        for idx in order:
            if scores[idx] <= 0:
                break
            gid = int(self._game_ids[idx])
            if gid not in exclude:
                result.append(gid)
                if len(result) == n:
                    break
        return result


@dataclass
class ALSModel:
    """Матричная факторизация (implicit ALS) по бинарным train-лайкам."""

    name: str = "als"
    factors: int = 64
    iterations: int = 20
    regularization: float = 0.05
    seed: int = 42
    _model: implicit.als.AlternatingLeastSquares = field(default=None, repr=False)
    _matrix: sparse.csr_matrix = field(default=None, repr=False)
    _game_ids: np.ndarray = field(default=None, repr=False)
    _user_pos: dict[int, int] = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "ALSModel":
        self._game_ids = games["app_id"].to_numpy()
        self._matrix, self._user_pos, _ = build_user_item(train, self._game_ids)
        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.seed,
        )
        self._model.fit(self._matrix, show_progress=False)
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        pos = self._user_pos.get(user_id)
        if pos is None:
            return []
        ids, _ = self._model.recommend(
            pos, self._matrix[pos], N=n + len(exclude), filter_already_liked_items=True
        )
        result = []
        for idx in ids:
            gid = int(self._game_ids[idx])
            if gid not in exclude:
                result.append(gid)
                if len(result) == n:
                    break
        return result


@dataclass
class HybridModel:
    """Взвешенный гибрид, формула идентична serving (train/serving parity):

    score = w_sim * sim_norm + w_pop * pop_norm, где sim — сумма item-item
    косинусов к лайкам пользователя, pop — decay-популярность по позиции в ранге.
    Без лайков — чистая популярность (cold start).
    """

    name: str = "hybrid"
    w_sim: float = 0.6
    w_pop: float = 0.4
    _item_item: ItemItemModel = field(default=None, repr=False)
    _pop_norm: dict[int, float] = field(default=None, repr=False)
    _game_ids: np.ndarray = field(default=None, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "HybridModel":
        self._item_item = ItemItemModel().fit(games, train)
        decay = DecayPopularityModel().fit(games, train)
        n = len(decay._ranked)
        self._pop_norm = {gid: 1 - rank / n for rank, gid in enumerate(decay._ranked)}
        self._game_ids = games["app_id"].to_numpy()
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        items = self._item_item._user_items.get(user_id)
        sims: dict[int, float] = {}
        if items:
            scores = np.asarray(self._item_item._sim[items].sum(axis=0)).ravel()
            max_sim = scores.max()
            if max_sim > 0:
                for idx in np.nonzero(scores)[0]:
                    sims[int(self._game_ids[idx])] = scores[idx] / max_sim

        fused = []
        for gid in self._game_ids:
            gid = int(gid)
            if gid in exclude:
                continue
            score = self.w_sim * sims.get(gid, 0.0) + self.w_pop * self._pop_norm.get(gid, 0.0)
            fused.append((-score, gid))
        fused.sort()
        return [gid for _, gid in fused[:n]]


@dataclass
class ArtifactHybridModel:
    """Гибрид, считающий по serving-артефакту (top-20 усечённые соседи).

    Метрика этой модели описывает то, что реально отвечает пользователям,
    в отличие от HybridModel с полной матрицей похожестей.
    """

    name: str = "hybrid-artifact"
    artifact_path: str = "ml/artifacts/hybrid_current.json"
    _pop: dict[int, float] = field(default=None, repr=False)
    _neighbors: dict[int, dict[int, float]] = field(default=None, repr=False)
    _game_ids: np.ndarray = field(default=None, repr=False)
    _user_likes: dict[int, list[int]] = field(default=None, repr=False)
    _w_sim: float = field(default=0.6, repr=False)
    _w_pop: float = field(default=0.4, repr=False)

    def fit(self, games: pd.DataFrame, train: pd.DataFrame) -> "ArtifactHybridModel":
        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root / self.artifact_path).read_text(encoding="utf-8"))
        self._w_sim = raw["params"]["weights"]["similarity"]
        self._w_pop = raw["params"]["weights"]["popularity"]
        self._pop = {int(k): float(v) for k, v in raw["popularity"].items()}
        self._neighbors = {
            int(k): {int(g): float(s) for g, s in pairs} for k, pairs in raw["neighbors"].items()
        }
        self._game_ids = games["app_id"].to_numpy()
        liked = train[train["is_recommended"]]
        self._user_likes = liked.groupby("user_id")["app_id"].agg(list).to_dict()
        return self

    def recommend(self, user_id: int, n: int, exclude: set[int]) -> list[int]:
        sims: dict[int, float] = {}
        for liked in self._user_likes.get(user_id, []):
            for gid, s in self._neighbors.get(liked, {}).items():
                sims[gid] = sims.get(gid, 0.0) + s
        max_sim = max(sims.values()) if sims else 0.0

        fused = []
        for gid in self._game_ids:
            gid = int(gid)
            if gid in exclude:
                continue
            sim_norm = sims.get(gid, 0.0) / max_sim if max_sim > 0 else 0.0
            score = self._w_sim * sim_norm + self._w_pop * self._pop.get(gid, 0.0)
            fused.append((-score, gid))
        fused.sort()
        return [gid for _, gid in fused[:n]]
