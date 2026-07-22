import pandas as pd

from models import DecayPopularityModel, HybridModel, ItemItemModel

GAMES = pd.DataFrame(
    {
        "app_id": [1, 2, 3, 4],
        "title": ["A", "B", "C", "D"],
        "description": ["", "", "", ""],
        "tags": ['["Action"]', '["Action"]', '["Puzzle"]', '["Puzzle"]'],
    }
)


def make_train(rows):
    return pd.DataFrame(rows, columns=["user_id", "app_id", "is_recommended", "date"])


def test_item_item_recommends_co_liked_games():
    # игроки 10 и 11 лайкают {1,2}: у пользователя 12 с лайком 1 сосед — 2
    train = make_train(
        [
            (10, 1, True, "2021-01-01"),
            (10, 2, True, "2021-01-02"),
            (11, 1, True, "2021-01-03"),
            (11, 2, True, "2021-01-04"),
            (12, 1, True, "2021-01-05"),
            (12, 3, False, "2021-01-06"),
        ]
    )
    model = ItemItemModel().fit(GAMES, train)
    recs = model.recommend(12, n=2, exclude={1})
    assert recs[0] == 2


def test_item_item_unknown_user_returns_empty():
    train = make_train([(10, 1, True, "2021-01-01")])
    model = ItemItemModel().fit(GAMES, train)
    assert model.recommend(999, n=5, exclude=set()) == []


def test_decay_popularity_prefers_recent_games():
    # у игры 2 отзывы свежее при равном количестве — она должна стоять выше
    train = make_train(
        [
            (10, 1, True, "2015-01-01"),
            (11, 1, True, "2015-06-01"),
            (12, 2, True, "2021-11-01"),
            (13, 2, True, "2021-12-01"),
        ]
    )
    model = DecayPopularityModel().fit(GAMES, train)
    recs = model.recommend(99, n=2, exclude=set())
    assert recs[0] == 2


def test_hybrid_returns_top_n_without_duplicates():
    train = make_train(
        [
            (10, 1, True, "2021-01-01"),
            (10, 2, True, "2021-01-02"),
            (11, 2, True, "2021-01-03"),
            (11, 3, True, "2021-01-04"),
            (12, 1, True, "2021-01-05"),
        ]
    )
    model = HybridModel().fit(GAMES, train)
    recs = model.recommend(12, n=3, exclude={1})
    assert len(recs) == len(set(recs)) <= 3
    assert 1 not in recs
