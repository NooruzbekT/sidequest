from app.services.baseline import Candidate, rank_popular


def make_candidate(game_id: int, **kwargs) -> Candidate:
    defaults = {
        "title": f"Game {game_id}",
        "tags": ["Action"],
        "price": 10.0,
        "positive_ratio": 90,
        "user_reviews": 1000,
    }
    defaults.update(kwargs)
    return Candidate(game_id=game_id, **defaults)


def rank(candidates, **kwargs):
    defaults = {
        "max_price": None,
        "blocked_tags": [],
        "preferred_genres": [],
        "exclude_game_ids": set(),
    }
    defaults.update(kwargs)
    return rank_popular(candidates, **defaults)


def test_price_filter_excludes_expensive_games():
    candidates = [make_candidate(1, price=60.0), make_candidate(2, price=20.0)]
    result = rank(candidates, max_price=25.0)
    assert [s.game_id for s in result] == [2]


def test_price_filter_keeps_free_games():
    result = rank([make_candidate(1, price=0.0)], max_price=5.0)
    assert len(result) == 1


def test_blocked_tags_exclude_games_case_insensitive():
    candidates = [
        make_candidate(1, tags=["Horror", "Action"]),
        make_candidate(2, tags=["Adventure"]),
    ]
    result = rank(candidates, blocked_tags=["horror"])
    assert [s.game_id for s in result] == [2]

def test_interacted_games_excluded():
    candidates = [make_candidate(1), make_candidate(2)]
    result = rank(candidates, exclude_game_ids={1})
    assert [s.game_id for s in result] == [2]


def test_genre_match_boosts_score_and_appears_in_reason():
    candidates = [
        make_candidate(1, tags=["RPG"], positive_ratio=80, user_reviews=1000),
        make_candidate(2, tags=["Action"], positive_ratio=80, user_reviews=1000),
    ]
    result = rank(candidates, preferred_genres=["RPG"])
    assert result[0].game_id == 1
    assert "rpg" in result[0].reason


def test_top_n_limit_and_sorted_by_score():
    candidates = [make_candidate(i, user_reviews=100 * i) for i in range(1, 30)]
    result = rank(candidates)
    assert len(result) == 10
    assert result[0].score >= result[-1].score


def test_no_candidates_after_filters_returns_empty():
    result = rank([make_candidate(1, price=100.0)], max_price=10.0)
    assert result == []


def test_reason_is_built_from_data():
    result = rank([make_candidate(1, positive_ratio=94, user_reviews=5000, price=0.0)])
    reason = result[0].reason
    assert "94%" in reason
    assert "5 000" in reason
    assert "бесплатная" in reason
