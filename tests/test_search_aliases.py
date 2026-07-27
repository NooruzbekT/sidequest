from app.routers.games import resolve_query


def test_known_aliases_resolve_case_insensitive():
    assert resolve_query("CS") == "counter-strike"
    assert resolve_query("  csgo ") == "counter-strike: global offensive"
    assert resolve_query("GTA") == "grand theft auto"


def test_unknown_query_passes_through():
    assert resolve_query("hollow knight") == "hollow knight"
