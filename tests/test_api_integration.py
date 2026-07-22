"""Интеграционные тесты: реальное приложение + реальный Postgres.

Локально нужен поднятый docker compose (БД на 127.0.0.1:5433 с demo-данными);
если база недоступна — тесты пропускаются. В CI база и demo-импорт готовятся шагами
workflow.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text

from app.config import settings
from app.main import app

pytestmark = pytest.mark.integration


def _db_available() -> bool:
    try:
        engine = create_engine(settings.dsn, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM games")).scalar() > 0
    except Exception:
        return False


if not _db_available():
    pytest.skip("БД с demo-данными недоступна", allow_module_level=True)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_full_user_flow_with_duplicate_feedback(client):
    resp = await client.post("/users", json={"name": "integration"})
    assert resp.status_code == 201
    user_id = resp.json()["id"]

    # пустые стоп-теги запрещены валидацией
    resp = await client.post(
        f"/users/{user_id}/preferences",
        json={"genres": ["RPG"], "max_price": 30, "blocked_tags": []},
    )
    assert resp.status_code == 422

    resp = await client.post(
        f"/users/{user_id}/preferences",
        json={"genres": ["RPG"], "max_price": 30, "blocked_tags": ["Horror"]},
    )
    assert resp.status_code == 200

    resp = await client.get("/games", params={"query": "", "limit": 5})
    assert resp.status_code == 200
    game_id = resp.json()[0]["id"]

    first = await client.post(
        f"/users/{user_id}/interactions", json={"game_id": game_id, "kind": "like"}
    )
    repeat = await client.post(
        f"/users/{user_id}/interactions", json={"game_id": game_id, "kind": "like"}
    )
    assert first.json()["duplicate"] is False
    assert repeat.json()["duplicate"] is True


@pytest.mark.asyncio
async def test_recommendations_respect_filters_and_report_model(client):
    resp = await client.post("/users", json={"name": "integration-recs"})
    user_id = resp.json()["id"]
    await client.post(
        f"/users/{user_id}/preferences",
        json={"genres": ["Strategy"], "max_price": 20, "blocked_tags": ["Horror"]},
    )

    resp = await client.get(f"/users/{user_id}/recommendations")
    assert resp.status_code == 200
    body = resp.json()
    # без артефакта на месте сервис обязан честно ответить baseline-моделью
    assert body["model_name"] in {"hybrid", "baseline"}
    assert 1 <= len(body["items"]) <= 10
    for item in body["items"]:
        assert item["game"]["price"] <= 20
        assert "horror" not in {t.lower() for t in item["game"]["tags"]}
        assert item["reason"]


@pytest.mark.asyncio
async def test_metrics_endpoint_reports_counters(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert {"recommendation_requests", "errors", "served_by", "active_model"} <= set(body)
