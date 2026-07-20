from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_component_statuses():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["db"], bool)
    assert isinstance(body["redis"], bool)
    assert body["version"]
