"""P0 smoke tests: the service imports cleanly, /health reports OCR + model
config, auth is enforced, and session endpoints behave before any batch is
uploaded. Honors CV_API_KEY when the container sets one."""

import os

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
KEY = os.environ.get("CV_API_KEY", "")
H = {"X-API-Key": KEY} if KEY else {}


def test_health_ok():
    r = client.get("/health")  # open endpoint, no auth
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "cv-screening"
    assert body["status"] == "ok"
    assert "langs" in body["ocr"]
    assert body["models"]["llm"]
    assert "active" in body["sessions"]


def test_status_404_without_session():
    r = client.get("/sessions/chat-none/status", headers=H)
    assert r.status_code == 404


def test_reset_is_idempotent():
    r = client.post("/sessions/chat-none/reset", headers=H)
    assert r.status_code == 200
    assert r.json() == {"reset": False}


def test_analyze_rejects_empty_batch():
    # No supported files -> 4xx, not a crash.
    r = client.post("/sessions/chat-x/analyze", headers=H, files={}, data={})
    assert r.status_code in (400, 415, 422)


def test_auth_enforced_when_key_set():
    # When a key is configured, protected endpoints reject a missing key.
    if KEY:
        assert client.get("/sessions/whatever/status").status_code == 401
    else:
        # No key configured -> protected endpoints are open.
        assert client.get("/sessions/whatever/status").status_code in (404, 200)
