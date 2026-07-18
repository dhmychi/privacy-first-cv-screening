"""Async analyze + progress (live; opt-in CV_LIVE_LLM=1). POST /analyze returns
202 + job_id immediately; polling /status reaches 'ready' with the roster."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests import make_fixtures

pytestmark = pytest.mark.skipif(
    os.environ.get("CV_LIVE_LLM") != "1",
    reason="set CV_LIVE_LLM=1 to run the live async test",
)

client = TestClient(app)
KEY = os.environ.get("CV_API_KEY", "")
H = {"X-API-Key": KEY} if KEY else {}


def test_async_analyze_then_poll(tmp_path):
    make_fixtures.main(str(tmp_path))
    paths = [os.path.join(str(tmp_path), n) for n in ("cv_ahmed.pdf", "cv_sara.pdf")]
    handles = [open(p, "rb") for p in paths]
    try:
        files = [
            ("files", (os.path.basename(p), h, "application/pdf"))
            for p, h in zip(paths, handles, strict=False)
        ]
        r = client.post("/sessions/atest/analyze", headers=H, files=files)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] in ("queued", "running") and body["job_id"]
    finally:
        for h in handles:
            h.close()

    status = {}
    for _ in range(90):
        status = client.get("/sessions/atest/status", headers=H).json()
        if status["status"] in ("ready", "error"):
            break
        time.sleep(1)
    assert status["status"] == "ready", status
    assert status["candidate_count"] == 2
    assert "roster" in status and len(status["roster"]) == 2
