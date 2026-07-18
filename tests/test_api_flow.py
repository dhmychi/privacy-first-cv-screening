"""End-to-end HTTP API flow in mock mode: the real service surface (auth,
multipart ZIP upload -> ingest, async status polling, scoring, fairness audit,
reset). Exercises the paths a unit test cannot: routing, auth, file handling and
the background worker - all with zero models."""

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

KEY = "secret-test-key"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CV_LLM_MODE", "mock")
    monkeypatch.setenv("CV_API_KEY", KEY)
    monkeypatch.setenv("CV_DATA_DIR", "")


@pytest.fixture
def client(monkeypatch):
    from app import main

    monkeypatch.setattr(main._STORE, "_dir", "")  # no on-disk persistence in the test
    return TestClient(main.app)


def _cvs_zip() -> bytes:
    from tests import make_fixtures

    d = tempfile.mkdtemp()
    make_fixtures.main(d)
    with open(os.path.join(d, "cvs.zip"), "rb") as f:
        return f.read()


def _wait_ready(client, cid, timeout=20.0):
    st = {}
    for _ in range(int(timeout / 0.2)):
        st = client.get(f"/sessions/{cid}/status", headers=HEADERS).json()
        if st.get("status") in ("ready", "error"):
            return st
        time.sleep(0.2)
    return st


def test_health_is_open_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "cv-screening" and r.json()["llm_mode"] == "mock"


def test_auth_required_on_protected_endpoints(client):
    assert client.get("/sessions/x/status").status_code == 401
    assert client.get("/sessions/x/status", headers={"X-API-Key": "wrong"}).status_code == 401


def test_full_flow_analyze_status_score_audit_reset(client):
    zip_bytes = _cvs_zip()

    # 1. analyze a multipart ZIP (exercises ingest + async worker)
    r = client.post(
        "/sessions/flow/analyze",
        headers=HEADERS,
        files={"files": ("cvs.zip", zip_bytes, "application/zip")},
    )
    assert r.status_code == 202

    # 2. poll status until the background worker publishes the batch
    st = _wait_ready(client, "flow")
    assert st["status"] == "ready", f"analysis did not finish: {st}"
    assert st["candidate_count"] >= 2
    assert st["llm_mode"] == "mock"
    assert "Demo (mock) mode" in st["roster_markdown"]  # mock clearly labelled
    cids = [c["candidate_id"] for c in st["roster"]]
    assert len(cids) >= 2

    # 3. score against a JD -> matched/missing + ranking
    jd = "Recruiter\nRequired skills: Recruitment, Onboarding\nMinimum 3 years experience."
    sc = client.post("/sessions/flow/score", headers=HEADERS, json={"jd_text": jd}).json()
    assert sc["scorable"] is True and sc["ranking"]
    assert sc["llm_mode"] == "mock"

    # 4a. audit with no body -> always returns the protected-leakage report
    a1 = client.post("/sessions/flow/audit", headers=HEADERS, json={}).json()
    assert a1["leakage"]["clean"] is True  # no protected fields were extracted

    # 4b. audit with synthetic labels/selection -> EEOC four-fifths analysis.
    # group A selected, group B not -> adverse impact on B.
    labels = {cids[0]: {"grp": "A"}, cids[1]: {"grp": "B"}}
    selected = {cids[0]: True, cids[1]: False}
    a2 = client.post(
        "/sessions/flow/audit", headers=HEADERS, json={"labels": labels, "selected": selected}
    ).json()
    ff = a2["adverse_impact"]
    assert ff["adverse_impact"] is True
    assert ff["axes"]["grp"]["groups"]["B"]["impact_ratio"] < 0.8

    # 5. reset drops the session (privacy)
    rr = client.post("/sessions/flow/reset", headers=HEADERS)
    assert rr.status_code == 200 and rr.json()["reset"] is True
    gone = client.get("/sessions/flow/status", headers=HEADERS)
    assert gone.status_code == 404


def test_unsupported_upload_is_handled(client):
    # a file with no supported CV content must not crash the worker
    r = client.post(
        "/sessions/bad/analyze",
        headers=HEADERS,
        files={"files": ("note.txt", b"just some notes", "text/plain")},
    )
    assert r.status_code == 202
    st = _wait_ready(client, "bad")
    # unsupported content -> the batch finishes in an error/empty state, gracefully
    assert st["status"] in ("error", "ready")
