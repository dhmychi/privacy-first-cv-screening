"""Live extraction integration test (hits the local LLM). Opt-in:
    CV_LIVE_LLM=1 python -m pytest tests/test_extract_live.py
Skipped by default so the offline suite stays fast and network-free."""

import os

import pytest

from app.config import get_settings
from app.pipeline import acquire, extract, segment
from tests import make_fixtures

pytestmark = pytest.mark.skipif(
    os.environ.get("CV_LIVE_LLM") != "1",
    reason="set CV_LIVE_LLM=1 to run live-LLM extraction",
)


def _ahmed_profile(tmp):
    make_fixtures.main(tmp)
    s = get_settings()
    doc = acquire.acquire_document("cv_ahmed.pdf", os.path.join(tmp, "cv_ahmed.pdf"), "pdf", s)
    cand = segment.documents_to_candidates([doc])[0]
    return extract.extract_candidate(cand, 1, s)


def test_live_extraction_quality(tmp_path):
    p = _ahmed_profile(str(tmp_path))

    # identity (deterministic anchors + LLM name)
    assert p["identity"]["full_name"]["value"] == "Ahmed Ali"
    assert "ahmed.ali@example.com" in p["identity"]["emails"]
    assert any("966" in ph for ph in p["identity"]["phones"])

    # experiences present, each with a quote that exists on the cited page
    assert len(p["experiences"]) >= 2
    for e in p["experiences"]:
        assert e["evidence"] and e["evidence"]["page"] == 1

    # years computed deterministically (never from the model's own number)
    ty = p["total_years_experience"]
    assert ty["value"] and "computed" in ty["basis"]

    # skills grounded: every 'stated' skill carries real evidence
    names = {s["name"].lower() for s in p["skills"]}
    assert "sap successfactors" in names
    for s in p["skills"]:
        if s["source"] == "stated":
            assert s["evidence"] is not None

    # clean gating, protected attributes excluded by policy
    assert p["extraction"]["status"] == "OK"
    assert "nationality" in p["_excluded_by_policy"]
