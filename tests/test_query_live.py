"""Live multi-turn conversation test (LLM + embeddings). Opt-in:
    CV_LIVE_LLM=1 python -m pytest tests/test_query_live.py
Builds a real ready session from synthetic CVs, then drives the 7-turn HR flow
and asserts the conversation state composes across turns."""

import os

import pytest

from app.config import get_settings
from app.pipeline import acquire, dedup, extract, segment
from app.query import engine
from app.session_store import Session
from tests import make_fixtures

pytestmark = pytest.mark.skipif(
    os.environ.get("CV_LIVE_LLM") != "1",
    reason="set CV_LIVE_LLM=1 to run the live multi-turn test",
)


def _ready_session(tmp):
    make_fixtures.main(tmp)
    s = get_settings()
    names = ["cv_ahmed.pdf", "cv_sara.pdf", "cv_layla.pdf"]
    docs = [acquire.acquire_document(n, os.path.join(tmp, n), "pdf", s) for n in names]
    cands = segment.documents_to_candidates(docs)
    profs = [extract.extract_candidate(c, i + 1, s) for i, c in enumerate(cands)]
    dedup.mark_duplicates(profs)
    sess = Session(chat_id="t")
    sess.profiles = profs
    sess.roster_order = [p["candidate_id"] for p in profs]
    sess.view_state["current_set"] = [
        p["candidate_id"]
        for p in profs
        if p["extraction"]["status"] == "OK" and not p["duplicate_of"]
    ]
    sess.status = "ready"
    return sess, s


def test_seven_turn_conversation(tmp_path):
    sess, settings = _ready_session(str(tmp_path))
    assert len(sess.profiles) == 3

    assert engine.answer(sess, "How many candidates?", settings)["intent"] == "count"

    r = engine.answer(sess, "Recommend the best candidates for HR Specialist", settings)
    assert r["intent"] == "rank" and len(sess.view_state["last_ranking"]) == 3

    r = engine.answer(sess, "Who has SAP experience?", settings)
    assert r["intent"] == "who_has" and len(r["candidate_ids"]) == 3  # all three have SAP

    r = engine.answer(sess, "Compare candidate 1 and candidate 2", settings)
    assert r["intent"] == "compare" and len(r["candidate_ids"]) == 2

    r = engine.answer(sess, "Exclude candidates with less than 5 years of experience", settings)
    assert r["intent"] == "exclude"
    assert len(engine._current_ids(sess)) == 2  # Layla (~3y) removed from the working set

    r = engine.answer(sess, "Give me the final shortlist with reasons and evidence", settings)
    assert r["intent"] == "shortlist" and len(r["candidate_ids"]) == 2
    # rationale carries grounded skill evidence
    assert "p.1" in r["answer"]
