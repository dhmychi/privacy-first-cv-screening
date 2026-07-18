"""Safe-logging guarantees: operational logs carry only non-identifying
metadata - never candidate PII or CV content."""

import json
import logging
import os

import pytest

from app import logging_safe

_ALLOWED = {
    "ts",
    "event",
    "session",
    "job",
    "files",
    "status",
    "duration_ms",
    "error_type",
    "stage",
}


def test_only_allowlisted_fields_are_emitted():
    rec = logging_safe.log_event(
        "analyze_done",
        session_id="chat-abc",
        job_id="job123",
        file_count=3,
        status="ready",
        duration_ms=1234.7,
        stage="ready",
    )
    assert set(rec) <= _ALLOWED
    assert rec["files"] == 3 and rec["duration_ms"] == 1234
    assert rec["event"] == "analyze_done" and rec["status"] == "ready"


def test_session_id_is_hashed_never_raw():
    raw = "chat-for-ahmed.ali@example.com"
    rec = logging_safe.log_event("evt", session_id=raw)
    assert rec["session"].startswith("s_")
    assert "ahmed" not in rec["session"] and "@" not in rec["session"]
    assert raw not in json.dumps(rec)


def test_slug_fields_strip_free_text_and_pii():
    rec = logging_safe.log_event("evt", status="ready name=Ahmed Ali <ahmed@x.com>")
    assert " " not in rec["status"] and "@" not in rec["status"] and "<" not in rec["status"]


def test_error_type_takes_class_name_not_message():
    try:
        raise ValueError("candidate Ahmed Ali email ahmed@example.com")
    except ValueError as e:
        rec = logging_safe.log_event("analyze_error", error_type=type(e).__name__)
    assert rec["error_type"] == "ValueError"
    assert "Ahmed" not in json.dumps(rec) and "@" not in json.dumps(rec)


def test_no_arbitrary_kwargs_accepted():
    # there is deliberately no parameter that can carry CV content
    with pytest.raises(TypeError):
        logging_safe.log_event("evt", cv_text="Ahmed Ali, ahmed@example.com")


def test_full_analyze_run_logs_contain_no_pii(tmp_path, monkeypatch, caplog):
    """Run the real analysis worker (mock mode) on a CV with known PII and assert
    none of it appears in any emitted log line."""
    monkeypatch.setenv("CV_LLM_MODE", "mock")
    monkeypatch.setenv("CV_API_KEY", "test-key")
    monkeypatch.setenv("CV_DATA_DIR", "")

    from app import main
    from app.config import get_settings
    from app.session_store import Session
    from tests import make_fixtures

    make_fixtures.main(str(tmp_path))
    ahmed = os.path.join(str(tmp_path), "cv_ahmed.pdf")
    upl = str(tmp_path / "upl")
    os.makedirs(upl, exist_ok=True)

    pii_markers = ["Ahmed", "Ali", "acmecorp", "4567", "@", "Riyadh", "Almarai"]
    sess = Session(chat_id="chat-secret-xyz")
    with caplog.at_level(logging.INFO, logger="cv_screening"):
        main._run_analysis(sess, [("cv_ahmed.pdf", ahmed)], upl, get_settings())

    assert sess.status == "ready", f"analysis did not complete: {sess.error}"
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "analyze_start" in blob and "analyze_done" in blob
    for pii in pii_markers:
        assert pii not in blob, f"PII leaked into logs: {pii!r}\nLOGS:\n{blob}"
