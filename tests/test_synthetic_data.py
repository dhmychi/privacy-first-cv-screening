"""Capability tests over the fully-synthetic AR/EN corpus (scripts/
generate_synthetic_data.py). Runs entirely in mock mode - no models, no network.
Covers: Arabic / English / multilingual extraction, OCR-failure degradation,
duplicate detection, same-name separation, prompt-injection detection, evidence
grounding, matched/missing scoring, protected-attribute exclusion, fairness
(four-fifths) math, human-review gating, and deterministic outputs."""

import importlib.util
import json
import os

import pytest

_GEN = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_synthetic_data.py")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    spec = importlib.util.spec_from_file_location("gen", _GEN)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.main(str(d))
    return str(d)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CV_LLM_MODE", "mock")
    monkeypatch.setenv("CV_API_KEY", "test-key")
    monkeypatch.setenv("CV_DATA_DIR", "")


def _prof(corpus, fn, idx=1):
    from app.config import get_settings
    from app.pipeline import acquire, extract, segment

    s = get_settings()
    kind = "docx" if fn.endswith(".docx") else "pdf"
    doc = acquire.acquire_document(fn, os.path.join(corpus, "cvs", fn), kind, s)
    return extract.extract_candidate(segment.documents_to_candidates([doc])[0], idx, s)


def test_arabic_extraction(corpus):
    p = _prof(corpus, "ar_senior_ai.docx")
    assert p["identity"]["full_name"]["value"] == "سلمان الحربي"
    names = {s["name"].lower() for s in p["skills"]}
    assert {"python", "fastapi", "rag"} <= names
    assert (p["total_years_experience"]["value"] or 0) >= 6
    assert p["extraction"]["status"] == "OK"


def test_english_extraction(corpus):
    p = _prof(corpus, "en_mid.pdf")
    assert p["identity"]["full_name"]["value"] == "Jordan Miller"
    names = {s["name"].lower() for s in p["skills"]}
    assert {"python", "fastapi", "sql"} <= names
    assert any("202" in ph for ph in p["identity"]["phones"])


def test_multilingual_extraction(corpus):
    p = _prof(corpus, "mixed_ar_en.docx")
    names = {s["name"].lower() for s in p["skills"]}
    assert "python" in names and "fastapi" in names
    # a bilingual CV yields both scripts: at least one Arabic-script skill token
    assert any(any("؀" <= ch <= "ۿ" for ch in s["name"]) for s in p["skills"])


def test_ocr_failure_degrades_to_review(corpus):
    # image-only PDF; with OCR unavailable it must degrade to NEEDS_REVIEW, not crash
    p = _prof(corpus, "scanned_poor.pdf")
    assert p["extraction"]["status"] == "NEEDS_REVIEW"


def test_duplicate_detection(corpus):
    from app.pipeline import dedup

    ps = [_prof(corpus, "en_mid.pdf", 1), _prof(corpus, "duplicate_en_mid.pdf", 2)]
    dedup.mark_duplicates(ps)
    assert ps[1]["duplicate_of"] == ps[0]["candidate_id"]


def test_same_name_not_merged(corpus):
    from app.pipeline import dedup

    ps = [_prof(corpus, "samename_1.docx", 1), _prof(corpus, "samename_2.docx", 2)]
    dedup.mark_duplicates(ps)
    assert ps[0]["duplicate_of"] is None and ps[1]["duplicate_of"] is None


def test_prompt_injection_flagged(corpus):
    p = _prof(corpus, "injection.docx")
    assert "INJECTION_SUSPECTED" in p["extraction"]["reasons"]


def test_evidence_grounding(corpus):
    p = _prof(corpus, "en_mid.pdf")
    assert p["experiences"]
    for e in p["experiences"]:
        assert e["evidence"] and e["evidence"].get("page")
    assert any(s.get("evidence") for s in p["skills"] if s["source"] == "stated")


def test_protected_attribute_exclusion(corpus):
    from app import fairness

    p = _prof(corpus, "protected_attrs.docx")
    # (1) policy bookkeeping lists the protected attributes as excluded
    excl = set(p.get("_excluded_by_policy", []))
    assert {"gender", "age", "nationality", "religion", "marital", "photo"} <= excl
    # (2) the product's own leakage scan: no protected FIELD is populated
    assert fairness.scan_profile_leakage([p]) == []
    # (3) protected VALUES never reach the surfaced (non-private) profile fields.
    #     (Raw source text is kept only under private _-prefixed keys for evidence
    #     grounding and is never surfaced or scored as a structured attribute.)
    surfaced = {k: v for k, v in p.items() if not k.startswith("_")}
    blob = json.dumps(surfaced, ensure_ascii=False).lower()
    for leak in ["utopian", "female", "married", "1990-02-11"]:
        assert leak not in blob, f"protected value leaked into surfaced profile: {leak}"


def test_human_review_gating(corpus):
    p = _prof(corpus, "sparse.pdf")
    assert p["extraction"]["status"] == "NEEDS_REVIEW"


def test_matched_and_missing_against_ai_jd(corpus):
    from app.config import get_settings
    from app.scoring import service
    from app.session_store import Session

    s = get_settings()
    with open(os.path.join(corpus, "jds", "jd_ai_engineer.txt"), encoding="utf-8") as f:
        jd = f.read()
    sess = Session(chat_id="syn")
    sess.profiles = [_prof(corpus, "ar_senior_ai.docx", 1)]
    res = service.score_session(sess, jd, s)
    assert res["scorable"] is True
    assert res["ranking"][0]["score"] > 0
    md = res["report_md"]
    assert "Python" in md and "FastAPI" in md


def test_fairness_four_fifths(corpus):
    from app import fairness

    groups = {
        "c1": {"group": "A"},
        "c2": {"group": "A"},
        "c3": {"group": "B"},
        "c4": {"group": "B"},
    }
    # group A all selected, group B none -> adverse impact
    adverse = fairness.four_fifths_audit({"c1": True, "c2": True, "c3": False, "c4": False}, groups)
    assert adverse["adverse_impact"] is True
    assert adverse["axes"]["group"]["groups"]["B"]["impact_ratio"] < 0.8
    # balanced selection -> no adverse impact
    fair = fairness.four_fifths_audit({"c1": True, "c2": False, "c3": True, "c4": False}, groups)
    assert fair["adverse_impact"] is False


def test_deterministic_outputs(corpus):
    a = _prof(corpus, "ar_senior_ai.docx", 1)
    b = _prof(corpus, "ar_senior_ai.docx", 1)
    assert a["skills"] == b["skills"]
    assert a["experiences"] == b["experiences"]
    assert a["total_years_experience"] == b["total_years_experience"]


def test_corpus_manifest_and_bundles(corpus):
    with open(os.path.join(corpus, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["cvs"]) >= 16 and len(manifest["jds"]) == 3
    for jd in ("jd_ai_engineer.txt", "jd_en.txt", "jd_ar.txt"):
        assert os.path.exists(os.path.join(corpus, "jds", jd))
    assert os.path.exists(os.path.join(corpus, "bundles", "demo_full_batch.zip"))
