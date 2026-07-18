"""Privacy-first CV screening — FastAPI service.

Endpoints:
  GET  /health                       (open)
  POST /sessions/{chat_id}/analyze   (auth, multipart)  -> ingest a batch ONCE
  GET  /sessions/{chat_id}/status    (auth)
  POST /sessions/{chat_id}/query     (auth)              -> multi-turn Q&A (P3)
  POST /sessions/{chat_id}/reset     (auth)              -> drop the session (privacy)

Design rules: candidate PII never leaves the box; analysis
gaps are returned as 200 results, not HTTP errors, so the calling client's
error handling never masks a safe "needs review" outcome. Only infra / input
failures use 4xx/5xx.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import __version__, fairness, logging_safe
from .config import get_settings, vision_is_local
from .extraction.ocr import tesseract_available, tesseract_langs
from .models import HealthResponse, QueryRequest, ScoreRequest, parse_options
from .pipeline import acquire, dedup, extract, ingest, segment
from .query import engine, render
from .scoring import rubric as scoring_rubric
from .scoring import service as scoring_service
from .session_store import SessionStore


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Startup guard: refuse to run if an off-box OCR endpoint is configured
    # without explicit approval — candidate PII must stay local.
    s = get_settings()
    url = s.vision_ocr_url
    if url and not s.allow_cloud_ocr and not vision_is_local(url):
        raise RuntimeError(
            "Refusing to start: CV_VISION_OCR_URL points off-box but "
            "CV_ALLOW_CLOUD_OCR is false. Candidate PII must stay local."
        )
    yield


app = FastAPI(title="Privacy-First CV Screening", version=__version__, lifespan=_lifespan)
_STORE = SessionStore(
    ttl_minutes=get_settings().session_ttl_minutes, data_dir=get_settings().data_dir
)
_EXEC = ThreadPoolExecutor(max_workers=2)  # background analysis workers


def require_key(x_api_key: str | None = Header(None)):
    s = get_settings()
    if s.api_key and x_api_key != s.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _safe_name(name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name or "file"))


def _safe_chat_id(chat_id: str) -> str:
    cid = re.sub(r"[^A-Za-z0-9._:-]", "_", chat_id or "")
    if not cid:
        raise HTTPException(status_code=400, detail="missing chat_id")
    return cid[:128]


@app.get("/health", response_model=HealthResponse)
def health():
    s = get_settings()
    return {
        "status": "ok",
        "service": "cv-screening",
        "version": __version__,
        "llm_mode": s.llm_mode,
        "ocr": {
            "tesseract": tesseract_available(),
            "langs": tesseract_langs(),
            "mode": s.ocr_mode,
            "vision_enabled": bool(s.vision_ocr_url),
            "cloud_ocr": s.allow_cloud_ocr,
        },
        "models": {"llm": s.llm_model, "embed": s.embed_model, "ollama": s.ollama_url},
        "sessions": {"active": _STORE.count(), "ttl_minutes": s.session_ttl_minutes},
        "time": _dt.datetime.now(_dt.UTC).isoformat(),
    }


@app.post("/sessions/{chat_id}/analyze")
async def analyze_endpoint(
    chat_id: str,
    files: list[UploadFile] = File(...),
    options: str = Form("{}"),
    signature: str = Form(""),
    mode: str = Form("replace"),
    _=Depends(require_key),
):
    """Start analysis of an uploaded batch (files and/or ZIP) for this chat and
    return IMMEDIATELY with a job_id. Poll GET /sessions/{chat_id}/status for live
    progress; when status is 'ready' the status response includes the roster +
    summary. Analysis runs in a background worker so a 25-200 CV batch never blocks
    the request.

    mode='replace' (default): analyze this upload as a fresh batch (previous batch
    for this chat is discarded). mode='append': MERGE this upload into the existing
    ready batch for this chat -- candidate numbering continues (e.g. 60 -> 61..80)
    and duplicate detection re-runs across the full merged batch. append falls back
    to a fresh analysis when there is no ready batch yet."""
    s = get_settings()
    cid = _safe_chat_id(chat_id)
    _ = parse_options(options)
    limit = s.max_upload_mb * 1024 * 1024
    tmp = tempfile.mkdtemp(prefix="cv_")
    raw = []  # [(name, path, content_hash)]
    try:
        for uf in files:
            data = await uf.read()
            if len(data) > limit:
                raise HTTPException(413, f"file too large: {uf.filename}")
            p = os.path.join(tmp, _safe_name(uf.filename))
            with open(p, "wb") as f:
                f.write(data)
            raw.append((uf.filename or "file", p, hashlib.sha256(data).hexdigest()))
    except HTTPException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if not raw:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(415, "no files uploaded")

    # Some chat UIs re-send every file attached earlier in the chat on each turn, so
    # ingest ONLY files this session has not already processed (by content hash).
    existing = _STORE.get(cid)
    processed = existing.processed_hashes if existing is not None else set()
    new_raw = [r for r in raw if r[2] not in processed]
    want_append = (
        (mode or "replace").lower() == "append"
        and existing is not None
        and existing.status == "ready"
        and bool(existing.profiles)
    )

    if want_append:
        assert existing is not None  # guaranteed by the want_append condition
        to_ingest = new_raw
        if not to_ingest:
            # only already-known files were re-sent; nothing new to add
            existing.client_signature = signature
            shutil.rmtree(tmp, ignore_errors=True)
            return JSONResponse(
                status_code=202,
                content={
                    "chat_id": cid,
                    "job_id": existing.job_id,
                    "status": "ready",
                    "mode": "append",
                    "message": "no new files to add; existing batch unchanged",
                    "status_url": f"/sessions/{cid}/status",
                },
            )
        base_no = max((p["display_no"] for p in existing.profiles), default=0)
        merged_hashes = set(processed) | {r[2] for r in to_ingest}
        existing.status = "running"
        existing.error = None
        existing.client_signature = signature
        existing.progress = {"done": 0, "total": len(to_ingest), "stage": "queued"}
        _EXEC.submit(
            _run_append,
            existing,
            [(n, p) for n, p, _h in to_ingest],
            tmp,
            s,
            base_no,
            merged_hashes,
        )
        return JSONResponse(
            status_code=202,
            content={
                "chat_id": cid,
                "job_id": existing.job_id,
                "status": "queued",
                "mode": "append",
                "message": "appending new files to existing batch; poll status",
                "status_url": f"/sessions/{cid}/status",
            },
        )

    # replace / reset: if an old batch exists AND genuinely-new files were uploaded,
    # analyze ONLY the new upload (discard the old batch, e.g. "reset and analyze only
    # this file"); otherwise analyze everything provided (fresh first upload).
    to_ingest = new_raw if (existing is not None and new_raw) else raw
    hashes = {r[2] for r in to_ingest}
    session = _STORE.create(cid)
    session.status = "queued"
    session.client_signature = signature
    session.progress = {"done": 0, "total": len(to_ingest), "stage": "queued"}
    _EXEC.submit(_run_analysis, session, [(n, p) for n, p, _h in to_ingest], tmp, s, hashes)

    return JSONResponse(
        status_code=202,
        content={
            "chat_id": cid,
            "job_id": session.job_id,
            "status": "queued",  # informational; poll /status for the live state
            "mode": "replace",
            "message": "analysis started; poll status for progress and results",
            "status_url": f"/sessions/{cid}/status",
        },
    )


def _process_documents(session, raw, tmp, settings, base_no=0):
    """PHASE-PARALLEL acquire -> vlm -> extract, structured around the single
    local GPU:

      phase 1  CPU  - text layer + OCR for ALL documents in parallel
                      (acquire_workers Tesseract subprocesses; no GPU use)
      phase 2  GPU  - VLM rescue of the pages OCR could not read, GROUPED so
                      the vision model runs alone (it must never compete with
                      the extraction model - measured to time out and silently
                      lose rescues when interleaved)
      phase 3  GPU  - LLM extraction with the same extract_workers concurrency
                      as always

    Same functions, same escalation conditions, same numbering (original file
    order, base_no+1..base_no+N) - only the scheduling changes, so outputs are
    equivalent while CPU work no longer serializes in front of GPU work.

    Returns (documents, profiles) in original file order, or (None, None) when
    the upload contained no supported files."""
    from concurrent.futures import ThreadPoolExecutor

    docs_in = ingest.expand_uploads(raw, tmp, settings)
    if not docs_in:
        return None, None
    n = len(docs_in)

    # phase 1: CPU-parallel acquisition (VLM deferred)
    session.progress = {"done": 0, "total": n, "stage": "reading"}
    documents = [None] * n
    with ThreadPoolExecutor(max_workers=min(max(1, settings.acquire_workers), n)) as ex:
        futs = {
            ex.submit(acquire.acquire_document, d["name"], d["path"], d["kind"], settings, True): i
            for i, d in enumerate(docs_in)
        }
        for f, i in futs.items():
            try:
                documents[i] = f.result()
            except Exception:  # noqa: BLE001 - one bad file must NOT kill the batch
                documents[i] = {
                    "file": docs_in[i]["name"],
                    "kind": docs_in[i]["kind"],
                    "page_count": 0,
                    "pages": [],
                    "ocr_page_count": 0,
                    "vlm_page_count": 0,
                    "mean_ocr_confidence": None,
                    "text_chars": 0,
                    "empty": True,
                    "unreadable": True,
                }
            session.progress["done"] = sum(1 for d in documents if d is not None)

    # phase 2: grouped VLM rescue (vision model resident once, no competition)
    pending = [i for i, d in enumerate(documents) if d.get("_vlm_pending")]
    if pending:
        session.progress = {"done": 0, "total": len(pending), "stage": "deep-reading"}
        for k, i in enumerate(pending):
            documents[i] = acquire.apply_vlm_rescue(documents[i], settings)
            session.progress["done"] = k + 1
    for d in documents:
        d.pop("_vlm_pending", None)
        d.pop("_path", None)

    # phase 3: extraction (unchanged concurrency and numbering)
    candidates = segment.documents_to_candidates(documents)
    session.progress = {"done": 0, "total": len(candidates), "stage": "extracting"}
    profiles = [None] * len(candidates)
    workers = max(1, min(settings.extract_workers, len(candidates) or 1))
    done = [0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(extract.extract_candidate, c, base_no + i + 1, settings): i
            for i, c in enumerate(candidates)
        }
        for f, i in futs.items():
            try:
                profiles[i] = f.result()
            except Exception:  # noqa: BLE001 - route the failure to review, keep the batch
                profiles[i] = extract.error_profile(candidates[i], base_no + i + 1)
            done[0] += 1
            session.progress["done"] = done[0]
    return documents, profiles


def _clear_dupes(profiles):
    """Reset duplicate/conflict flags before a full re-dedup (used on merge)."""
    for p in profiles:
        rs = p.get("extraction", {}).get("reasons", [])
        changed = False
        if p.get("duplicate_of"):
            p["duplicate_of"] = None
            if "DUPLICATE" in rs:
                rs.remove("DUPLICATE")
            changed = True
        if p.get("conflict_with"):
            p.pop("conflict_with", None)
            if "CONFLICTING_VERSIONS" in rs:
                rs.remove("CONFLICTING_VERSIONS")
            changed = True
        p.pop("verification_notes", None)  # dedup-derived; recomputed on re-dedup
        if changed and p.get("extraction", {}).get("status") == "NEEDS_REVIEW" and not rs:
            p["extraction"]["status"] = "OK"


def _finalize(session, profiles):
    """Dedup across the whole (possibly merged) batch and publish it."""
    dedup.mark_duplicates(profiles)
    session.profiles = profiles
    session.roster_order = [p["candidate_id"] for p in profiles]
    # Working set = ALL candidates except exact duplicates (flagged candidates stay
    # visible/queryable; only literal duplicates are dropped to avoid double-counting).
    non_dup = [p["candidate_id"] for p in profiles if not p["duplicate_of"]]
    session.view_state["current_set"] = non_dup or [p["candidate_id"] for p in profiles]
    session.summary = _summary(profiles)
    session.progress["stage"] = "ready"
    session.status = "ready"
    _STORE.persist(session)  # survive a service restart mid-conversation


def _run_analysis(session, raw, tmp, settings, hashes=None) -> None:
    """Background worker: ingest -> pipelined acquire+extract -> dedup as a FRESH
    batch, updating progress. Owns and cleans up the temp upload dir."""
    t0 = time.monotonic()
    logging_safe.log_event(
        "analyze_start",
        session_id=session.chat_id,
        job_id=session.job_id,
        file_count=len(raw),
        status="running",
        stage="reading",
    )
    try:
        session.status = "running"
        documents, profiles = _process_documents(session, raw, tmp, settings, base_no=0)
        if documents is None:
            session.status = "error"
            session.error = (
                "no supported CV files found (PDF, Word .docx, or image, optionally inside a ZIP)"
            )
            logging_safe.log_event(
                "analyze_done",
                session_id=session.chat_id,
                job_id=session.job_id,
                status="error",
                stage="ingest",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
            return
        session.documents = documents
        _finalize(session, profiles)
        session.processed_hashes = set(hashes or set())
        logging_safe.log_event(
            "analyze_done",
            session_id=session.chat_id,
            job_id=session.job_id,
            file_count=len(profiles),
            status="ready",
            stage="ready",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    except Exception as e:  # noqa: BLE001
        session.status = "error"
        session.error = f"analysis failed: {e}"
        logging_safe.log_event(
            "analyze_error",
            session_id=session.chat_id,
            job_id=session.job_id,
            status="error",
            error_type=type(e).__name__,
            stage="analysis",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_append(session, raw, tmp, settings, base_no, hashes=None) -> None:
    """Background worker: analyze the NEW upload and MERGE it into the existing batch,
    continuing candidate numbering from base_no and re-running duplicate detection over
    the full merged set. The previous batch is preserved if the append fails."""
    old_profiles = list(session.profiles)
    try:
        session.status = "running"
        documents, new_profiles = _process_documents(session, raw, tmp, settings, base_no=base_no)
        if documents is None:
            session.profiles = old_profiles
            session.error = "no supported CV files found in the new upload; batch unchanged"
            session.progress = {"done": 0, "total": 0, "stage": "ready"}
            session.status = "ready"
            return
        session.documents = (session.documents or []) + documents
        merged = old_profiles + new_profiles
        _clear_dupes(merged)  # recompute duplicates across the FULL merged batch
        _finalize(session, merged)
        session.last_added = len(new_profiles)
        if hashes is not None:
            session.processed_hashes = set(hashes)
    except Exception as e:  # noqa: BLE001
        # keep the old batch intact on failure
        session.profiles = old_profiles
        session.roster_order = [p["candidate_id"] for p in old_profiles]
        session.summary = _summary(old_profiles)
        session.progress = {"done": len(old_profiles), "total": len(old_profiles), "stage": "ready"}
        session.status = "ready"
        session.error = f"append failed (batch left unchanged): {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _summary(profiles) -> dict:
    """Deterministic, guaranteed-DISJOINT KPI partition for any batch.

    Duplicate takes priority over needs-review, so a candidate is counted in
    exactly ONE category. By construction:
        processed + needs_review + duplicates == candidate_count
    This is the single source of truth for the KPI (the filter injects these
    numbers; the model never recounts).
    """
    total = len(profiles)
    duplicates = sum(1 for p in profiles if p["duplicate_of"])
    # Needs-review counts ONLY non-duplicate candidates (duplicate wins), so the
    # three buckets never overlap and always sum to the total.
    needs_review = sum(
        1 for p in profiles if p["extraction"]["status"] == "NEEDS_REVIEW" and not p["duplicate_of"]
    )
    processed = total - duplicates - needs_review
    return {
        "candidate_count": total,
        "processed": processed,
        "needs_review": needs_review,
        "duplicates": duplicates,
        "ready": total - needs_review,
    }  # legacy field, kept for compatibility


def _roster(profiles) -> list:
    out = []
    for p in profiles:
        ident = p["identity"]
        out.append(
            {
                "no": p["display_no"],
                "candidate_id": p["candidate_id"],
                "name": ident["full_name"]["value"] or "(name not found)",
                "headline": p.get("headline"),
                "years": p["total_years_experience"]["value"],
                "top_skills": [s["name"] for s in p["skills"][:6]],
                "email": ident["emails"][0] if ident["emails"] else None,
                "status": p["extraction"]["status"],
                "flags": p["extraction"]["reasons"],
                "duplicate_of": p["duplicate_of"],
                "source": p["source"]["file"],
            }
        )
    return out


_DEMO_BANNER = (
    "> 🧪 **Demo (mock) mode** — these results are deterministic stand-ins, not "
    "real model inference. Set `CV_LLM_MODE=ollama` to use local models.\n\n"
)


def _with_mode(payload: dict[str, Any], settings) -> dict[str, Any]:
    """Tag a response with the active backend mode and, in mock mode, prefix any
    rendered markdown with a clear demo banner so mock output is never mistaken
    for real model inference."""
    mode = getattr(settings, "llm_mode", "ollama")
    payload["llm_mode"] = mode
    if mode == "mock":
        for k in ("roster_markdown", "report_md"):
            if payload.get(k):
                payload[k] = _DEMO_BANNER + payload[k]
    return payload


@app.get("/sessions/{chat_id}/status")
def status_endpoint(chat_id: str, _=Depends(require_key)):
    s = _STORE.get(_safe_chat_id(chat_id))
    if s is None:
        raise HTTPException(
            404, "no active session for this chat (upload CVs first, or it expired)"
        )
    out = s.public_status()
    if s.status == "ready":
        out["summary"] = s.summary
        roster = _roster(s.profiles)
        out["roster"] = roster
        out["roster_markdown"] = render.roster_markdown(s.summary, roster)
    return _with_mode(out, get_settings())


@app.get("/sessions/{chat_id}/facts")
def facts_endpoint(chat_id: str, _=Depends(require_key)):
    """Grounded candidate facts for the model to reason over (the recommended path:
    the model ranks/filters/compares using ONLY these facts). Returns 200 with a
    status when not yet ready so the caller can show progress."""
    s = _STORE.get(_safe_chat_id(chat_id))
    if s is None:
        raise HTTPException(
            404, "no active session for this chat (upload CVs first, or it expired)"
        )
    if s.status != "ready":
        return {"status": s.status, "progress": s.progress, "error": s.error}
    facts = render.facts_block(s.profiles, s.summary)
    last = (s.view_state or {}).get("last_score") or {}
    if last.get("appendix"):
        # keep follow-up answers consistent with the deterministic engine's scores
        facts = facts + "\n\n" + last["appendix"]
    return _with_mode(
        {
            "status": "ready",
            "count": len(s.profiles),
            "summary": s.summary,
            "roster_markdown": render.roster_markdown(s.summary, _roster(s.profiles)),
            "facts": facts,
        },
        get_settings(),
    )


@app.post("/sessions/{chat_id}/audit")
def audit_endpoint(chat_id: str, body: dict[str, Any] | None = None, _=Depends(require_key)):
    """Fairness / bias audit for the chat's batch. ALWAYS returns the
    protected-attribute leakage report (proves the group-blind invariant held).
    If the caller supplies demographic ``labels`` (candidate_id -> {axis: group})
    and a ``selected`` map (candidate_id -> bool) for a formal EEOC / NYC LL144
    audit, it also returns the four-fifths adverse-impact analysis."""
    s = _STORE.get(_safe_chat_id(chat_id))
    if s is None:
        raise HTTPException(
            404, "no active session for this chat (upload CVs first, or it expired)"
        )
    texts = []
    last = (s.view_state or {}).get("last_score") or {}
    if last.get("appendix"):
        texts.append(last["appendix"])
    report = {"chat_id": s.chat_id, "leakage": fairness.leakage_report(s.profiles, texts)}
    body = body or {}
    labels, selected = body.get("labels"), body.get("selected")
    if isinstance(labels, dict) and isinstance(selected, dict):
        report["adverse_impact"] = fairness.four_fifths_audit(
            {k: bool(v) for k, v in selected.items()}, labels
        )
    return report


@app.post("/sessions/{chat_id}/query")
def query_endpoint(chat_id: str, body: QueryRequest, _=Depends(require_key)):
    """Multi-turn conversational query over the already-analyzed batch. The chat's
    batch is resolved by chat_id — no re-upload, no re-processing."""
    s = _STORE.get(_safe_chat_id(chat_id))
    if s is None:
        raise HTTPException(
            404, "no active session for this chat (upload CVs first, or it expired)"
        )
    if s.status != "ready":
        return JSONResponse(
            status_code=200,
            content={
                "chat_id": s.chat_id,
                "status": s.status,
                "answer": f"The batch is still **{s.status}** "
                f"({s.progress.get('done')}/{s.progress.get('total')}). Ask again shortly.",
            },
        )
    result = engine.answer(s, body.question, get_settings())
    return JSONResponse(
        status_code=200, content={"chat_id": s.chat_id, "status": s.status, **result}
    )


@app.post("/sessions/{chat_id}/score")
def score_endpoint(chat_id: str, body: ScoreRequest, _=Depends(require_key)):
    """Deterministic fit scoring of the chat's analyzed batch against a job
    description. The rubric is derived from THIS JD at runtime (any role /
    industry / language); matching is evidence-based; aggregation, ranking and
    the rendered report are fully deterministic. Same session + same JD =>
    byte-identical report. Returns 200 with scorable=false (and an explanatory
    report) when the JD lacks concrete requirements."""
    s = _STORE.get(_safe_chat_id(chat_id))
    if s is None:
        raise HTTPException(
            404, "no active session for this chat (upload CVs first, or it expired)"
        )
    if s.status != "ready":
        return JSONResponse(
            status_code=200,
            content={
                "chat_id": s.chat_id,
                "status": s.status,
                "scorable": False,
                "reason": f"batch still {s.status} "
                f"({s.progress.get('done')}/{s.progress.get('total')})",
            },
        )
    t0 = time.monotonic()
    try:
        result = scoring_service.score_session(
            s, body.jd_text, get_settings(), lang=(body.lang or "en"), top_n=body.top_n
        )
    except scoring_rubric.RubricError as e:
        logging_safe.log_event(
            "score_error",
            session_id=chat_id,
            job_id=s.job_id,
            status="error",
            error_type=type(e).__name__,
            stage="rubric",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
        return JSONResponse(
            status_code=200,
            content={"chat_id": s.chat_id, "status": "ready", "scorable": False, "reason": str(e)},
        )
    _STORE.persist(s)
    logging_safe.log_event(
        "score_done",
        session_id=chat_id,
        job_id=s.job_id,
        status="ready",
        stage="scoring",
        duration_ms=(time.monotonic() - t0) * 1000,
    )
    return JSONResponse(
        status_code=200,
        content=_with_mode({"chat_id": s.chat_id, "status": "ready", **result}, get_settings()),
    )


@app.post("/sessions/{chat_id}/reset")
def reset_endpoint(chat_id: str, _=Depends(require_key)):
    dropped = _STORE.reset(_safe_chat_id(chat_id))
    return {"reset": dropped}
