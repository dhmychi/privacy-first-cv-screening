"""Scoring orchestrator: session + JD text -> deterministic scored report.

Flow: rubric (LLM, cached by JD hash) -> vectors (batch embed) -> per-candidate
verdicts -> engine aggregation -> engine ranking -> code-rendered report.
Results are cached per (jd_hash, lang, top_n) inside the session, so repeating
the same JD in a chat returns the identical bytes instantly.

Exclusions (honest lanes, never silent):
  * duplicates             -> excluded entirely (counted in the KPI)
  * NEEDS_REVIEW extraction-> review lane, listed under 'Needs Review'
  * insufficient evidence  -> review lane (engine refused to score)
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import SCORING_REV, engine, judge, match, report, rubric

# Bumped when the matcher CONTRACT changes, so judge/legacy never share a cache
# entry and a matcher switch never serves a stale score.
MATCHER_REV = "1"


def _global_rubric_path(settings, h: str) -> str:
    d = os.path.join(getattr(settings, "data_dir", "") or "", "rubrics")
    # the rubric depends on the backend that built it: mock and ollama produce
    # different rubrics for the same JD, so they must never share a cache entry.
    mode = getattr(settings, "llm_mode", "ollama")
    return os.path.join(d, f"{SCORING_REV}_{mode}_{h}.json")


def _load_global_rubric(settings, h: str) -> dict[str, Any] | None:
    """Cross-session rubric cache: the same JD text yields the SAME rubric on
    every chat/session, removing rubric-side variance from repeat scorings."""
    if not (getattr(settings, "data_dir", "") or ""):
        return None  # persistence disabled -> no on-disk rubric cache
    try:
        with open(_global_rubric_path(settings, h), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_global_rubric(settings, h: str, rub: dict[str, Any]) -> None:
    if not (getattr(settings, "data_dir", "") or ""):
        return  # persistence disabled -> do not write a relative cache dir
    path = _global_rubric_path(settings, h)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rub, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _display(profile: dict[str, Any]) -> dict[str, Any]:
    ident = profile.get("identity") or {}
    name = ((ident.get("full_name") or {}).get("value") or "").strip()
    return {
        "candidate_id": profile.get("candidate_id"),
        "display_no": profile.get("display_no"),
        "name": name or "(name not detected)",
        "years": (profile.get("total_years_experience") or {}).get("value"),
        "field_coverage": (profile.get("extraction") or {}).get("field_coverage"),
    }


def score_session(
    session, jd_text: str, settings, lang: str = "en", top_n: int | None = None
) -> dict[str, Any]:
    vs = session.view_state
    h = rubric.jd_hash(jd_text)
    matcher = getattr(settings, "matcher", "legacy")
    mode = getattr(settings, "llm_mode", "ollama")
    cache = vs.setdefault("score_cache", {})
    ck = f"{SCORING_REV}:{MATCHER_REV}:{matcher}:{mode}:{h}:{lang}:{top_n or 0}"
    if ck in cache:
        return cache[ck]

    rub_cache = vs.setdefault("rubric_cache", {})
    rub_key = f"{SCORING_REV}:{mode}:{h}"
    rub = rub_cache.get(rub_key)
    if rub is None:
        rub = _load_global_rubric(settings, h)  # same JD => same rubric, any session
    if rub is None:
        rub = rubric.build(settings, jd_text)  # may raise RubricError (infra)
        _save_global_rubric(settings, h, rub)
    rub_cache[rub_key] = rub
    while len(rub_cache) > 4:
        rub_cache.pop(next(iter(rub_cache)))

    if rub.get("clarity") != "ok":
        result = {
            "scorable": False,
            "reason": rub.get("clarity_reason", ""),
            "rubric": _rubric_public(rub),
            "report_md": report.render_not_scorable(rub.get("clarity_reason", ""), lang),
        }
        cache[ck] = result
        return result

    profiles = [p for p in session.profiles if not p.get("duplicate_of")]
    scored_pool: list[dict[str, Any]] = []
    review_lane: list[dict[str, Any]] = []
    for p in profiles:
        ex = p.get("extraction") or {}
        if ex.get("status") == "NEEDS_REVIEW":
            review_lane.append(
                {**_display(p), "kind": "needs_review", "flags": ex.get("reasons") or []}
            )
        else:
            scored_pool.append(p)

    # Candidate-phrase embeddings depend ONLY on the analyzed profiles, not the
    # JD, so embed them ONCE per session and reuse for every job description.
    # Cached in-memory on the session object (NOT persisted - a whitelist governs
    # persistence; recomputed after a restart or an append/reset that changes the
    # candidate set). Only the JD's requirement texts are embedded per score.
    pool_key = tuple(p["candidate_id"] for p in scored_pool)
    cached = getattr(session, "_phrase_cache", None)
    if cached is not None and cached[0] == pool_key:
        phrase_vecs = cached[1]
    else:
        phrase_vecs = match.phrase_vectors(settings, scored_pool)
        try:
            session._phrase_cache = (pool_key, phrase_vecs)
        except Exception:  # pragma: no cover - exotic session objects
            pass
    req_vecs = match.requirement_vectors(settings, rub)

    results: list[dict[str, Any]] = []
    for p in scored_pool:
        verdicts = None
        if matcher == "judge":
            # grounded LLM reasoning; None on infra failure -> legacy fallback for
            # THIS candidate, so judge mode is never worse than legacy on error.
            verdicts = judge.judge_candidate(settings, p, rub)
        if verdicts is None:
            verdicts = match.match_candidate(p, rub, req_vecs, phrase_vecs)
        res = engine.score_candidate(rub, verdicts)
        base = _display(p)
        if res.get("scored"):
            results.append({**base, **res})
        else:
            review_lane.append(
                {
                    **base,
                    "kind": "insufficient",
                    "verifiable_fraction": res.get("verifiable_fraction"),
                    "flags": [],
                }
            )

    ranked = engine.rank(results)
    dup_count = sum(1 for p in session.profiles if p.get("duplicate_of"))
    md = report.render(rub, ranked, review_lane, lang=lang, top_n=top_n, dup_count=dup_count)
    appendix = report.facts_appendix(rub, ranked, review_lane)

    vs["last_ranking"] = [{"candidate_id": r["candidate_id"], "score": r["score"]} for r in ranked]
    vs["last_score"] = {"jd_hash": h, "role": rub.get("role_title"), "appendix": appendix}

    result = {
        "scorable": True,
        "rubric": _rubric_public(rub),
        "report_md": md,
        "scored_count": len(ranked),
        "shown": report.top_n_for(len(ranked), top_n),
        "review_count": len(review_lane),
        "ranking": [
            {
                "rank": r["rank"],
                "cv": r["display_no"],
                "name": r["name"],
                "score": r["score"],
                "level": r["level"],
            }
            for r in ranked
        ],
    }
    cache[ck] = result
    while len(cache) > 4:
        cache.pop(next(iter(cache)))
    return result


def _rubric_public(rub: dict[str, Any]) -> dict[str, Any]:
    return {
        "role_title": rub.get("role_title"),
        "clarity": rub.get("clarity"),
        "requirements": [
            {"id": r["id"], "label": r["label"], "kind": r["kind"], "rtype": r["rtype"]}
            for r in rub.get("requirements", [])
        ],
        "jd_hash": rub.get("jd_hash"),
    }
