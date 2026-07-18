"""Deterministic score aggregation, gating, banding and ranking.

Pure functions over (rubric, verdicts): no I/O, no randomness, no model calls.
Same inputs => identical outputs, always. All policy constants are explicit,
generic (role-independent) and env-overridable.

Scoring contract for uncertainty:
  * 'unverified' requirements are EXCLUDED from the denominator and disclosed
    ("scored on k of n requirements") - never a silent zero.
  * 'not_assessable' (soft traits) are excluded from scoring entirely.
  * A candidate whose verifiable required coverage is below MIN_VERIFIABLE is
    NOT scored - routed to the insufficient-evidence lane instead.
  * A definitively-missing HARD requirement (stated years floor / degree /
    license / language) caps the score at the Partial ceiling - the candidate
    can never look 'Strong' while failing a stated hard minimum.
  * When the evidenced coverage of core required capabilities is near zero the
    score is capped in the Weak band (role-mismatch guard, derived purely from
    THIS JD's requirements).
"""

from __future__ import annotations

import os
from typing import Any

CREDITS = {"met": 1.0, "partial": 0.5, "missing": 0.0}

BAND_STRONG, BAND_GOOD, BAND_PARTIAL = 80, 60, 40
CAP_HARD_MISS = 45  # Partial ceiling when a hard minimum is definitively missing
CAP_NO_CORE = 25  # Weak cap when no core required capability is evidenced
CAP_LOW_CORE = 39  # Weak ceiling when core evidenced coverage < 25%
CAP_UNVERIFIED_REQ = BAND_STRONG - 1  # a candidate can never look 'Strong'
# while a REQUIRED item is unverifiable —
# missing information must not pay
MIN_VERIFIABLE = 0.5  # below this fraction of verifiable required items -> review lane

HARD_RTYPES = {"experience_years", "education", "certification", "language"}
CORE_RTYPES = {"skill", "responsibility"}


def _w_required() -> float:
    try:
        return float(os.environ.get("CV_W_REQUIRED", "1.0"))
    except ValueError:
        return 1.0


def _w_preferred() -> float:
    try:
        return float(os.environ.get("CV_W_PREFERRED", "0.4"))
    except ValueError:
        return 0.4


def level_for(score: int) -> str:
    if score >= BAND_STRONG:
        return "strong"
    if score >= BAND_GOOD:
        return "good"
    if score >= BAND_PARTIAL:
        return "partial"
    return "weak"


def score_candidate(rubric: dict[str, Any], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    reqs = {r["id"]: r for r in rubric.get("requirements") or []}
    w_req, w_pref = _w_required(), _w_preferred()

    num = den = 0.0
    included = 0
    unverified: list[str] = []
    unverified_required: list[str] = []
    missing_required: list[str] = []
    matched: list[dict[str, Any]] = []
    hard_missing: list[str] = []
    core_total = core_hit = 0
    req_met_count = 0

    for v in verdicts:
        req = reqs.get(v.get("req_id"))
        if req is None:
            continue
        verdict = v.get("verdict")
        if verdict == "not_assessable":
            continue
        weight = w_req if req["kind"] == "required" else w_pref
        if verdict == "unverified":
            unverified.append(req["label"])
            if req["kind"] == "required":
                unverified_required.append(req["label"])
            continue
        credit = CREDITS.get(verdict or "", 0.0)
        num += weight * credit
        den += weight
        included += 1
        if req["kind"] == "required" and req["rtype"] in CORE_RTYPES:
            core_total += 1
            if verdict in ("met", "partial"):
                core_hit += 1
        if verdict in ("met", "partial"):
            matched.append(
                {
                    "label": req["label"],
                    "verdict": verdict,
                    "detail": v.get("detail", ""),
                    "evidence": v.get("evidence"),
                    "kind": req["kind"],
                }
            )
            if verdict == "met" and req["kind"] == "required":
                req_met_count += 1
        elif req["kind"] == "required":
            missing_required.append(req["label"])
            if req["rtype"] in HARD_RTYPES:
                hard_missing.append(req["label"])

    assessable = [
        v
        for v in verdicts
        if reqs.get(v.get("req_id"), {}).get("rtype") != "soft" and v.get("req_id") in reqs
    ]
    required_assessable = [v for v in assessable if reqs[v["req_id"]]["kind"] == "required"]
    required_verifiable = [v for v in required_assessable if v.get("verdict") != "unverified"]
    verifiable_fraction = (
        len(required_verifiable) / len(required_assessable) if required_assessable else 0.0
    )

    if den <= 0 or verifiable_fraction < MIN_VERIFIABLE:
        return {
            "scored": False,
            "reason": "insufficient verifiable evidence",
            "unverified": unverified,
            "verifiable_fraction": round(verifiable_fraction, 2),
        }

    raw = 100.0 * num / den
    caps: list[dict[str, str]] = []
    if hard_missing:
        raw = min(raw, CAP_HARD_MISS)
        caps.append({"cap": "hard_minimum_missing", "items": ", ".join(hard_missing[:3])})
    if unverified_required:
        # Unverified REQUIRED items are excluded from the denominator (never a
        # silent zero) but must not let missing information out-rank verified
        # candidates: cap below the Strong band until a human verifies them.
        raw = min(raw, CAP_UNVERIFIED_REQ)
        caps.append({"cap": "required_unverified", "items": ", ".join(unverified_required[:3])})
    if core_total > 0:
        frac = core_hit / core_total
        if frac == 0:
            raw = min(raw, CAP_NO_CORE)
            caps.append({"cap": "no_core_capability_evidenced", "items": ""})
        elif frac < 0.25:
            raw = min(raw, CAP_LOW_CORE)
            caps.append({"cap": "low_core_coverage", "items": ""})

    score = int(round(raw))
    return {
        "scored": True,
        "score": score,
        "level": level_for(score),
        "matched": matched,
        "missing_required": missing_required,
        "unverified": unverified,
        "caps": caps,
        "req_met_count": req_met_count,
        "included_requirements": included,
        "total_assessable": len(assessable),
        "verifiable_fraction": round(verifiable_fraction, 2),
    }


def rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic total order over scored candidates.
    Order within equal scores: required-met count desc -> verified years desc
    (unknown last) -> field coverage desc -> stable roster number asc.

    COMPETITION RANKING on the score: candidates with the SAME score share the
    same rank (1, 1, 3) and each row carries tied=True — the report must show a
    tie as a tie, never invent a unique winner from upload order."""

    def key(r: dict[str, Any]):
        yrs = r.get("years")
        cov = r.get("field_coverage") or 0.0
        return (
            -r["score"],
            -r.get("req_met_count", 0),
            -(yrs if isinstance(yrs, int | float) else -1.0),
            -cov,
            r.get("display_no", 10**9),
        )

    ordered = sorted((r for r in results if r.get("scored")), key=key)
    for i, r in enumerate(ordered):
        if i and r["score"] == ordered[i - 1]["score"]:
            r["rank"] = ordered[i - 1]["rank"]
        else:
            r["rank"] = i + 1
    counts: dict[int, int] = {}
    for r in ordered:
        counts[r["score"]] = counts.get(r["score"], 0) + 1
    for r in ordered:
        r["tied"] = counts[r["score"]] > 1
    return ordered
