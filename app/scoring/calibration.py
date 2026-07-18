"""Judge calibration against a human-authored gold-set, with drift monitoring.

LLM-as-judge is reliable only when its verdicts are periodically checked against
a fixed, human-authored gold-set and monitored for drift (the consensus best
practice: keep a gold-set, measure agreement/kappa, alert when divergence
exceeds ~20-25%). This module:

  * defines a criterion-level GOLD_SET of known-correct per-requirement verdicts
    (label-agnostic: a case is located by rtype/keyword, so it survives the
    non-deterministic rubric wording),
  * runs the live judge over it and reports accuracy + Cohen's kappa + a
    confusion breakdown,
  * appends each run to a calibration log and flags DRIFT when accuracy falls
    materially below the recorded baseline.

Deterministic given a fixed model + gold-set (temperature 0). Pure evaluation:
it never changes scoring behaviour, it only measures it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from . import judge, rubric

# Divergence above this fraction from the baseline signals "recalibrate".
DRIFT_DIVERGENCE = 0.25

# Each case: (candidate name substring, requirement finder, allowed verdicts, why).
# Finder = {"rtype": ..., "needles": [...]} located in the built rubric, so a
# label re-wording never breaks the case (it tests the JUDGE, not the parser).
GOLD_SET: list[dict[str, Any]] = [
    {
        "cand": "Tariq",
        "rtype": "experience_years",
        "needles": [],
        "allow": ["missing"],
        "why": "sales tenure is not software-development experience",
    },
    {
        "cand": "Tariq",
        "rtype": "skill",
        "needles": ["Java", "Python"],
        "allow": ["missing"],
        "why": "sales candidate has no programming languages",
    },
    {
        "cand": "Nora",
        "rtype": "education",
        "needles": [],
        "allow": ["partial"],
        "why": "BA History is not a CS-related degree (level ok, field wrong)",
    },
    {
        "cand": "Nora",
        "rtype": "skill",
        "needles": ["Java", "Python"],
        "allow": ["met"],
        "why": "Nora lists Python -> satisfies the alternatives",
    },
    {
        "cand": "Lina",
        "rtype": "skill",
        "needles": ["Java", "Python"],
        "allow": ["missing"],
        "why": "JavaScript is not Java/Python",
    },
    {
        "cand": "Lina",
        "rtype": "skill",
        "needles": ["SQL"],
        "allow": ["missing"],
        "why": "MongoDB (NoSQL) does not prove SQL",
    },
    {
        "cand": "Yousef",
        "rtype": "education",
        "needles": [],
        "allow": ["met"],
        "why": "BSc Computer Science satisfies a CS degree",
    },
    {
        "cand": "Yousef",
        "rtype": "skill",
        "needles": ["AWS", "Azure"],
        "allow": ["missing", "unverified"],
        "why": "Yousef lists no AWS/cloud",
    },
    {
        "cand": "مهند",
        "rtype": "education",
        "needles": [],
        "allow": ["met"],
        "why": "Arabic CV: CS degree recognised",
    },
    {
        "cand": "مهند",
        "rtype": "language",
        "needles": ["Arabic"],
        "allow": ["met"],
        "why": "Arabic CV satisfies the Arabic-language preferred",
    },
]

_VERDICTS = ["met", "partial", "missing", "unverified", "not_assessable"]


def _find_req(rub: dict[str, Any], needles: list[str], rtype: str | None) -> str | None:
    for r in rub.get("requirements") or []:
        if rtype and r["rtype"] != rtype:
            continue
        hay = (
            r.get("label", "")
            + " | "
            + " ".join(r.get("keywords") or [])
            + " | "
            + r.get("text", "")
        ).lower()
        if not needles or any(n.lower() in hay for n in needles):
            return r["id"]
    return None


def cohens_kappa(pairs: list[tuple]) -> float:
    """Cohen's kappa between the gold rater and the judge over categorical
    verdicts. pairs = [(gold_primary, predicted), ...]."""
    n = len(pairs)
    if n == 0:
        return 0.0
    cats = _VERDICTS
    po = sum(1 for g, p in pairs if g == p) / n
    pe = 0.0
    for c in cats:
        pg = sum(1 for g, _ in pairs if g == c) / n
        pp = sum(1 for _, p in pairs if p == c) / n
        pe += pg * pp
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 3)


def evaluate(settings, profiles: list[dict[str, Any]], jd_text: str) -> dict[str, Any]:
    """Run the judge over the gold-set and score agreement."""
    rub = rubric.build(settings, jd_text)
    by_name = {}
    for p in profiles:
        nm = ((p.get("identity", {}) or {}).get("full_name", {}) or {}).get("value") or ""
        by_name[nm] = p
    lbl = {r["id"]: r["label"] for r in rub.get("requirements") or []}

    cases: list[dict[str, Any]] = []
    pairs: list[tuple] = []
    verdict_cache: dict[str, Any] = {}
    for gc in GOLD_SET:
        prof = next((p for nm, p in by_name.items() if gc["cand"].lower() in nm.lower()), None)
        rid = _find_req(rub, gc["needles"], gc["rtype"])
        got = "NO-CANDIDATE" if prof is None else ("NO-REQ" if rid is None else None)
        if got is None:
            assert prof is not None
            key = prof["candidate_id"]
            if key not in verdict_cache:
                verdict_cache[key] = {
                    v["req_id"]: v["verdict"]
                    for v in (judge.judge_candidate(settings, prof, rub) or [])
                }
            got = verdict_cache[key].get(rid, "NO-VERDICT")
        ok = got in gc["allow"]
        cases.append(
            {
                "cand": gc["cand"],
                "req": lbl.get(rid, gc["rtype"]),
                "expected": gc["allow"],
                "got": got,
                "pass": ok,
                "why": gc["why"],
            }
        )
        if got in _VERDICTS:
            pairs.append((gc["allow"][0], got))
    n = len(cases)
    passed = sum(1 for c in cases if c["pass"])
    confusion: dict[str, int] = {}
    for c in cases:
        confusion[c["got"]] = confusion.get(c["got"], 0) + 1
    return {
        "n": n,
        "passed": passed,
        "accuracy": round(passed / n, 3) if n else 0.0,
        "kappa": cohens_kappa(pairs),
        "confusion": confusion,
        "matcher": getattr(settings, "matcher", "?"),
        "cases": cases,
    }


def _log_path(settings) -> str:
    d = getattr(settings, "data_dir", "") or "/tmp"
    return os.path.join(d, "calibration_log.json")


def record_and_check_drift(settings, metrics: dict[str, Any]) -> dict[str, Any]:
    """Append this run to the calibration log and compare against the baseline
    (the best accuracy recorded to date). Flags drift when the latest accuracy
    falls more than DRIFT_DIVERGENCE below that baseline."""
    path = _log_path(settings)
    log: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
    except (OSError, ValueError):
        log = []
    entry = {
        "ts": int(time.time()),
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "matcher": metrics.get("matcher"),
    }
    log.append(entry)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log[-50:], f, ensure_ascii=False)
    except OSError:
        pass
    baseline = max((e["accuracy"] for e in log[:-1]), default=metrics["accuracy"])
    divergence = round(baseline - metrics["accuracy"], 3)
    return {
        "baseline_accuracy": baseline,
        "latest_accuracy": metrics["accuracy"],
        "divergence": divergence,
        "threshold": DRIFT_DIVERGENCE,
        "drift_alert": divergence > DRIFT_DIVERGENCE,
        "runs_logged": len(log),
    }
