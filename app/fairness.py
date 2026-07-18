"""Fairness / bias-audit layer for HR screening.

Two complementary checks, both deterministic:

1. PROTECTED-ATTRIBUTE LEAKAGE (always-on, group-blind by design):
   the pipeline deliberately never extracts gender / age / nationality / race /
   religion / marital status / photo, so ranking is blind to them. This check
   PROVES that invariant held for a given batch — it asserts no protected field
   was populated on any profile and that no rendered, customer-facing text
   exposes a protected attribute as a labelled field. It turns the golden-test
   guarantee ("0 protected leaks") into a runtime, reportable capability.

2. FOUR-FIFTHS / ADVERSE-IMPACT AUDIT (opt-in, needs demographic labels):
   the EEOC four-fifths rule and NYC Local Law 144 measure whether a screening
   tool selects protected groups at materially different rates. Because the tool
   is group-blind it cannot know the labels itself; for a formal audit the
   customer supplies demographic labels for a representative sample. Given those
   labels and each candidate's selection outcome, this computes the selection
   rate per group and the impact ratio (min-rate / max-rate); an impact ratio
   below 0.80 flags potential adverse impact.

Nothing here scores or ranks candidates and nothing stores PII — it only audits.
"""

from __future__ import annotations

import re
from typing import Any

from .pipeline.extract import PROTECTED_KEYS

FOUR_FIFTHS = 0.80

# Labelled protected fields that must NEVER appear in customer-facing output.
# Matching a LABEL (e.g. "gender:") avoids false positives on ordinary words
# ("management" contains "man"). Bilingual.
_LEAK_PATTERNS = [
    (r"\b(gender|sex)\s*[:=]", "gender"),
    (r"\b(date of birth|d\.?o\.?b\.?)\s*[:=]", "date_of_birth"),
    (r"\bage\s*[:=]\s*\d", "age"),
    (r"\bnationality\s*[:=]", "nationality"),
    (r"\b(marital status|marital)\s*[:=]", "marital_status"),
    (r"\breligion\s*[:=]", "religion"),
    (r"\b(race|ethnicity)\s*[:=]", "race"),
    (r"(الجنس|النوع)\s*[:：]", "gender"),
    (r"(تاريخ الميلاد|الميلاد)\s*[:：]", "date_of_birth"),
    (r"(الجنسية)\s*[:：]", "nationality"),
    (r"(الحالة الاجتماعية|الحالة الزوجية)\s*[:：]", "marital_status"),
    (r"(الديانة|الدين)\s*[:：]", "religion"),
]
_LEAK_RX = [(re.compile(p, re.I), tag) for p, tag in _LEAK_PATTERNS]


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for _k, v in obj.items():
            yield from _walk_strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _walk_strings(v)


def scan_profile_leakage(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Confirm no profile carries a populated protected attribute. Returns a list
    of leak records (empty = clean). Ignores the audit bookkeeping key
    ``_excluded_by_policy`` (which merely lists what is excluded)."""
    leaks: list[dict[str, Any]] = []
    for p in profiles:
        for key in p:
            if key == "_excluded_by_policy":
                continue
            if key.lower() in PROTECTED_KEYS and p.get(key) not in (None, "", [], {}):
                leaks.append({"candidate": p.get("display_no"), "field": key, "where": "profile"})
    return leaks


def scan_text_leakage(text: str) -> list[dict[str, Any]]:
    """Flag any protected attribute exposed as a labelled field in customer text."""
    out: list[dict[str, Any]] = []
    for rx, tag in _LEAK_RX:
        m = rx.search(text or "")
        if m:
            out.append({"attribute": tag, "match": m.group(0), "where": "output"})
    return out


def leakage_report(
    profiles: list[dict[str, Any]], texts: list[str] | None = None
) -> dict[str, Any]:
    leaks = scan_profile_leakage(profiles)
    for t in texts or []:
        leaks.extend(scan_text_leakage(t))
    return {
        "protected_leaks": len(leaks),
        "clean": not leaks,
        "details": leaks,
        "excluded_by_policy": sorted(PROTECTED_KEYS),
    }


def four_fifths_audit(
    selected: dict[str, bool], groups: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """EEOC four-fifths / NYC LL144 adverse-impact audit.

    ``selected`` maps candidate_id -> was-selected (e.g. shortlisted / above a
    fit threshold). ``groups`` maps candidate_id -> {axis: label} for one or
    more protected axes (e.g. {"gender": "female", "race": "asian"}), supplied
    externally for the audit. Returns per-axis selection rates, impact ratios
    and an adverse-impact flag (impact ratio < 0.80)."""
    axes: dict[str, dict[str, dict[str, int]]] = {}
    for cid, grp in groups.items():
        if cid not in selected:
            continue
        for axis, label in (grp or {}).items():
            a = axes.setdefault(axis, {})
            g = a.setdefault(str(label), {"total": 0, "selected": 0})
            g["total"] += 1
            g["selected"] += 1 if selected[cid] else 0

    result: dict[str, Any] = {
        "four_fifths_threshold": FOUR_FIFTHS,
        "axes": {},
        "adverse_impact": False,
    }
    for axis, gmap in axes.items():
        rates = {
            label: (c["selected"] / c["total"] if c["total"] else 0.0) for label, c in gmap.items()
        }
        top = max(rates.values()) if rates else 0.0
        rows = {}
        axis_flag = False
        for label, c in gmap.items():
            rate = rates[label]
            ratio = (rate / top) if top > 0 else 1.0
            below = ratio < FOUR_FIFTHS and c["total"] > 0
            axis_flag = axis_flag or below
            rows[label] = {
                "total": c["total"],
                "selected": c["selected"],
                "selection_rate": round(rate, 3),
                "impact_ratio": round(ratio, 3),
                "adverse": below,
            }
        result["axes"][axis] = {
            "groups": rows,
            "reference_rate": round(top, 3),
            "adverse_impact": axis_flag,
        }
        result["adverse_impact"] = result["adverse_impact"] or axis_flag
    return result
