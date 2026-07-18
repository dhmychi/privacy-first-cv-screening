"""Resolve natural-language references to concrete candidate_ids (plan section
14). Deterministic-first: ordinals ('candidate 2', 'the second one'), explicit
ids, names, and anaphora ('those', 'the previous shortlist/ranking'). Fuzzy
references ('the weak ones') are handled in the engine with an echoed
interpretation, never silently guessed here."""

from __future__ import annotations

import re

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _by_no(session) -> dict[int, str]:
    return {p["display_no"]: p["candidate_id"] for p in session.profiles}


def _dedupe(ids: list[str]) -> list[str]:
    out: list[str] = []
    for i in ids:
        if i and i not in out:
            out.append(i)
    return out


def resolve_candidate_refs(question: str, session) -> list[str]:
    ql = (question or "").lower()
    by_no = _by_no(session)
    ids: list[str] = []

    def _add(cid: str | None) -> None:
        if cid:
            ids.append(cid)

    # explicit "candidate 2", "#3", "no. 4", "number 5", "c_002"
    for m in re.finditer(r"(?:candidate|cand|#|no\.?|number)\s*#?\s*(\d{1,3})", ql):
        _add(by_no.get(int(m.group(1))))
    for m in re.finditer(r"\bc_(\d{1,3})\b", ql):
        _add(by_no.get(int(m.group(1))))
    # second operand in "candidate 1 and 2", "2 vs 5", "3 & 4"
    for m in re.finditer(r"(?:\band\b|&|,|vs\.?|versus)\s*#?\s*(\d{1,3})", ql):
        _add(by_no.get(int(m.group(1))))

    # ordinals: "the second", "second candidate"
    for word, n in ORDINALS.items():
        if re.search(r"\b" + word + r"\b", ql):
            _add(by_no.get(n))

    # names (full, then first name on a word boundary)
    for p in session.profiles:
        nm = (p["identity"]["full_name"]["value"] or "").strip()
        if not nm:
            continue
        if nm.lower() in ql:
            ids.append(p["candidate_id"])
            continue
        first = nm.split()[0].lower()
        if len(first) >= 3 and re.search(r"\b" + re.escape(first) + r"\b", ql):
            ids.append(p["candidate_id"])

    # anaphora -> conversation memory
    vs = session.view_state
    if re.search(r"\b(those|them|these|that group|the same)\b", ql):
        ids += vs.get("last_result", [])
    if "shortlist" in ql or "short list" in ql or "short-list" in ql:
        ids += vs.get("last_shortlist", [])
    if re.search(r"\b(previous|the)\s+ranking\b|\branked\b|\bthe ranking\b", ql):
        ids += [r["candidate_id"] for r in vs.get("last_ranking", [])]

    return _dedupe(ids)


def parse_top_n(question: str, default: int = 5) -> int:
    m = re.search(r"\b(?:top|best|first)\s+(\d{1,3})\b", (question or "").lower())
    return int(m.group(1)) if m else default


def parse_years_threshold(question: str):
    """Return (op, n) where op in {'gte','lt'} or None. Handles 'more than 5
    years', 'at least 5', 'over 5', '>= 5', 'less than 5', 'under 5'."""
    ql = (question or "").lower()
    m = re.search(
        r"(?:more than|over|at least|minimum|min|greater than|>=?|above)\s*(\d{1,2})\s*\+?\s*year",
        ql,
    )
    if m:
        return ("gte", int(m.group(1)))
    m = re.search(r"(?:less than|under|below|fewer than|<)\s*(\d{1,2})\s*year", ql)
    if m:
        return ("lt", int(m.group(1)))
    m = re.search(r"(\d{1,2})\s*\+\s*years?", ql)  # "5+ years"
    if m:
        return ("gte", int(m.group(1)))
    return None
