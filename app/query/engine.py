"""Conversational query engine (plan sections 13-14). Routes a question to an
intent, runs it over the session's structured store, updates the multi-turn
view-state, and returns an evidence-cited markdown answer. Deterministic-first;
the LLM is used only for JD parsing. Every answer ends with the HR disclaimer.
"""

from __future__ import annotations

import re
from typing import Any

from . import jd, references, scoring

DISCLAIMER = (
    "\n\n---\n_Decision support for HR — not a final hiring decision. "
    "Findings cite the source CV; please verify candidates flagged for review._"
)


# ---------------------------------------------------------------- helpers
def _pid(session) -> dict[str, Any]:
    return {p["candidate_id"]: p for p in session.profiles}


def _years(p) -> float:
    return p["total_years_experience"]["value"] or 0


def _current_ids(session) -> list[str]:
    ids = session.view_state.get("current_set") or [p["candidate_id"] for p in session.profiles]
    order = {cid: i for i, cid in enumerate(session.roster_order)}
    return sorted([i for i in ids if i in order], key=lambda c: order[c])


def _profiles(session, ids: list[str]) -> list[dict[str, Any]]:
    pid = _pid(session)
    return [pid[i] for i in ids if i in pid]


def _label(p) -> str:
    nm = p["identity"]["full_name"]["value"] or "(name not found)"
    return f"Candidate {p['display_no']} — {nm}"


def _ev(ev) -> str:
    if not ev:
        return ""
    return f" _(p.{ev['page']}: “{ev['quote'][:70]}”)_"


def _flags(p) -> str:
    r = p["extraction"]["reasons"]
    return f"  ⚠️ _{', '.join(r)}_" if r else ""


def _set_history(session, op: str, detail: str):
    session.view_state.setdefault("history", []).append({"op": op, "detail": detail})


def _result(intent, md, ids, session) -> dict[str, Any]:
    return {
        "intent": intent,
        "answer": md + DISCLAIMER,
        "candidate_ids": ids,
        "current_set_size": len(_current_ids(session)),
    }


def _index(session, profiles, settings):
    return scoring.EmbedIndex(settings, profiles)


# ---------------------------------------------------------------- intent tests
def _is_reset(ql):
    return bool(
        re.search(
            r"\b(reset|start over|show all|all candidates|full list|everyone|clear (the )?filter)\b",
            ql,
        )
    )


def _is_count(ql):
    return bool(re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|how many candidates", ql))


def _is_compare(ql):
    return bool(re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b|difference between", ql))


def _is_exclude(ql):
    return bool(re.search(r"\bexclude\b|\bremove\b|\bdrop\b|\bfilter out\b|\bget rid of\b", ql))


def _is_who_has(ql):
    return bool(
        re.search(
            r"\bwho (has|have|knows|is|are|with)\b|candidates? with|experience (in|with)|"
            r"skilled in|proficient in|knows\b|\bwhat about\b|anyone with|any candidate|"
            r"\bwhich candidates?\b|list (the )?candidates? (with|who)",
            ql,
        )
    )


def _is_shortlist(ql):
    return bool(re.search(r"\bshortlist\b|short[- ]?list|final list|final selection", ql))


def _is_rank(ql):
    return bool(
        re.search(
            r"\brank\b|\brecommend\b|best candidate|top candidate|strongest|most suitable|who should (we|i)|suitable for",
            ql,
        )
    )


def _is_jd(q, ql):
    return (
        "job description" in ql
        or ql.strip().startswith("jd:")
        or "responsibilities" in ql
        or (len(q) > 240 and ("requirement" in ql or "skills" in ql or "experience" in ql))
    )


# ---------------------------------------------------------------- intents
def answer(session, question: str, settings) -> dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return _help(session)
    ql = q.lower()

    if _is_reset(ql):
        return _do_show_all(session)
    if _is_count(ql):
        return _do_count(session)
    if _is_compare(ql):
        return _do_compare(session, q, settings)
    if _is_jd(q, ql):
        return _do_jd_rank(session, q, settings)
    if _is_exclude(ql):
        return _do_exclude(session, q, settings)
    yt = references.parse_years_threshold(q)
    if yt and "year" in ql:
        return _do_years_filter(session, q, yt)
    if _is_shortlist(ql):
        return _do_shortlist(session, q, settings)
    if _is_rank(ql):
        return _do_rank(session, q, settings)
    if _is_who_has(ql):
        return _do_who_has(session, q, settings)
    return _do_rank(session, q, settings)


def _help(session):
    md = (
        "I can: count candidates, find who has a skill, filter by years, compare "
        "candidates, rank/recommend for a role or job description, exclude, and build a "
        "shortlist. Ask away."
    )
    return _result("help", md, [], session)


def _do_count(session):
    n = len(session.profiles)
    nr = sum(1 for p in session.profiles if p["extraction"]["status"] == "NEEDS_REVIEW")
    dups = sum(1 for p in session.profiles if p["duplicate_of"])
    lines = [
        f"**{n} candidate(s)** in this batch — {n - nr} ready, {nr} flagged for review"
        + (f", {dups} duplicate(s)" if dups else "")
        + "."
    ]
    for p in session.profiles:
        lines.append(f"- {_label(p)} ({_years(p):g}y){_flags(p)}")
    return _result("count", "\n".join(lines), session.roster_order, session)


def _do_show_all(session):
    session.view_state["current_set"] = list(session.roster_order)
    session.view_state["last_result"] = list(session.roster_order)
    _set_history(session, "show_all", "reset working set to all candidates")
    return _do_count(session)


def _do_who_has(session, q, settings):
    terms = _extract_skill_terms(q)
    if not terms:
        return _help(session)
    ids = _current_ids(session)
    profiles = _profiles(session, ids)
    index = _index(session, profiles, settings)
    matched, rows = [], []
    for ci, p in enumerate(profiles):
        best = None
        for t in terms:
            m = scoring.match_skill_in_candidate(p, t, ci, index)
            if m:
                best = (t, m)
                break
        if best:
            _, m = best
            matched.append(p["candidate_id"])
            rows.append(
                f"- {_label(p)} — **{m['skill']}** ({m['how']}){_ev(m.get('evidence'))}{_flags(p)}"
            )
    label = ", ".join(terms)
    if matched:
        session.view_state["current_set"] = matched
        session.view_state["last_result"] = matched
        _set_history(session, "who_has", f"{label} -> {len(matched)} of {len(ids)}")
        head = f"**{len(matched)} of {len(ids)}** candidate(s) match **{label}**:"
        return _result("who_has", head + "\n" + "\n".join(rows), matched, session)
    session.view_state["last_result"] = []
    return _result(
        "who_has", f"No candidate in the current {len(ids)} matches **{label}**.", [], session
    )


def _do_years_filter(session, q, yt):
    op, n = yt
    ids = _current_ids(session)
    pid = _pid(session)
    if op == "gte":
        keep = [i for i in ids if _years(pid[i]) >= n]
        label = f"{n}+ years"
    else:
        keep = [i for i in ids if _years(pid[i]) < n]
        label = f"under {n} years"
    session.view_state["current_set"] = keep
    session.view_state["last_result"] = keep
    _set_history(session, "years_filter", f"{label}: {len(keep)} of {len(ids)}")
    rows = [f"- {_label(pid[i])} — {_years(pid[i]):g} years{_flags(pid[i])}" for i in keep]
    head = f"**{len(keep)} of {len(ids)}** candidate(s) have **{label}**:"
    return _result("years_filter", head + ("\n" + "\n".join(rows) if rows else ""), keep, session)


def _do_exclude(session, q, settings):
    ql = q.lower()
    ids = _current_ids(session)
    pid = _pid(session)
    yt = references.parse_years_threshold(q)
    refs = references.resolve_candidate_refs(q, session)
    if "weak" in ql or "flagged" in ql or "review" in ql:
        removed = [
            i
            for i in ids
            if pid[i]["extraction"]["status"] == "NEEDS_REVIEW"
            or pid[i]["extraction"]["field_coverage"] < 0.6
        ]
        reason = "flagged for review / low extraction confidence"
    elif yt:
        op, n = yt
        removed = [i for i in ids if (_years(pid[i]) < n if op == "lt" else _years(pid[i]) >= n)]
        reason = f"less than {n} years" if op == "lt" else f"{n}+ years"
    elif refs:
        removed = [i for i in refs if i in ids]
        reason = "named"
    else:
        return _result(
            "exclude",
            "Tell me what to exclude — e.g. “exclude candidates "
            "with less than 5 years”, “exclude the weak ones”, or “exclude candidate 3”.",
            [],
            session,
        )
    remaining = [i for i in ids if i not in removed]
    session.view_state["current_set"] = remaining
    session.view_state["last_result"] = remaining
    _set_history(session, "exclude", f"{reason}: -{len(removed)} -> {len(remaining)}")
    ex = ", ".join(f"Candidate {pid[i]['display_no']}" for i in removed) or "none"
    rows = [f"- {_label(pid[i])} ({_years(pid[i]):g}y){_flags(pid[i])}" for i in remaining]
    head = f"Excluded **{len(removed)}** ({reason}): {ex}.\n\n**{len(remaining)} remain:**"
    return _result("exclude", head + ("\n" + "\n".join(rows) if rows else ""), remaining, session)


def _do_compare(session, q, settings):
    refs = references.resolve_candidate_refs(q, session)
    if len(refs) < 2:
        return _result(
            "compare",
            "Tell me which two to compare, e.g. “compare candidate 2 and 5”.",
            refs,
            session,
        )
    refs = refs[:2]
    pid = _pid(session)
    a, b = pid[refs[0]], pid[refs[1]]
    session.view_state["last_result"] = refs

    def col(p):
        sk = ", ".join(s["name"] for s in p["skills"][:8]) or "—"
        edu = (
            "; ".join(
                (e.get("degree") or e.get("field") or e.get("institution") or "").strip()
                for e in p["education"]
            )
            or "—"
        )
        return p, sk, edu

    pa, ska, edua = col(a)
    pb, skb, edub = col(b)
    md = [
        f"**Comparison — {_label(a)} vs {_label(b)}**\n",
        f"| | {_label(a)} | {_label(b)} |",
        "|---|---|---|",
        f"| Headline | {pa['headline'] or '—'} | {pb['headline'] or '—'} |",
        f"| Years | {_years(a):g} | {_years(b):g} |",
        f"| Skills | {ska} | {skb} |",
        f"| Education | {edua} | {edub} |",
        f"| Status | {a['extraction']['status']} | {b['extraction']['status']} |",
    ]
    return _result("compare", "\n".join(md), refs, session)


def _do_rank(session, q, settings):
    crit = _criteria_from_query(q)
    ids = _current_ids(session)
    profiles = _profiles(session, ids)
    index = _index(session, profiles, settings)
    scored = scoring.rank(profiles, crit, index)
    session.view_state["last_ranking"] = [
        {"candidate_id": p["candidate_id"], "score": sc["score"]} for p, sc in scored
    ]
    session.view_state["last_result"] = [p["candidate_id"] for p, _ in scored]
    n = references.parse_top_n(q, default=min(len(scored), 10))
    head = f"**Ranking for “{crit['role_text']}”** (over {len(profiles)} candidate(s)):"
    return _result(
        "rank",
        head + "\n" + _render_ranking(scored[:n]),
        session.view_state["last_result"],
        session,
    )


def _do_shortlist(session, q, settings):
    crit = _criteria_from_query(q)
    ids = _current_ids(session)
    profiles = _profiles(session, ids)
    index = _index(session, profiles, settings)
    scored = scoring.rank(profiles, crit, index)
    n = references.parse_top_n(q, default=min(3, len(scored)))
    top = scored[:n]
    session.view_state["last_shortlist"] = [p["candidate_id"] for p, _ in top]
    session.view_state["last_result"] = session.view_state["last_shortlist"]
    _set_history(session, "shortlist", f"top {n} for '{crit['role_text']}'")
    head = f"**Shortlist — top {len(top)} for “{crit['role_text']}”:**"
    return _result(
        "shortlist",
        head + "\n" + _render_ranking(top, with_reasons=True),
        session.view_state["last_shortlist"],
        session,
    )


def _do_jd_rank(session, q, settings):
    crit = jd.parse_jd(q, settings)
    ids = _current_ids(session)
    profiles = _profiles(session, ids)
    index = _index(session, profiles, settings)
    scored = scoring.rank(profiles, crit, index)
    session.view_state["last_ranking"] = [
        {"candidate_id": p["candidate_id"], "score": sc["score"]} for p, sc in scored
    ]
    session.view_state["last_result"] = [p["candidate_id"] for p, _ in scored]
    mh = ", ".join(crit["must_have"]) or "—"
    head = (
        f"**Job-description match — “{crit['role_text']}”**\n"
        f"Must-have: {mh}"
        + (f" · Min years: {crit['min_years']}" if crit.get("min_years") else "")
        + f"\n\nRanked over {len(profiles)} candidate(s):"
    )
    return _result(
        "jd_match",
        head + "\n" + _render_ranking(scored[:10], with_reasons=True),
        session.view_state["last_result"],
        session,
    )


# ---------------------------------------------------------------- rendering
def _render_ranking(scored, with_reasons: bool = False) -> str:
    lines = []
    for rank_i, (p, sc) in enumerate(scored, 1):
        cov = (
            ""
            if sc["coverage"] is None
            else f", must-haves {len(sc['hits'])}/{len(sc['hits']) + len(sc['missing'])}"
        )
        lines.append(
            f"\n**{rank_i}. {_label(p)}** — score {sc['score']:.2f} "
            f"({_years(p):g}y{cov}, role-fit {sc['semantic']:.2f}){_flags(p)}"
        )
        if with_reasons and (sc["hits"] or sc["missing"]):
            for h in sc["hits"]:
                lines.append(
                    f"   - ✅ {h['requirement']} → {h['skill']} ({h['how']}){_ev(h.get('evidence'))}"
                )
            if sc["missing"]:
                lines.append(f"   - ❌ missing: {', '.join(sc['missing'])}")
        elif with_reasons:
            # no JD must-haves -> rationale = grounded skills + years
            for sk in [x for x in p["skills"] if x["source"] == "stated"][:4]:
                lines.append(f"   - ✅ {sk['name']}{_ev(sk.get('evidence'))}")
            lines.append(f"   - {_years(p):g} years of experience")
        else:
            top = ", ".join(s["name"] for s in p["skills"][:5]) or "—"
            lines.append(f"   - skills: {top}")
    return "\n".join(lines)


# ---------------------------------------------------------------- query parsing
# Instructional phrases that are NOT part of the skill/role — stripped before parsing.
_NOISE = [
    re.compile(p, re.I)
    for p in [
        r"\bshow (me )?(the )?evidence( for each( candidate)?)?\b",
        r"\bwith evidence\b",
        r"\bexplain( the)? reasons?\b",
        r"\bexplain why\b",
        r"\band why\b",
        r"\bwith reasons?( and evidence)?\b",
        r"\b(include|show)( the)? missing requirements?\b",
        r"\bmissing requirements?\b",
        r"\bside by side\b",
        r"\bin detail\b",
        r"\bfor each( candidate)?\b",
        r"\bplease\b",
        r"\bgive me\b",
        r"\bcan you\b",
        r"\bi want\b",
        r"\btell me\b",
        r"\bif available\b",
        r"\bfrom the cvs?\b",
        r"\bwould you recommend\b",
        r"\bfor an? interview( first)?\b",
    ]
]
_SKILL_LEAD = re.compile(
    r".*?\b(who (?:has|have|knows|is|are|with)|candidates? with|which candidates?|"
    r"has experience (?:in|with)|experience (?:in|with)|skilled in|proficient in|"
    r"strong(?:est)? in|good (?:at|in)|knows|what about|anyone with|"
    r"any candidates? with|list (?:the )?candidates? (?:with|who))\b",
    re.I,
)
_SKILL_STOP = re.compile(
    r"\b(experience|expertise|skills?|knowledge|background|using|candidates?|ones|people|"
    r"the|a|an|who|with|in|of)\b",
    re.I,
)


def _clean_query(q: str) -> str:
    s = " " + (q or "").lower() + " "
    for pat in _NOISE:
        s = pat.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_skill_terms(q: str) -> list[str]:
    s = _clean_query(q)
    s2 = _SKILL_LEAD.sub("", s, count=1)
    if s2 == s:  # no lead-in matched; take after 'with'/'in' if present
        m = re.search(r"\b(?:with|in)\s+(.+)$", s)
        s2 = m.group(1) if m else s
    s2 = _SKILL_STOP.sub(" ", s2)
    s2 = re.sub(r"[?.!]", " ", s2)
    parts = re.split(r",|/|\bor\b|\band\b", s2)
    terms = []
    for p in parts:
        t = re.sub(r"\s+", " ", p).strip()
        if len(t) >= 2 and t not in terms:
            terms.append(t)
    return terms[:6]


def _criteria_from_query(q: str) -> dict[str, Any]:
    ql = _clean_query(q)
    role = ""
    m = re.search(r"\bfor\s+(?:the\s+)?(?:role of\s+|position of\s+|a\s+|an\s+)?(.+)$", ql)
    if m:
        role = m.group(1)
    else:
        m = re.search(
            r"\b(?:best|top|strongest|recommend|suitable|rank)\s+"
            r"(?:\d+\s+)?(?:candidates?\s+)?(?:for\s+)?(.+)$",
            ql,
        )
        if m:
            role = m.group(1)
        else:
            m = re.search(r"\bstrong(?:est)? in\s+(.+)$", ql)
            role = m.group(1) if m else ""
    role = re.sub(r"[?.!]", "", role)
    role = re.sub(r"\b(candidates?|now|the|a|an|role|position|for)\b", " ", role)
    role = re.sub(r"\s+", " ", role).strip()
    yt = references.parse_years_threshold(q)
    return {
        "must_have": [],
        "min_years": yt[1] if (yt and yt[0] == "gte") else None,
        "role_text": role or "overall suitability",
        "source": "role",
    }
