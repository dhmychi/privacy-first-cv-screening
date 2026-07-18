"""Server-side markdown rendering so the calling client stays a pure pass-through
(single source of truth for formatting). Builds the post-analysis roster shown in
chat after a batch is analyzed."""

from __future__ import annotations

import re
from typing import Any

from ..reasons import describe as describe_reasons

DISCLAIMER = "\n\n---\n_Decision support for HR — not a final hiring decision._"


def roster_markdown(summary: dict[str, Any], roster: list[dict[str, Any]]) -> str:
    n = summary.get("candidate_count", len(roster))
    lines = [f"## 👥 CV Batch Analyzed — {n} candidate(s)"]
    meta = []
    if summary.get("needs_review"):
        meta.append(f"{summary['needs_review']} flagged for review")
    if summary.get("duplicates"):
        meta.append(f"{summary['duplicates']} duplicate(s)")
    if meta:
        lines.append("_" + ", ".join(meta) + "._")
    lines.append("")
    for r in roster:
        yrs = r.get("years")
        yrs = f"{yrs:g}y" if isinstance(yrs, int | float) else "—"
        skills = ", ".join(r.get("top_skills", [])[:5]) or "—"
        flag = f"  ⚠️ _{', '.join(r['flags'])}_" if r.get("flags") else ""
        lines.append(f"- **Candidate {r['no']} — {r['name']}** ({yrs}) · {skills}{flag}")
    lines.append(
        "\nAsk me anything about this batch: *who has a skill, compare candidates, "
        "filter by years, rank for a role, exclude, or build a shortlist.*"
    )
    return "\n".join(lines) + DISCLAIMER


def _skill_str(p):
    parts = []
    for s in p.get("skills", []):
        parts.append(s["name"])
    return ", ".join(parts) if parts else "(none extracted)"


def _exp_str(p):
    out = []
    for e in p.get("experiences", [])[:4]:
        seg = (e.get("title") or "").strip()
        if e.get("organization"):
            seg += f" @ {e['organization']}"
        dates = "-".join(x for x in [str(e.get("start") or ""), str(e.get("end") or "")] if x)
        if dates:
            seg += f" ({dates})"
        if seg.strip():
            out.append(seg.strip())
    return "; ".join(out) if out else "(none extracted)"


def _edu_str(p):
    out = []
    for e in p.get("education", []):
        seg = " ".join(
            x
            for x in [
                e.get("degree", ""),
                e.get("field", ""),
                e.get("institution", ""),
                str(e.get("year") or ""),
            ]
            if x
        ).strip()
        if seg:
            out.append(seg)
    return "; ".join(out) if out else "(none extracted)"


def facts_block(profiles, summary) -> str:
    """Compact, grounded per-candidate facts for the model to reason over. The model
    ranks / filters / compares using ONLY these facts (it must not invent)."""
    n = summary.get("candidate_count", len(profiles))
    # map candidate_id -> "#<display_no> (<name>)" so duplicate references use the
    # display number the user/model sees, not the internal candidate_id.
    disp = {}
    for p in profiles:
        nm = (p.get("identity", {}).get("full_name", {}) or {}).get(
            "value"
        ) or "(name not detected)"
        disp[p["candidate_id"]] = f"#{p['display_no']} ({nm})"
    head = [
        f"There are {n} analyzed candidate(s) in the uploaded batch. "
        f"Each candidate's verified facts (extracted from their CV, with the page "
        f"the content was found on) are listed below.\n"
    ]

    # SHARED-NAME NOTICE (deterministic, generic): when two or more NON-duplicate
    # candidates were extracted with the same name, the model tends to speculate
    # they are one person 'resubmitting' and to BLEND their separate histories
    # (a candidate-mixing error). State the fact authoritatively so the model
    # keeps them distinct. Names are compared case/space-insensitively; anyone
    # already marked STATUS = DUPLICATE is excluded (that is a real merge, not a
    # coincidence). No per-batch or per-name assumption - purely data-driven.
    def _norm_name(nm: str) -> str:
        return re.sub(r"\s+", " ", (nm or "").strip()).lower()

    name_groups: dict[str, list[dict[str, Any]]] = {}
    for p in profiles:
        if p.get("duplicate_of"):
            continue
        nm = (p.get("identity", {}).get("full_name", {}) or {}).get("value") or ""
        key = _norm_name(nm)
        if key and key != "(name not detected)":
            name_groups.setdefault(key, []).append(p)
    shared = [grp for grp in name_groups.values() if len(grp) > 1]
    if shared:
        notice = [
            "IMPORTANT - SHARED NAMES: some candidates share the SAME name but are "
            "SEPARATE people with different CVs (they were NOT merged). Treat each as a "
            "DISTINCT individual: do NOT combine or cross-reference their experience, "
            "education, employers or dates, and do NOT say they are 'the same person' or "
            "'resubmitted' unless a candidate is explicitly marked STATUS = DUPLICATE. "
            "When you flag this, name the specific candidates and state plainly that "
            "distinct people share a name and should be reviewed separately."
        ]
        for grp in sorted(shared, key=lambda g: min(x["display_no"] for x in g)):
            nm = (grp[0].get("identity", {}).get("full_name", {}) or {}).get("value")
            refs = ", ".join(
                f"#{p['display_no']}" for p in sorted(grp, key=lambda x: x["display_no"])
            )
            notice.append(f'- {len(grp)} distinct candidates share the name "{nm}": {refs}.')
        head.append("\n".join(notice) + "\n")
    # Deterministic experience-order rank (screening): non-duplicate, readable
    # candidates sorted by years desc (unknown years last), then by original
    # upload number for a stable total order. The model COPIES this rank -
    # it must never sort or number the roster itself.
    rankable = [
        p
        for p in profiles
        if not p.get("duplicate_of") and p["extraction"]["status"] != "NEEDS_REVIEW"
    ]

    def _rank_key(p):
        y = p["total_years_experience"]["value"]
        return (-(y if isinstance(y, int | float) else -1.0), p["display_no"])

    screening_rank = {
        p["candidate_id"]: i + 1 for i, p in enumerate(sorted(rankable, key=_rank_key))
    }
    for p in profiles:
        ident = p["identity"]
        name = ident["full_name"]["value"] or "(name not detected)"
        tye = p["total_years_experience"]
        yrs = tye["value"]
        yrs = f"{yrs:g}" if isinstance(yrs, int | float) else "not stated"
        if tye.get("note"):
            # stated-vs-computed contradiction: the computed value stays
            # authoritative; the note is surfaced for honest verification.
            yrs += f" (VERIFY: {tye['note']})"
        pages = sorted(
            {s["evidence"]["page"] for s in p.get("skills", []) if s.get("evidence")}
            | {e["evidence"]["page"] for e in p.get("experiences", []) if e.get("evidence")}
        )
        pg = ("p." + ",".join(str(x) for x in pages)) if pages else "CV"
        flags = p["extraction"]["reasons"]
        block = [
            f"### Candidate {p['display_no']} — {name}",
            f"- Years of experience: {yrs}",
            f"- Headline/role: {p.get('headline') or '(not stated)'}",
            f"- Skills: {_skill_str(p)}",
            f"- Experience: {_exp_str(p)}",
            f"- Education: {_edu_str(p)}",
            f"- Source evidence: {pg}",
        ]
        for vn in p.get("verification_notes") or []:
            block.append(f"- VERIFY: {vn}.")
        # Authoritative per-candidate status, matching the disjoint _summary partition
        # (duplicate takes priority over needs-review). The roster Status follows this.
        if p.get("duplicate_of"):
            block.append(
                f"- STATUS = DUPLICATE of {disp.get(p['duplicate_of'], 'an earlier candidate')} "
                f"— exclude this candidate from any ranking/fit table."
            )
        elif p["extraction"]["status"] == "NEEDS_REVIEW":
            why = describe_reasons(flags, "en")
            block.append(
                f"- STATUS = NEEDS REVIEW — {why}; "
                f"mark this candidate's Status 'Needs review' and describe the "
                f"issue with THAT phrase (never say 'could not be read' unless "
                f"that phrase says so)."
            )
        else:
            # Affirmative status so the model never INFERS one (e.g. from a
            # missing name): processed is processed, even without a name.
            block.append(
                "- STATUS = PROCESSED — extraction verified; show Status "
                "'Processed' (never 'Needs review') for this candidate."
            )
            rk = screening_rank.get(p["candidate_id"])
            if rk:
                block.append(
                    f"- SCREENING RANK = {rk} of {len(screening_rank)} "
                    f"(authoritative experience-order position — COPY it exactly "
                    f"into the roster's Rank column and sort rows by it; never "
                    f"recompute or renumber)."
                )
        head.append("\n".join(block))
    return "\n\n".join(head)
