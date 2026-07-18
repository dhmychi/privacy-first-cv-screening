"""Grounded requirement-reasoning judge (partial-redesign brain).

Replaces the brittle keyword+cosine matcher with a per-requirement REASONING
judgment made by the local LLM, hard-constrained to be evidence-grounded:

  * the model sees the FULL requirement text (meaning preserved, not a label) and
    a NUMBERED digest of the candidate's evidence IN CONTEXT (role/company/dates,
    explicit-vs-inferred, OCR/VLM source) - never a flat keyword list;
  * for each requirement it returns a verdict + the evidence item numbers it
    relied on + a one-line reason;
  * CODE then verifies every cited item exists and grounds it - an ungrounded
    "met" collapses to unverified, so hallucination is structurally bounded;
  * ARITHMETIC stays deterministic: total years / dates are computed in code and
    handed to the judge as facts; the judge only makes the semantic judgment
    (is this the RELEVANT experience? is this a RELATED degree? is this the
    SPECIFIC credential?), never the numbers.

Output is the SAME verdict list `engine.score_candidate` already consumes, so
scoring, caps, banding, ranking and the code-rendered report are unchanged. The
call is BATCHED (one LLM call per candidate) and temperature 0; the caller caches
per (candidate signature, jd hash) so repeats are byte-identical.

Selected only when CV_MATCHER=judge; otherwise the legacy matcher runs. Any
infra failure on a candidate falls back to the legacy matcher for THAT candidate,
so judge mode is never worse than legacy on error.
"""

from __future__ import annotations

from typing import Any

from ..pipeline import llm
from ..pipeline.extract import verify_quote, verify_term

# Engine-compatible verdict vocabulary. 'contradicted'/'review' are richer states
# the judge may emit; we keep the deterministic engine unchanged by mapping them
# to an engine verdict while preserving the nuance in the detail text.
_ENGINE_VERDICTS = {"met", "partial", "missing", "unverified", "not_assessable"}
_MAP = {"contradicted": "missing", "review": "unverified", "not applicable": "not_assessable"}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "req_id": {"type": "string"},
                    "verdict": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": ["req_id", "verdict"],
            },
        }
    },
    "required": ["verdicts"],
}

SYSTEM = (
    "You are an HR fit-assessment judge. You decide, for each job requirement, "
    "whether the candidate satisfies it, using ONLY the numbered CANDIDATE EVIDENCE. "
    "Reason about MEANING and CONTEXT, not keyword overlap.\n"
    "For EACH requirement output: req_id, verdict, evidence_quote (a SHORT verbatim "
    "substring copied EXACTLY from the CV text that supports your verdict - empty if "
    "missing/unverified), a one-line reason, and confidence (high|medium|low).\n"
    "VERDICTS (choose exactly, do not inflate):\n"
    "- met: the evidence clearly satisfies the requirement.\n"
    "- partial: related/adjacent but not a full match - a NEAR miss only (e.g. 4 of "
    "5 required years, a closely-related degree field, a general skill where a "
    "specific one is asked). A value FAR below the bar (e.g. 2 years vs 5+ required, "
    "or an unrelated field) is 'missing', NOT partial.\n"
    "- missing: the CV is readable but does NOT show this. **A required skill, "
    "technology, responsibility or credential that simply does not appear in a "
    "readable CV is 'missing' - NEVER 'unverified'. A candidate must not benefit "
    "from omitting a required item.**\n"
    "- unverified: use ONLY when the information needed to judge is genuinely "
    "absent-as-a-section or UNREADABLE - e.g. total years cannot be computed, the "
    "whole education/certifications section is missing, or the text is OCR-garbled. "
    "Not for 'the candidate lacks it' - that is 'missing'.\n"
    "- contradicted: the evidence conflicts with the requirement.\n"
    "JUDGEMENT RULES (apply by reasoning, they are not keyword tricks):\n"
    "- RELEVANT vs GENERAL experience: a years requirement that names a DOMAIN "
    "(e.g. 'software development', 'refinery maintenance') is met only by years in "
    "that domain. General tenure in an unrelated field is at most 'partial'. Sales "
    "experience is not software experience; HR experience is not engineering.\n"
    "- DEGREE relatedness: judge the field against THIS role. A degree of the right "
    "LEVEL but an unrelated FIELD (e.g. BA History for a Computer Science role) is "
    "'partial', not 'met'. A closely-related field is 'met'. A general degree is not "
    "automatically a technical degree.\n"
    "- SPECIFIC CREDENTIALS: a named certification/license (CPA, PMP, CCNA, PE, bar "
    "admission...) requires EXPLICIT evidence of THAT credential. Never infer it "
    "from a job title or role (being an 'Accountant' is NOT a CPA).\n"
    "- SPECIFIC vs GENERIC tech: if a requirement names a specific technology, a "
    "different one in the same family is 'partial' at most (SQL does not prove "
    "PostgreSQL; Java does not prove JavaScript).\n"
    "- ALTERNATIVES: 'Python, Java, or C#' is satisfied by ANY one - cite the "
    "specific one the candidate actually has; do not claim they have all.\n"
    "- PREFERRED vs REQUIRED: judge each on its own evidence; do not treat a "
    "preferred item as mandatory.\n"
    "- LANGUAGE: a CV written substantially IN a language is direct evidence the "
    "candidate reads and writes it - credit a 'language' requirement for the CV's "
    "own language as met (quote a representative phrase from that text).\n"
    "- OCR/VLM-sourced evidence is less certain: if a 'met' relies only on evidence "
    "marked (ocr) or (vlm), use confidence low and prefer 'partial'.\n"
    "- If the evidence is not present, say 'unverified' - NEVER guess 'met'.\n"
    "- Cite ONLY evidence item numbers that appear in the list. Do not invent."
)


def _src_tag(source: str | None) -> str:
    return " (ocr)" if source == "ocr" else (" (vlm)" if source == "vlm" else "")


_RAW_TEXT_BUDGET = 5000  # chars of raw CV text given to the judge to quote from


def build_digest(profile: dict[str, Any]) -> str:
    """CONTEXTUAL structured summary (roles+org+dates+domain, degree+field, skills,
    certs, languages, computed years) FOLLOWED BY the candidate's raw CV text, so
    the judge can reason about context AND quote the actual bullet that supports a
    verdict (responsibilities like 'design backend services' live in bullets, not
    in the structured fields). Grounding then verifies the quote is a real span."""
    lines: list[str] = []
    exps = profile.get("experiences") or []
    if exps:
        lines.append("EXPERIENCE (role @ org, dates):")
        for e in exps:
            dates = "-".join(x for x in [str(e.get("start") or ""), str(e.get("end") or "")] if x)
            head = " @ ".join(
                x
                for x in [
                    (e.get("title") or "").strip(),
                    (e.get("organization") or e.get("org") or "").strip(),
                ]
                if x
            )
            summ = (e.get("summary") or "").strip()
            lines.append(
                f"  - {head}" + (f" ({dates})" if dates else "") + (f" - {summ}" if summ else "")
            )
    edu = profile.get("education") or []
    if edu:
        lines.append("EDUCATION:")
        for e in edu:
            lines.append(
                "  - "
                + " ".join(
                    x
                    for x in [
                        str(e.get("degree") or ""),
                        ("in " + str(e.get("field"))) if e.get("field") else "",
                        ("- " + str(e.get("institution"))) if e.get("institution") else "",
                        str(e.get("year") or ""),
                    ]
                    if x
                ).strip()
            )
    sk = [s for s in (profile.get("skills") or []) if (s.get("name") or "").strip()]
    if sk:
        lines.append(
            "STATED SKILLS: "
            + ", ".join(
                (s["name"] + ("" if s.get("source") == "stated" else " [inferred]")) for s in sk
            )
        )
    for key, header in (("certifications", "CERTIFICATIONS"), ("languages", "LANGUAGES")):
        items = [
            c.get("name") + (f" ({c.get('level')})" if c.get("level") else "")
            for c in (profile.get(key) or [])
            if (c.get("name") or "").strip()
        ]
        if items:
            lines.append(f"{header}: " + ", ".join(items))
    ty = (profile.get("total_years_experience") or {}).get("value")
    if isinstance(ty, int | float):
        lines.append(f"COMPUTED TOTAL EXPERIENCE: {ty:g} years (authoritative - do not recompute).")

    # raw CV text (the source of truth the judge quotes from), with page markers
    raw_parts: list[str] = []
    for p in profile.get("_pages") or []:
        t = (p.get("text") or "").strip()
        if t:
            src = p.get("source") or "text"
            raw_parts.append(f"[page {p.get('page')}{_src_tag(src)}]\n{t}")
    raw = "\n".join(raw_parts)[:_RAW_TEXT_BUDGET]
    return "\n".join(lines) + "\n\nFULL CV TEXT (quote verbatim to support verdicts):\n" + raw


def _page_source(profile: dict[str, Any], page: int | None) -> str:
    for p in profile.get("_pages") or []:
        if p.get("page") == page:
            return p.get("source") or "text"
    return "text"


def _requirements_block(rubric: dict[str, Any]) -> str:
    out: list[str] = ["REQUIREMENTS to judge:"]
    for r in rubric.get("requirements") or []:
        if r["rtype"] == "soft":
            continue
        extra = []
        if r.get("min_years"):
            extra.append(f"min {r['min_years']:g} yrs")
        if r.get("degree_level"):
            extra.append(f"degree level {r['degree_level']}")
        tail = (" [" + "; ".join(extra) + "]") if extra else ""
        out.append(f'[{r["id"]}] {r["kind"]}, {r["rtype"]}: "{r["text"]}"{tail}')
    return "\n".join(out)


def _ground(profile: dict[str, Any], quote: str | None) -> dict[str, Any] | None:
    """Ground the judge's cited span: return {page, quote, source} iff the verbatim
    quote actually occurs in the candidate's text. None if it does not exist (so an
    ungrounded verdict cannot stand). WE assign the page, never the model."""
    q = (quote or "").strip()
    if len(q) < 4:
        return None
    pages = profile.get("_pages") or []
    hit = verify_quote(q, pages) or verify_term(q, pages, min_len=4)
    if hit:
        return {
            "page": hit.get("page"),
            "quote": q[:200],
            "source": _page_source(profile, hit.get("page")),
        }
    return None


def judge_candidate(
    settings, profile: dict[str, Any], rubric: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Per-requirement verdicts for ONE candidate via the reasoning judge.
    Returns the engine-compatible verdict list, or None on infra failure (caller
    falls back to the legacy matcher)."""
    reqs = [r for r in rubric.get("requirements") or [] if r["rtype"] != "soft"]
    if not reqs:
        return [
            {
                "req_id": r["id"],
                "verdict": "not_assessable",
                "detail": "personal trait - not assessable from a CV",
            }
            for r in rubric.get("requirements") or []
        ]
    digest = build_digest(profile)
    user = (
        "CANDIDATE EVIDENCE:\n"
        + (digest or "(no evidence extracted)")
        + "\n\n"
        + _requirements_block(rubric)
        + '\n\nReturn JSON {"verdicts":[{req_id, verdict, evidence_quote, reason, confidence}]} '
        "with EXACTLY one entry per requirement id above."
    )
    try:
        raw = llm.chat_json(
            settings,
            SYSTEM,
            user,
            schema=JUDGE_SCHEMA,
            timeout=getattr(settings, "score_timeout", 240),
            num_ctx=16384,
            task="judge",
        )
    except llm.LLMError:
        return None
    got = {}
    for v in raw.get("verdicts") or []:
        if isinstance(v, dict) and v.get("req_id"):
            got[str(v["req_id"])] = v

    out: list[dict[str, Any]] = []
    for r in rubric.get("requirements") or []:
        if r["rtype"] == "soft":
            out.append(
                {
                    "req_id": r["id"],
                    "verdict": "not_assessable",
                    "detail": "personal trait - not assessable from a CV",
                }
            )
            continue
        v = got.get(r["id"]) or {}
        verdict = str(v.get("verdict") or "unverified").strip().lower()
        verdict = _MAP.get(verdict, verdict)
        if verdict not in _ENGINE_VERDICTS:
            verdict = "unverified"
        reason = str(v.get("reason") or "")[:150]
        rt = r["rtype"]
        # Calibration guard (general, not profession-specific): these candidates
        # already passed the readability gate, so a skill/responsibility the CV
        # does not show is MISSING, not unverified - a candidate must not gain by
        # omitting a required item. 'unverified' stays valid only where absence is
        # genuinely unknowable: no certifications/languages section, or years that
        # could not be computed.
        if verdict == "unverified":
            if rt in ("skill", "responsibility"):
                verdict = "missing"
            elif rt == "certification" and (profile.get("certifications") or []):
                verdict = "missing"
            elif rt == "language" and (profile.get("languages") or []):
                verdict = "missing"
            elif (
                rt == "experience_years"
                and (profile.get("total_years_experience") or {}).get("value") is not None
            ):
                verdict = "missing"
        ev = None
        if verdict in ("met", "partial"):
            if rt in ("experience_years", "education"):
                # Structured / computed facts: the years are code-computed and the
                # degree fields are code-extracted+verified, so the judge's verdict
                # rests on data we already trust - do NOT demand a verbatim quote
                # (there is no literal "7 years" span). Attach the structured evidence.
                ev = _structured_evidence(profile, rt)
            else:
                # skill / responsibility / certification / language: require real
                # textual support. Try the judge's cited span first, then fall back
                # to the requirement's OWN keywords appearing in the CV. Only if
                # NOTHING supports it do we downgrade (anti-hallucination gate).
                ev = _ground(profile, v.get("evidence_quote")) or _ground_by_keywords(profile, r)
                if ev is None:
                    verdict = "unverified"
                    reason = "no verifiable evidence in CV - " + reason
                elif (
                    ev.get("source") in ("ocr", "vlm")
                    and verdict == "met"
                    and str(v.get("confidence", "")).lower() != "high"
                ):
                    verdict = "partial"  # OCR/VLM-only 'met' w/o high confidence -> verify
        detail = reason if reason else verdict
        rec = {"req_id": r["id"], "verdict": verdict, "detail": detail}
        if ev:
            rec["evidence"] = ev
        out.append(rec)
    return out


def _structured_evidence(profile: dict[str, Any], rtype: str) -> dict[str, Any] | None:
    """Evidence pointer for a code-verified structured verdict (years/degree)."""
    if rtype == "education":
        for e in profile.get("education") or []:
            evd = e.get("evidence") or {}
            if evd.get("page"):
                return {
                    "page": evd["page"],
                    "quote": (evd.get("quote") or "")[:200],
                    "source": _page_source(profile, evd["page"]),
                }
    # experience_years: point at the first dated role's evidence page
    for e in profile.get("experiences") or []:
        evd = e.get("evidence") or {}
        if evd.get("page"):
            return {
                "page": evd["page"],
                "quote": (evd.get("quote") or "")[:200],
                "source": _page_source(profile, evd["page"]),
            }
    return None


def _ground_by_keywords(profile: dict[str, Any], req: dict[str, Any]) -> dict[str, Any] | None:
    """Fallback grounding: does any of the requirement's own keywords literally
    occur in the candidate's text? Confirms real support when the judge's quote
    was paraphrased. Word-boundary, so no spurious substring hits."""
    pages = profile.get("_pages") or []
    for kw in req.get("keywords") or []:
        hit = verify_term(kw, pages, min_len=3)
        if hit:
            return {
                "page": hit.get("page"),
                "quote": kw[:120],
                "source": _page_source(profile, hit.get("page")),
            }
    return None
