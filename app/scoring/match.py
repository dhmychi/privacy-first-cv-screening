"""Requirement x candidate matching. Deterministic-first, embeddings-second,
NO per-candidate LLM (reproducible and fast at batch scale).

Verdicts per (candidate, requirement):
  met            - literal evidence found (word-boundary term hit, degree entry,
                   verified years >= minimum) -> full credit
  partial        - close-but-not-literal: near-miss years, 'or equivalent
                   experience' satisfied by tenure, or a semantically-equivalent
                   evidenced skill (bge-m3 cosine) -> half credit
  missing        - candidate's evidenced data does not satisfy it -> no credit
  unverified     - the CANDIDATE-side field needed to judge is absent/unreadable
                   (e.g. years unknown) -> excluded from the denominator, never
                   a silent zero
  not_assessable - soft traits a CV cannot evidence -> excluded from scoring

Every met/partial verdict carries evidence {page, quote, source} so the report
can cite it; OCR/VLM-sourced evidence is marked for interview verification.

Similarity thresholds are properties of the FIXED embedding model (bge-m3),
not of any dataset: >=SEM_MET cosine between short skill phrases means the
same concept differently worded; >=SEM_PARTIAL means closely related.
"""

from __future__ import annotations

import re
from typing import Any

from ..pipeline.anchors import compute_total_years
from ..pipeline.extract import verify_term
from ..query import embeddings
from .rubric import DEGREE_LEVELS

SEM_MET = 0.85
SEM_PARTIAL = 0.72

YEARS_PARTIAL_FRACTION = 0.75  # e.g. 6+ verified years against an 8-year floor
EQUIV_EXPERIENCE_MIN = 4.0  # 'or equivalent experience' needs real tenure


def _pages(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return profile.get("_pages") or []


def _page_source(profile: dict[str, Any], page_no: int | None) -> str:
    for p in _pages(profile):
        if p.get("page") == page_no:
            return p.get("source") or "text"
    return "text"


def _ev(profile: dict[str, Any], hit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hit:
        return None
    return {
        "page": hit.get("page"),
        "quote": hit.get("quote"),
        "source": _page_source(profile, hit.get("page")),
    }


def candidate_phrases(profile: dict[str, Any]) -> list[str]:
    """Evidenced short phrases describing the candidate (for semantic matching):
    stated skills + experience titles + education entries (so a degree's field
    can be semantically compared to a JD's required field). Deterministic order,
    deduplicated."""
    out: list[str] = []
    seen = set()
    for s in profile.get("skills") or []:
        n = (s.get("name") or "").strip()
        if n and s.get("source") == "stated" and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    for e in profile.get("experiences") or []:
        t = (e.get("title") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    for e in profile.get("education") or []:
        b = _entry_blob(e).strip()
        if b and b.lower() not in seen:
            seen.add(b.lower())
            out.append(b)
    return out[:70]


# Tenure vocabulary stripped from an experience_years requirement to isolate
# its DOMAIN qualifier ("4+ years of software development experience" ->
# "software development"). Bilingual, generic.
_TENURE_FILLER = {
    "year",
    "years",
    "yr",
    "yrs",
    "experience",
    "experiences",
    "professional",
    "minimum",
    "min",
    "least",
    "at",
    "of",
    "in",
    "with",
    "a",
    "an",
    "the",
    "or",
    "and",
    "plus",
    "hands",
    "on",
    "hands-on",
    "work",
    "working",
    "related",
    "relevant",
    "field",
    "سنوات",
    "سنة",
    "عام",
    "أعوام",
    "خبرة",
    "لا",
    "تقل",
    "عن",
    "في",
}


def _domain_terms(req: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for kw in (req.get("keywords") or []) + [req.get("text") or ""]:
        toks = [
            t
            for t in re.findall(r"[A-Za-z؀-ۿ][A-Za-z0-9؀-ۿ.+#&-]*", str(kw))
            if t.lower() not in _TENURE_FILLER
        ]
        term = " ".join(toks).strip()
        if len(term) >= 3 and term.lower() not in {t.lower() for t in out}:
            out.append(term)
    return out[:6]


def _exp_text(e: dict[str, Any]) -> str:
    return " ".join(str(e.get(k) or "") for k in ("title", "organization", "org", "summary"))


def _match_years(
    profile: dict[str, Any],
    req: dict[str, Any],
    req_vecs_i: list[list[float]] | None = None,
    phrase_vecs: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    ty = (profile.get("total_years_experience") or {}).get("value")
    need = req.get("min_years")
    if need is None:
        # Defense-in-depth: rubric post-validation reclassifies floorless
        # experience requirements to 'responsibility', so this should not be
        # reached. Never blanket-satisfy from mere tenure.
        return {
            "verdict": "unverified",
            "detail": "no stated years floor; cannot assess from tenure alone",
        }
    if ty is None:
        return {
            "verdict": "unverified",
            "detail": f"needs {need:g}+ years; total experience not readable from CV",
        }

    # If the JD qualifies the tenure with a DOMAIN ("... of software development
    # experience"), only years in evidenced-domain roles fully satisfy it —
    # general tenure alone earns at most partial credit, with an honest detail.
    terms = _domain_terms(req)
    if terms:
        label = terms[0][:40]
        # Content tokens (>=5 chars) of the domain terms: full phrases rarely
        # appear verbatim in role titles ('software development' vs 'Software
        # Developer'), but their distinctive tokens do. Tokens are matched in
        # the TITLE+SUMMARY only — an employer NAME must not domain-qualify a
        # role ('Gulf Software Trading' does not make a sales rep a developer).
        tokens = [
            w
            for t in terms
            for w in re.findall(r"[A-Za-z؀-ۿ]{5,}", t)
            if w.lower() not in _TENURE_FILLER
        ]
        exps = profile.get("experiences") or []
        relevant = []
        for e in exps:
            full_pages = [{"page": 1, "text": _exp_text(e)}]
            ts_pages = [
                {"page": 1, "text": " ".join(str(e.get(k) or "") for k in ("title", "summary"))}
            ]
            hit = any(verify_term(t, full_pages, min_len=3) for t in terms) or any(
                verify_term(w, ts_pages, min_len=4) for w in tokens
            )
            if not hit and req_vecs_i and phrase_vecs:
                tv = phrase_vecs.get((e.get("title") or "").strip().lower())
                if tv:
                    sim = max((embeddings.cosine(rv, tv) for rv in req_vecs_i if rv), default=0.0)
                    hit = sim >= SEM_PARTIAL
            if hit:
                relevant.append(e)
        dy, _n = compute_total_years(relevant) if relevant else (0.0, 0)
        # domain years can never exceed the verified total
        dy = min(dy or 0.0, ty)
        if dy >= need:
            return {"verdict": "met", "detail": f"{dy:g} years in {label} vs {need:g}+ required"}
        if ty >= need:
            return {
                "verdict": "partial",
                "detail": (
                    f"{ty:g} years total, but only {dy:g} evidenced in "
                    f"{label} — verify domain depth at interview"
                ),
            }
    if ty >= need:
        return {"verdict": "met", "detail": f"{ty:g} years vs {need:g}+ required"}
    if ty >= need * YEARS_PARTIAL_FRACTION:
        return {"verdict": "partial", "detail": f"{ty:g} years vs {need:g}+ required (close)"}
    return {"verdict": "missing", "detail": f"{ty:g} years vs {need:g}+ required"}


_LEVEL_ORDER = ["diploma", "bachelor", "master", "phd"]

# Word-boundary degree-level matching: raw substring checks turned every
# "diploMA " into a master's ("ma " tail) and could hit "ba " inside words.
# Tokens must appear as standalone words/abbreviations.
_DEGREE_RES: dict[str, list[re.Pattern]] = {
    level: [
        re.compile(r"(?<![a-z0-9؀-ۿ])" + re.escape(kw.strip()) + r"(?![a-z0-9؀-ۿ])") for kw in kws
    ]
    for level, kws in DEGREE_LEVELS.items()
}


def _degree_rank(text: str) -> int | None:
    low = (text or "").lower()
    best = None
    for i, level in enumerate(_LEVEL_ORDER):
        for pat in _DEGREE_RES[level]:
            if pat.search(low):
                best = max(best, i) if best is not None else i
                break
    return best


# Vocabulary stripped from JD keywords to isolate the FIELD of study
# ("Bachelor's degree in Computer Science or related field" -> "computer science").
_EDU_FILLER = {
    "degree",
    "in",
    "of",
    "a",
    "an",
    "or",
    "the",
    "and",
    "with",
    "closely",
    "related",
    "field",
    "discipline",
    "equivalent",
    "experience",
    "required",
    "preferred",
    "بكالوريوس",
    "ماجستير",
    "دكتوراه",
    "دبلوم",
    "درجة",
    "في",
    "أو",
    "تخصص",
    "ذي",
    "صلة",
    "خبرة",
}
_DEGREE_WORDS = {w.strip() for kws in DEGREE_LEVELS.values() for w in kws} | {
    "bachelor's",
    "master's",
    "bachelors",
    "masters",
}


def _edu_field_terms(req: dict[str, Any]) -> list[str]:
    """Field-of-study terms from the requirement's own JD keywords (generic:
    whatever fields the JD names). Empty when the JD asks for a level only."""
    out: list[str] = []
    for kw in (req.get("keywords") or []) + [req.get("text") or ""]:
        toks = [
            t
            for t in re.findall(r"[A-Za-z؀-ۿ][A-Za-z0-9؀-ۿ.+#&-]*", str(kw))
            if t.lower() not in _EDU_FILLER and t.lower() not in _DEGREE_WORDS
        ]
        term = " ".join(toks).strip()
        if len(term) >= 2 and term.lower() not in {t.lower() for t in out}:
            out.append(term)
    return out[:6]


def _entry_blob(e: dict[str, Any]) -> str:
    return " ".join(str(e.get(k) or "") for k in ("degree", "field", "institution"))


def _match_education(
    profile: dict[str, Any],
    req: dict[str, Any],
    req_vec: list[float] | None = None,
    phrase_vecs: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    entries = profile.get("education") or []
    want = req.get("degree_level") or ""
    want_rank = _LEVEL_ORDER.index(want) if want in _LEVEL_ORDER else None
    best_rank, best_entry = None, None
    for e in entries:
        r = _degree_rank(_entry_blob(e))
        if r is not None and (best_rank is None or r > best_rank):
            best_rank, best_entry = r, e
    if best_rank is not None and (want_rank is None or best_rank >= want_rank):
        ev = best_entry.get("evidence") if isinstance(best_entry, dict) else None
        deg = (best_entry.get("degree") or _LEVEL_ORDER[best_rank]) if best_entry else ""
        # Level satisfied. If the JD names a FIELD of study, the field must be
        # evidenced too — a Bachelor of Arts in History must not fully match
        # "Bachelor's in Computer Science or related field".
        field_terms = _edu_field_terms(req)
        if not field_terms:
            return {
                "verdict": "met",
                "detail": f"degree evidenced: {deg}"[:120],
                "evidence": _ev(profile, ev),
            }
        edu_pages = [
            {"page": (e.get("evidence") or {}).get("page") or 1, "text": _entry_blob(e)}
            for e in entries
        ]
        for ft in field_terms:
            if verify_term(ft, edu_pages, min_len=2):
                return {
                    "verdict": "met",
                    "detail": f"degree evidenced: {deg} (field: {ft})"[:130],
                    "evidence": _ev(profile, ev),
                }
        # Literal miss -> semantic relatedness of the candidate's own degree
        # string vs the requirement ("Information Systems" ~ "Computer Science").
        if req_vec and phrase_vecs:
            best_sim = 0.0
            for e in entries:
                v = phrase_vecs.get(_entry_blob(e).strip().lower())
                if v:
                    best_sim = max(best_sim, embeddings.cosine(req_vec, v))
            if best_sim >= SEM_PARTIAL:
                return {
                    "verdict": "met",
                    "detail": f"degree evidenced: {deg} (related field)"[:130],
                    "evidence": _ev(profile, ev),
                    "semantic": round(best_sim, 3),
                }
        cand_field = _entry_blob(best_entry).strip()[:40] if best_entry else ""
        return {
            "verdict": "partial",
            "detail": (
                f"degree level met, but field ({cand_field}) does not match "
                f"required {field_terms[0]}"
            )[:150],
            "evidence": _ev(profile, ev),
        }
    if entries and want_rank is not None and best_rank is not None:
        return {"verdict": "missing", "detail": f"highest evidenced degree below required {want}"}
    # No parseable education at all
    if req.get("equivalent_experience_ok"):
        ty = (profile.get("total_years_experience") or {}).get("value")
        if ty is not None and ty >= EQUIV_EXPERIENCE_MIN:
            return {
                "verdict": "partial",
                "detail": f"no degree evidenced; {ty:g} years experience "
                "(JD allows equivalent experience)",
            }
    if not entries:
        return {"verdict": "unverified", "detail": "education not readable from CV"}
    return {"verdict": "missing", "detail": "required degree not evidenced"}


def _match_terms(profile: dict[str, Any], req: dict[str, Any]) -> dict[str, Any] | None:
    """Literal keyword evidence anywhere in the candidate's text (word-boundary,
    so 'R' never matches inside 'Recruitment'). ANY keyword satisfies (JD
    alternatives are alternatives)."""
    pages = _pages(profile)
    hits: list[tuple[str, dict[str, Any]]] = []
    for kw in req.get("keywords") or []:
        h = verify_term(kw, pages, min_len=2)
        if h:
            hits.append((kw, h))
    if hits:
        kw, h = hits[0]
        others = [k for k, _ in hits[1:3]]
        detail = kw if not others else kw + ", " + ", ".join(others)
        return {
            "verdict": "met",
            "detail": f"evidenced: {detail}"[:140],
            "evidence": _ev(profile, h),
        }
    return None


def match_candidate(
    profile: dict[str, Any],
    rubric: dict[str, Any],
    req_vecs: list[list[list[float]] | None],
    phrase_vecs: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """All requirement verdicts for one candidate. req_vecs holds a LIST of
    vectors per requirement (one per JD keyword — sharper alternatives and
    cross-lingual matching than a single joined phrase); phrase_vecs maps the
    candidate's evidenced phrases. Both may be empty -> literal-only matching."""
    phrases = candidate_phrases(profile)
    out: list[dict[str, Any]] = []
    for i, req in enumerate(rubric.get("requirements") or []):
        rtype = req["rtype"]
        rv = req_vecs[i] if i < len(req_vecs) else None
        if rtype == "soft":
            res: dict[str, Any] = {
                "verdict": "not_assessable",
                "detail": "personal trait - not assessable from a CV",
            }
        elif rtype == "experience_years":
            res = _match_years(profile, req, rv, phrase_vecs)
        elif rtype == "education":
            res = _match_education(profile, req, rv[0] if rv else None, phrase_vecs)
        else:  # skill / responsibility / certification / language
            res = _match_terms(profile, req) or {}
            # A certification/license is a SPECIFIC named credential - it must be
            # matched by literal evidence (its name/abbreviation), never inferred
            # by semantic role-proximity. Otherwise "CPA" would partial-match the
            # job title "Accountant" (cos 0.73), "PMP" would match "Project
            # Manager", "CCNA" would match "Network Engineer" - crediting a
            # credential the candidate does not hold. Skills/responsibilities may
            # still be semantically evidenced (recruitment ~ talent acquisition).
            if not res and rtype != "certification":
                res = _semantic(req, i, req_vecs, phrases, phrase_vecs) or {}
            if not res:
                # Absence of the whole SECTION is not evidence of absence: a
                # certification/language requirement with no hit AND no parsed
                # certifications/languages at all is UNVERIFIED (excluded from
                # the denominator, 'verify at interview'), never a definitive
                # miss that could trip a hard-minimum gate.
                section = {"certification": "certifications", "language": "languages"}.get(rtype)
                if section is not None and not (profile.get(section) or []):
                    res = {
                        "verdict": "unverified",
                        "detail": f"no {section} readable from CV - verify at interview",
                    }
                else:
                    res = {"verdict": "missing", "detail": "not evidenced in CV"}
        res["req_id"] = req["id"]
        out.append(res)
    return out


def _semantic(
    req: dict[str, Any],
    req_idx: int,
    req_vecs: list[list[list[float]] | None],
    phrases: list[str],
    phrase_vecs: dict[str, list[float]],
) -> dict[str, Any] | None:
    rvs = req_vecs[req_idx] if req_idx < len(req_vecs) else None
    rvs = [v for v in (rvs or []) if v]
    if not rvs or not phrases:
        return None
    best_sim, best_phrase = 0.0, ""
    for ph in phrases:
        v = phrase_vecs.get(ph.lower())
        if not v:
            continue
        # best alternative wins: 'Python' matches 'بايثون' even when the other
        # alternatives (Java, C#) do not.
        sim = max(embeddings.cosine(rv, v) for rv in rvs)
        if sim > best_sim:
            best_sim, best_phrase = sim, ph
    if best_sim >= SEM_MET:
        return {
            "verdict": "met",
            "detail": f"equivalent evidenced skill: {best_phrase}"[:140],
            "semantic": round(best_sim, 3),
        }
    if best_sim >= SEM_PARTIAL:
        return {
            "verdict": "partial",
            "detail": f"related evidenced skill: {best_phrase}"[:140],
            "semantic": round(best_sim, 3),
        }
    return None


def requirement_vectors(settings, rubric: dict[str, Any]) -> list[list[list[float]] | None]:
    """Embed each requirement's keywords INDIVIDUALLY (plus its text as a
    fallback) so JD alternatives stay sharp ('Python, Java, or C#' = three
    vectors, not one diluted blend) and cross-lingual keyword pairs land.
    Degrades to [None]*n (literal-only) when the embedder is unavailable."""
    reqs = rubric.get("requirements") or []
    if not reqs:
        return []
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for r in reqs:
        kws = [str(k) for k in (r.get("keywords") or [])[:4]] or [r["text"]]
        spans.append((len(texts), len(kws)))
        texts.extend(kws)
    vecs = embeddings.embed(settings, texts)
    if not vecs or len(vecs) != len(texts):
        return [None] * len(reqs)
    return [vecs[s : s + n] for s, n in spans]


def phrase_vectors(settings, profiles: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Embed all candidates' evidenced phrases. These depend ONLY on the
    analyzed profiles (not the JD), so the caller caches them per session and
    reuses them across every job description. Degrades to {} (literal-only)."""
    all_phrases: list[str] = []
    seen = set()
    for p in profiles:
        for ph in candidate_phrases(p):
            if ph.lower() not in seen:
                seen.add(ph.lower())
                all_phrases.append(ph)
    if not all_phrases:
        return {}
    vecs = embeddings.embed(settings, all_phrases)
    if not vecs or len(vecs) != len(all_phrases):
        return {}
    return {ph.lower(): v for ph, v in zip(all_phrases, vecs, strict=False)}
