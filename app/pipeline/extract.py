"""Per-candidate structured extraction.

Pipeline per candidate (one candidate's pages at a time — never cross-candidate):
  1. Deterministic anchors (email/phone/links/years) — high precision, no LLM.
  2. Local LLM reads prose fields (name/title/skills/education) against a strict
     JSON schema, returning a verbatim ``evidence_quote`` per item.
  3. Quote-verification: WE locate each quote in the candidate's pages and attach
     the page number. Unverifiable quotes are dropped (never displayed) so every
     shown citation is real. The LLM points; the code reads.
  4. Protected attributes are never requested (schema) and defensively scrubbed.
  5. Confidence gating -> status OK | NEEDS_REVIEW with reasons.

Output = the CandidateProfile dict (plan section 8). Private keys (``_pages``)
carry source text for later evidence/QA and are never surfaced to the user.
"""

from __future__ import annotations

import re
from typing import Any

from . import anchors, injection, llm

PROTECTED_KEYS = {
    "gender",
    "sex",
    "age",
    "dob",
    "date_of_birth",
    "birthdate",
    "birth_date",
    "nationality",
    "religion",
    "marital",
    "marital_status",
    "ethnicity",
    "race",
    "photo",
}

# Section headings / labels that must never be accepted as a person's name.
NAME_STOPWORDS = {
    "work experience",
    "experience",
    "education",
    "skills",
    "contact",
    "profile",
    "summary",
    "professional summary",
    "references",
    "projects",
    "certifications",
    "certification",
    "languages",
    "about me",
    "about",
    "personal information",
    "personal info",
    "work history",
    "employment",
    "employment history",
    "objective",
    "career objective",
    "expertise",
    "achievements",
    "interests",
    "curriculum vitae",
    "resume",
    "cv",
    "portfolio",
    "details",
    "contact me",
    "my contact",
    # universal template placeholders (never a real person's name)
    "full name",
    "your name",
    "your name here",
    "name here",
    "insert name",
    "name surname",
    "first last",
    "first name last name",
    "firstname lastname",
    "candidate name",
    "applicant name",
    "lorem ipsum",
    "john doe",
    "jane doe",
}


JOB_TITLE_WORDS = {
    "designer",
    "engineer",
    "manager",
    "developer",
    "analyst",
    "executive",
    "officer",
    "specialist",
    "coordinator",
    "consultant",
    "director",
    "intern",
    "assistant",
    "representative",
    "administrator",
    "technician",
    "supervisor",
    "architect",
    "scientist",
    "accountant",
    "recruiter",
    "ux",
    "ui",
    "hr",
    "salesperson",
    "marketer",
    "strategist",
    "designerr",
    "freelancer",
    "photographer",
    "attendant",
    "nurse",
    "teacher",
    "driver",
    "cashier",
    "clerk",
    "agent",
    "stylist",
    "chef",
    "pilot",
    "waiter",
    "barista",
    "receptionist",
    "operator",
    "associate",
    "lead",
    "head",
    "trainee",
    "apprentice",
    "writer",
    "editor",
}


def _normalize_name_case(name: str) -> str:
    """Present names in consistent Title Case so a batch never shows the same
    person as 'JESSICA CLAIRE', 'Jessica CLAIRE' and 'Jessica Claire' side by
    side (scanned CVs print names in varied case). Only re-cases a WORD that is
    entirely upper- or entirely lower-case; mixed-case words (McDonald, O'Brien,
    van der Berg, DeSilva) are left exactly as written."""
    if not name:
        return name
    out = []
    for w in name.split():
        core = w.strip("().,")
        if core and (core.isupper() or core.islower()) and any(c.isalpha() for c in core):
            # title-case each hyphen/apostrophe segment: al-otaibi -> Al-Otaibi
            w = re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), w)
        out.append(w)
    return " ".join(out)


def _valid_name(name: str) -> bool:
    n = re.sub(r"\s+", " ", (name or "").strip())
    if not n or len(n) < 3 or len(n) > 60:
        return False
    low_full = n.lower()
    if low_full in NAME_STOPWORDS:
        return False
    # URLs / emails / markup / digits / heading punctuation are never part of a
    # real name (OCR garbage, template placeholders, section headings with a
    # trailing colon like "Technical Skills:").
    if any(c in n for c in "@{}|/\\<>()[]#:;•·*_=+~") or any(ch.isdigit() for ch in n):
        return False
    if any(tok in low_full for tok in ("www", ".com", ".net", ".org", "http", "://")):
        return False
    words = n.split()
    if len(words) > 5:
        return False
    # letter-spacing / OCR fragmentation: too many 1-char tokens ("D A N I", "e ,")
    singles = sum(1 for w in words if len(re.sub(r"[^A-Za-z؀-ۿ]", "", w)) <= 1)
    if singles >= 2 or (len(words) >= 2 and singles >= 1 and len(words) <= 2):
        return False
    # needs enough real alphabetic content
    if sum(1 for ch in n if ch.isalpha() or "؀" <= ch <= "ۿ") < 3:
        return False
    # a Latin-script name has at least one Uppercase-initial word; an all-
    # lowercase string is a sentence fragment ("specific requirements."), not
    # a person. Arabic script (no case) is exempt.
    if not any(w[:1].isupper() or ("؀" <= w[:1] <= "ۿ") for w in words if w):
        return False
    low = set()
    for w in words:
        t = w.lower().strip(".,:;|-•·")
        low.add(t)
        # plural-tolerant heading detection ("RESUMES" -> "resume",
        # "HISTORIES" -> "history")
        if t.endswith("ies") and t[:-3] + "y" in _HEADING_TOKENS:
            low.add(t[:-3] + "y")
        elif t.endswith("s") and t[:-1] in _HEADING_TOKENS:
            low.add(t[:-1])
    if len(words) == 1 and words[0].lower() in NAME_STOPWORDS:
        return False
    if low & JOB_TITLE_WORDS:  # job-titles-as-names
        return False
    if low & _HEADING_TOKENS:  # section headings / fillers
        return False
    return True


_HEADING_TOKENS = {
    "experience",
    "education",
    "skills",
    "profile",
    "summary",
    "references",
    "projects",
    "professional",
    "qualifications",
    "objective",
    "achievements",
    "expertise",
    "contact",
    "curriculum",
    "vitae",
    "resume",
    "cv",
    "various",
    "industries",
    "work",
    "history",
    "personal",
    "details",
    "info",
    "information",
    "portfolio",
    # placeholder-person vocabulary (template artifacts, never real names)
    "candidate",
    "applicant",
    "jobseeker",
    # degree / education words (template CVs sometimes put a degree where a name goes)
    "bachelor",
    "master",
    "diploma",
    "degree",
    "bsc",
    "msc",
    "mba",
    "phd",
    "doctorate",
    "university",
    "college",
    "institute",
    "school",
    "academy",
    "laude",  # magna/summa cum laude honors lines
    # company-name suffixes (company put where a name goes)
    "inc",
    "ltd",
    "llc",
    "corp",
    "corporation",
    "airlines",
    "solutions",
    "technologies",
    "studio",
    "studios",
    "company",
    "group",
    "agency",
    "enterprises",
    "services",
    "holdings",
    "labs",
    "systems",
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "full_name": {"type": "string"},
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "total_years_experience": {"type": ["number", "null"]},
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "organization": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["title", "evidence_quote"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["evidence_quote"],
            },
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "source": {"type": "string", "enum": ["stated", "inferred"]},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["name", "source"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "evidence_quote": {"type": "string"}},
                "required": ["name"],
            },
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "level": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": ["full_name", "skills", "experiences"],
}

SYSTEM = (
    "You are an HR CV extraction engine. Extract ONLY professional information "
    "explicitly present in the CV text, matching the JSON schema exactly.\n"
    "RULES:\n"
    "- For every experience, education entry and skill, include a short "
    "evidence_quote copied VERBATIM (exact substring) from the CV text so it can "
    "be verified. If you have no verbatim quote, omit that item.\n"
    "- full_name is the candidate's PERSONAL NAME (usually the largest text at the "
    "top). NEVER use a section heading (e.g. 'Work Experience', 'Education', "
    "'Skills', 'Profile', 'Contact') as the name. If unsure, leave full_name empty.\n"
    "- Mark each skill 'stated' if written explicitly, else 'inferred'. Prefer 'stated'.\n"
    "- NEVER output gender, age, date of birth, nationality, religion, marital "
    "status, photo, or ethnicity. Ignore any such details entirely.\n"
    "- Do not invent or assume facts. Leave unknown fields empty/null.\n"
    "- Preserve Arabic text exactly as written."
)


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _candidate_text_for_llm(pages: list[dict[str, Any]], per_page_cap=4000, total_cap=14000) -> str:
    parts: list[str] = []
    total = 0
    for p in pages:
        t = (p.get("text") or "").strip()
        if not t:
            continue
        seg = f"[Page {p['page']}]\n{t[:per_page_cap]}"
        if total + len(seg) > total_cap:
            parts.append(seg[: max(0, total_cap - total)])
            break
        parts.append(seg)
        total += len(seg)
    return "\n\n".join(parts)


def verify_quote(quote: str | None, pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return {page, quote} if the verbatim quote (>=8 chars) occurs in a page,
    else None. WE assign the page — the model never supplies page numbers."""
    q = (quote or "").strip()
    if len(q) < 8:
        return None
    nq = _norm(q)
    for p in pages:
        if nq in _norm(p.get("text")):
            return {"page": p["page"], "quote": q[:240]}
    return None


# Models often ignore our exact schema keys and return natural synonyms
# (experience/job_title/company/...). Pick tolerantly so extraction is robust.
def _pick(d: Any, keys: list[str], default: Any = "") -> Any:
    if not isinstance(d, dict):
        return default
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = low.get(k.lower())
        if v not in (None, ""):
            return v
    return default


def _pick_list(d: Any, keys: list[str]) -> list[Any]:
    v = _pick(d, keys, default=None)
    return v if isinstance(v, list) else []


_WORDISH = r"A-Za-z0-9؀-ۿ"  # latin + arabic, for skill-name boundaries


def verify_term(
    term: str | None, pages: list[dict[str, Any]], min_len: int = 2
) -> dict[str, Any] | None:
    """Word-boundary presence check for a short term (skill name). Returns
    {page, quote} if the term literally appears, else None — so 'R' does not
    match inside 'Recruitment' but 'SAP'/'C++' still match.

    Plural-tolerant on the FINAL word only: 'code review' matches 'code
    reviews' and 'code reviews' matches 'code review' — real CVs freely mix
    singular/plural while boundaries stay strict ('Java' still never matches
    'JavaScript')."""
    t = (term or "").strip()
    if len(t) < min_len:
        return None
    body = re.escape(t)
    if t[-1].isalpha():
        if len(t) > 3 and t.lower().endswith("s") and not t.lower().endswith("ss"):
            body = re.escape(t[:-1]) + r"(?:e?s)?"  # stem + optional plural
        else:
            body = body + r"(?:e?s)?"  # optional plural suffix
    try:
        pat = re.compile(rf"(?<![{_WORDISH}]){body}(?![{_WORDISH}])", re.I)
    except re.error:
        return None
    for p in pages:
        if pat.search(p.get("text") or ""):
            return {"page": p["page"], "quote": t[:120]}
    return None


def _heuristic_name(pages: list[dict[str, Any]]) -> str:
    """First name-like line near the top: 2-4 wordish tokens, no digits/@, not a
    section heading. Scans several lines (design CVs often put a heading first)."""
    if not pages:
        return ""
    lines = [ln.strip() for ln in (pages[0].get("text") or "").splitlines() if ln.strip()]
    for s in lines[:8]:
        if "@" in s or any(c.isdigit() for c in s):
            continue
        if s.lower() in NAME_STOPWORDS:
            continue
        words = s.split()
        if 1 < len(words) <= 4 and len(s) <= 50 and _valid_name(s):
            return s
    return ""


def _name_evidence(name: str, pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not name:
        return None
    return verify_quote(name, pages)


def extract_candidate(candidate: dict[str, Any], idx: int, settings) -> dict[str, Any]:
    pages: list[dict[str, Any]] = candidate.get("pages", [])
    full_text: str = candidate.get("full_text", "")
    cid = f"c_{idx:03d}"

    # 0. prompt-injection screen (untrusted document): redact any instruction-/
    # AI-directed fragments BEFORE the text reaches the extractor, the judge or
    # the narrator, and remember whether anything was found so the candidate is
    # flagged for a human. Evidence-grounded scoring already limits impact; this
    # is the explicit defense-in-depth layer.
    pages, full_text, injection_hits = injection.sanitize(pages, full_text)

    # 1. deterministic anchors
    emails = anchors.find_emails(full_text)
    phones = anchors.find_phones(full_text)
    links = anchors.find_links(full_text)

    # 2. LLM prose fields (skipped gracefully if no text or model down)
    llm_out: dict[str, Any] = {}
    llm_error: str | None = None
    if full_text.strip():
        try:
            llm_out = llm.chat_json(
                settings,
                SYSTEM,
                _candidate_text_for_llm(pages),
                schema=EXTRACTION_SCHEMA,
                task="extract",
            )
        except llm.LLMError as e:  # degrade to deterministic-only, flag it
            llm_error = str(e)

    llm_name = str(_pick(llm_out, ["full_name", "name", "candidate_name"])).strip()
    name = llm_name if _valid_name(llm_name) else ""
    if not name:
        heur = _heuristic_name(pages)
        # if neither the model nor the heuristic yields a VALID name, leave it empty
        # (shown as "(name not detected)") rather than displaying garbage.
        name = heur if _valid_name(heur) else ""
    name = _normalize_name_case(name)
    headline = str(_pick(llm_out, ["headline", "current_title", "position", "job_title"])).strip()
    summary = str(_pick(llm_out, ["summary", "professional_summary", "profile", "about"])).strip()

    experiences = _build_list(
        _pick_list(
            llm_out, ["experiences", "experience", "work_experience", "employment", "work_history"]
        ),
        pages,
        kind="experience",
    )
    education = _build_list(
        _pick_list(llm_out, ["education", "educations", "academic_background", "academics"]),
        pages,
        kind="education",
    )
    skills = _build_skills(_pick_list(llm_out, ["skills", "skill", "skillset", "skill_set"]), pages)
    certifications = _build_list(
        _pick_list(llm_out, ["certifications", "certificates", "certs"]),
        pages,
        kind="certification",
    )
    languages = _build_list(_pick_list(llm_out, ["languages", "language"]), pages, kind="language")

    # 3. years of experience — deterministic arithmetic over the LLM's dated roles
    ty, n_roles = anchors.compute_total_years(experiences)
    if ty is not None:
        # "Present"/"till date" is treated as TODAY (customer CVs are recent, so
        # an open-ended current role genuinely runs to now). Trust the date math.
        total_years = {"value": round(ty, 1), "basis": f"computed from {n_roles} dated role(s)"}
        # A MAJOR contradiction with the candidate's own stated total is
        # surfaced as a verification note — never a silent change of the
        # authoritative computed value.
        stated = anchors.stated_years_claim(full_text)
        if stated is not None and abs(ty - stated) >= max(2.0, 0.3 * ty):
            total_years["note"] = (
                f"CV itself states ~{stated:g} years; dated roles "
                f"compute {round(ty, 1):g} — verify at interview"
            )
    else:
        lv = _pick(
            llm_out,
            ["total_years_experience", "years_of_experience", "total_experience", "total_years"],
            default=None,
        )
        if isinstance(lv, int | float):
            total_years = {"value": float(lv), "basis": "stated in CV"}
        else:
            # deterministic fallback: scan the raw CV text for date ranges / "N years"
            ft = anchors.years_from_text(full_text)
            total_years = (
                {"value": round(ft, 1), "basis": "computed from CV text"}
                if ft is not None
                else {"value": None, "basis": "unknown"}
            )

    identity = {
        "full_name": {"value": name, "evidence": _name_evidence(name, pages)},
        "emails": emails,
        "phones": phones,
        "links": links,
    }

    # 4. coverage + gating
    coverage = _coverage(name, emails, phones, experiences, skills, education)
    reasons = _gating_reasons(candidate, name, emails, phones, coverage, settings)
    if llm_error:
        reasons.append("LLM_UNAVAILABLE")
    if injection_hits:
        # untrusted-content safety: surface for a human (the offending text was
        # already stripped from everything the models saw).
        reasons.append("INJECTION_SUSPECTED")
    status = "NEEDS_REVIEW" if reasons else "OK"

    profile = {
        "candidate_id": cid,
        "display_no": idx,
        "source": {
            "file": candidate.get("document"),
            "page_range": candidate.get("page_range"),
            "container": candidate.get("container"),
        },
        "identity": identity,
        "headline": headline,
        "summary": summary,
        "total_years_experience": total_years,
        "experiences": experiences,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "languages": languages,
        "extraction": {
            "ocr_used": candidate.get("ocr_used", False),
            "vlm_used": candidate.get("vlm_used", False),
            "mean_ocr_confidence": candidate.get("mean_ocr_confidence"),
            "field_coverage": round(coverage, 2),
            "status": status,
            "reasons": reasons,
        },
        "duplicate_of": None,
        "_excluded_by_policy": sorted(PROTECTED_KEYS),
        # private (never surfaced) — source text for later evidence / QA:
        "_pages": [
            {
                "page": p["page"],
                "text": p.get("text", ""),
                "source": p.get("source"),
                "ocr_confidence": p.get("ocr_confidence"),
            }
            for p in pages
        ],
    }
    return _scrub_protected(profile)


def _build_list(items: Any, pages: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items or []:
        if isinstance(it, str):  # some models return plain strings
            nm = it.strip()
            if not nm:
                continue
            if kind == "certification":
                out.append({"name": nm, "evidence": verify_term(nm, pages)})
            elif kind == "language":
                out.append({"name": nm, "level": "", "evidence": verify_term(nm, pages)})
            continue
        if not isinstance(it, dict):
            continue
        ev = verify_quote(_pick(it, ["evidence_quote", "evidence", "quote"]), pages)
        if kind == "experience":
            title = str(_pick(it, ["title", "job_title", "role", "position"])).strip()
            org = str(_pick(it, ["organization", "company", "employer", "org"])).strip()
            if not title and not org:
                continue
            out.append(
                {
                    "title": title,
                    "organization": org,
                    "start": str(_pick(it, ["start", "start_date", "from", "start_year"])).strip(),
                    "end": str(_pick(it, ["end", "end_date", "to", "end_year"])).strip(),
                    "evidence": ev,
                }
            )
        elif kind == "education":
            degree = str(_pick(it, ["degree", "qualification"])).strip()
            field = str(_pick(it, ["field", "major", "field_of_study", "specialization"])).strip()
            inst = str(_pick(it, ["institution", "university", "school", "college"])).strip()
            if not any((degree, field, inst)):
                continue
            out.append(
                {
                    "degree": degree,
                    "field": field,
                    "institution": inst,
                    "year": str(
                        _pick(
                            it, ["year", "graduation_year", "grad_year", "completion_year", "date"]
                        )
                    ).strip(),
                    "evidence": ev,
                }
            )
        elif kind == "certification":
            nm = str(_pick(it, ["name", "title", "certification"])).strip()
            if nm:
                out.append({"name": nm, "evidence": ev})
        elif kind == "language":
            nm = str(_pick(it, ["name", "language"])).strip()
            if nm:
                out.append(
                    {
                        "name": nm,
                        "level": str(_pick(it, ["level", "proficiency", "fluency"])).strip(),
                        "evidence": ev,
                    }
                )
    return out


def _build_skills(items: Any, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A skill is 'stated' (with evidence) iff its name literally appears in the
    CV text — determined HERE, not from the model's self-label — else 'inferred'."""
    out: list[dict[str, Any]] = []
    seen = set()
    for it in items or []:
        if isinstance(it, str):
            nm = it.strip()
        elif isinstance(it, dict):
            nm = str(_pick(it, ["name", "skill", "skill_name"])).strip()
        else:
            continue
        if not nm or nm.lower() in seen:
            continue
        seen.add(nm.lower())
        ev = verify_term(nm, pages)
        out.append({"name": nm, "source": "stated" if ev else "inferred", "evidence": ev})
    return out


def _coverage(name, emails, phones, experiences, skills, education) -> float:
    cov = 0.0
    if name:
        cov += 0.20
    if emails or phones:
        cov += 0.20
    if experiences:
        cov += 0.25
    if len(skills) >= 3:
        cov += 0.20
    elif skills:
        cov += 0.10
    if education:
        cov += 0.15
    return cov


def _gating_reasons(candidate, name, emails, phones, coverage, settings) -> list[str]:
    reasons: list[str] = []
    if candidate.get("unreadable") or (
        candidate.get("page_count", 0) == 0 and candidate.get("text_chars", 0) == 0
    ):
        # A genuinely unreadable file is its OWN honest reason — never described
        # to the customer the same way as a readable-but-sparse CV.
        return ["UNREADABLE_FILE"]
    if not (emails or phones or name):
        reasons.append("NO_IDENTITY_ANCHOR")
    moc = candidate.get("mean_ocr_confidence")
    if candidate.get("ocr_used") and moc is not None and moc < settings.ocr_min_confidence:
        reasons.append("LOW_OCR_CONFIDENCE")
    if candidate.get("text_chars", 0) < 200:
        reasons.append("SPARSE_TEXT")
    if coverage < 0.40:
        reasons.append("SPARSE_EXTRACTION")
    if candidate.get("multi_cv"):
        reasons.append("MULTIPLE_CVS_IN_FILE")
    return reasons


def error_profile(candidate: dict[str, Any], idx: int) -> dict[str, Any]:
    """Minimal NEEDS_REVIEW profile for a candidate whose extraction crashed —
    the file stays visible and counted (review lane) instead of killing the
    whole batch."""
    return {
        "candidate_id": f"c_{idx:03d}",
        "display_no": idx,
        "source": {
            "file": candidate.get("document"),
            "page_range": candidate.get("page_range"),
            "container": candidate.get("container"),
        },
        "identity": {
            "full_name": {"value": "", "evidence": None},
            "emails": [],
            "phones": [],
            "links": [],
        },
        "headline": "",
        "total_years_experience": {"value": None, "basis": "unknown"},
        "experiences": [],
        "education": [],
        "skills": [],
        "certifications": [],
        "languages": [],
        "extraction": {
            "ocr_used": bool(candidate.get("ocr_used")),
            "vlm_used": bool(candidate.get("vlm_used")),
            "mean_ocr_confidence": candidate.get("mean_ocr_confidence"),
            "field_coverage": 0.0,
            "status": "NEEDS_REVIEW",
            "reasons": ["UNREADABLE_FILE"],
        },
        "duplicate_of": None,
        "_excluded_by_policy": sorted(PROTECTED_KEYS),
        "_pages": [],
    }


def _scrub_protected(profile: dict[str, Any]) -> dict[str, Any]:
    """Defensive: drop any protected keys that might have slipped into nested dicts."""

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: clean(v)
                for k, v in obj.items()
                if k == "_excluded_by_policy" or k.lower() not in PROTECTED_KEYS
            }
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    return clean(profile)
