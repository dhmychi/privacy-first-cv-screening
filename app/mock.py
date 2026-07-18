"""Deterministic mock backend for ``CV_LLM_MODE=mock`` (the public default).

Lets the WHOLE pipeline run with zero models: no Ollama, no local model, no
external API, no paid key, no internet. Every model-dependent step is replaced by
a deterministic, rule-based stand-in so ``docker compose up`` works out of the box
for anyone who clones the repo.

Design goals (match the real path, honestly):
  * DETERMINISTIC & reproducible - same input yields byte-identical output.
  * LOGICAL, not canned - the output is parsed from the ACTUAL input text, so
    different CVs / JDs yield different, testable results (never one fixed reply).
  * GROUNDED like the real path - evidence quotes are real substrings of the CV,
    so the code's quote-verification still runs and still drops anything it
    cannot locate. The mock proposes; the deterministic code still verifies.
  * CLEARLY MOCK - callers surface a "mock mode" banner and ``/health`` reports
    the active mode; nothing here is presented as a real model inference.

This is a demonstration backend, not a model. It understands the conventional CV
and JD layout used by the synthetic data (a name line, a title line, EXPERIENCE /
EDUCATION / SKILLS sections, comma-separated skill lists, ``YYYY - YYYY`` date
ranges, ``N years``, degree words). On free-form input it degrades gracefully -
it returns only what it can parse - exactly like the real extractor on a sparse
CV. For the grounded-reasoning LLM judge and true semantic embeddings, run with
``CV_LLM_MODE=ollama`` against a local model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

MOCK_MARKER = "mock"

_PAGE_RE = re.compile(r"^\[Page\s+\d+\]\s*$", re.I)
_DATE_RANGE = re.compile(
    r"\(?\b((?:19|20)\d{2})\b\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now|till date)\)?",
    re.I,
)
_YEARS = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years|yrs|year)\b", re.I)
# degree vocabulary (English + Arabic) so degrees are recognised in either script
_DEGREE_WORDS = {
    "bachelor": [
        "bachelor",
        "bsc",
        "b.sc",
        "b.a",
        "ba ",
        "beng",
        "btech",
        "undergraduate",
        "بكالوريوس",
    ],
    "master": ["master", "msc", "m.sc", "m.a", "mba", "meng", "postgraduate", "ماجستير"],
    "phd": ["phd", "ph.d", "doctorate", "doctoral", "دكتوراه"],
    "diploma": ["diploma", "associate degree", "دبلوم"],
}
# section headers (English + Arabic) so bilingual CVs segment correctly
_SECTION_HEADERS = {
    "experience": {
        "experience",
        "work experience",
        "employment",
        "work history",
        "professional experience",
        "employment history",
        "الخبرة",
        "الخبرات",
        "الخبرة العملية",
        "الخبرة المهنية",
        "الخبرات المهنية",
    },
    "education": {
        "education",
        "academic background",
        "academics",
        "qualifications",
        "التعليم",
        "المؤهلات",
        "المؤهلات العلمية",
        "المؤهل العلمي",
        "التعليم والمؤهلات",
    },
    "skills": {
        "skills",
        "technical skills",
        "key skills",
        "core skills",
        "skill set",
        "المهارات",
        "المهارات التقنية",
        "المهارات الأساسية",
        "المهارات الرئيسية",
    },
    "summary": {
        "summary",
        "professional summary",
        "profile",
        "about",
        "objective",
        "الملخص",
        "نبذة",
        "الملخص المهني",
        "نبذة مختصرة",
        "الهدف الوظيفي",
    },
    "certifications": {
        "certifications",
        "certificates",
        "certs",
        "licenses",
        "الشهادات",
        "الدورات",
        "الشهادات والدورات",
    },
    "languages": {"languages", "language", "اللغات", "اللغة"},
}
# comma set used everywhere a list is split (Latin + Arabic comma U+060C)
_COMMASPLIT = r"[,،;/|•·]"
# Tokens that must never be treated as a skill / requirement keyword.
_STOP = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "years",
    "year",
    "yrs",
    "experience",
    "skills",
    "skill",
    "must",
    "have",
    "required",
    "requirements",
    "nice",
    "preferred",
    "plus",
    "strong",
    "good",
    "knowledge",
    "proficient",
    "proficiency",
    "ability",
    "including",
    "such",
    "as",
    "etc",
    "e.g",
    "e.g.",
    "is",
    "are",
    "be",
    "will",
    "we",
    "you",
    "our",
    "role",
    "job",
    "position",
    "candidate",
    "team",
    "work",
    "using",
    "use",
}


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def _lines(text: str) -> list[str]:
    out: list[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s and not _PAGE_RE.match(s):
            out.append(s)
    return out


def _section_of(line: str) -> str | None:
    """Return the section key if the line is a section header, else None. Handles
    bilingual headers that pair two labels, e.g. 'Experience / الخبرة'."""
    raw = line.strip()
    if len(raw) > 40:
        return None
    for part in re.split(r"[/|]", raw):
        low = re.sub(r"[:\-\s]+$", "", part.strip().lower())
        if not low:
            continue
        for key, names in _SECTION_HEADERS.items():
            if low in names:
                return key
    return None


def _tokens(text: str) -> list[str]:
    return list(re.findall(r"[A-Za-z][A-Za-z0-9+#./]{1,29}", text or ""))


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _split_sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Return (header_lines_before_first_section, {section: [lines]})."""
    head: list[str] = []
    sections: dict[str, list[str]] = {}
    cur: str | None = None
    for ln in lines:
        sec = _section_of(ln)
        if sec is not None:
            cur = sec
            sections.setdefault(cur, [])
            continue
        if cur is None:
            head.append(ln)
        else:
            sections[cur].append(ln)
    return head, sections


def _parse_experience(sec_lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in sec_lines:
        m = _DATE_RANGE.search(ln)
        if not m:
            continue
        before = ln[: m.start()].strip(" ,،-–—(")
        title, org = before, ""
        halves = re.split(r"[,،]", before, maxsplit=1)
        if len(halves) == 2:
            title, org = halves[0].strip(), halves[1].strip()
        out.append(
            {
                "title": title[:120] or "Role",
                "organization": org[:120],
                "start": m.group(1),
                "end": m.group(2),
                "evidence_quote": ln[:200],
            }
        )
    return out


def _parse_education(sec_lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in sec_lines:
        low = ln.lower()
        if not any(w in low for grp in _DEGREE_WORDS.values() for w in grp):
            continue
        parts = [p.strip() for p in re.split(r"[,،]", ln)]
        degree = parts[0] if parts else ln
        institution = parts[1] if len(parts) > 1 else ""
        ym = re.search(r"\b(19|20)\d{2}\b", ln)
        out.append(
            {
                "degree": degree[:100],
                "field": "",
                "institution": institution[:100],
                "year": ym.group(0) if ym else "",
                "evidence_quote": ln[:200],
            }
        )
    return out


def _parse_skills(sec_lines: list[str], head: list[str]) -> list[dict[str, Any]]:
    raw: list[tuple[str, str]] = []  # (skill, evidence_line)
    # skills section: each line is a comma-separated list
    for ln in sec_lines:
        body = re.sub(r"(?i)^(technical\s+|key\s+|core\s+)?skills?\s*[:\-]\s*", "", ln)
        for part in re.split(_COMMASPLIT, body):
            s = part.strip()
            if 1 < len(s) <= 40:
                raw.append((s, ln))
    # inline "Skills: a, b, c" anywhere in the header block
    for ln in head:
        m = re.match(r"(?i)^\s*skills?\s*[:\-]\s*(.+)$", ln)
        if m:
            for part in re.split(_COMMASPLIT, m.group(1)):
                s = part.strip()
                if 1 < len(s) <= 40:
                    raw.append((s, ln))
    out: list[dict[str, Any]] = []
    seen = set()
    for skill, ev in raw:
        key = skill.lower()
        if key in seen or key in _STOP:
            continue
        seen.add(key)
        out.append({"name": skill, "source": "stated", "evidence_quote": ev[:200]})
    return out


def _extract(user_text: str) -> dict[str, Any]:
    lines = _lines(user_text)
    head, sections = _split_sections(lines)

    full_name = head[0] if head else ""
    headline = head[1] if len(head) > 1 else ""
    # a header line that is clearly contact info is not a headline
    if headline and ("@" in headline or any(c.isdigit() for c in headline)):
        headline = ""

    summary = ""
    if sections.get("summary"):
        summary = sections["summary"][0][:400]

    return {
        "full_name": full_name[:80],
        "headline": headline[:120],
        "summary": summary,
        "total_years_experience": None,  # code computes years from dated roles
        "experiences": _parse_experience(sections.get("experience", [])),
        "education": _parse_education(sections.get("education", [])),
        "skills": _parse_skills(sections.get("skills", []), head),
        "certifications": [
            {"name": c[:80], "evidence_quote": c[:200]} for c in sections.get("certifications", [])
        ],
        "languages": [
            {"name": lang[:60], "level": "", "evidence_quote": lang[:200]}
            for lang in sections.get("languages", [])
        ],
    }


# --------------------------------------------------------------------------
# JD parsing (shared by the rubric and the query-side JD parser)
# --------------------------------------------------------------------------


def _parse_jd(jd_text: str) -> dict[str, Any]:
    lines = _lines(jd_text)
    role = lines[0][:120] if lines else ""
    for ln in lines:
        m = re.match(r"(?i)^\s*(?:role|position|title|job\s*title)\s*[:\-]\s*(.+)$", ln)
        if m:
            role = m.group(1).strip()[:120]
            break

    ym = _YEARS.search(jd_text)
    min_years = int(ym.group(1)) if ym else None

    degree_level = ""
    low_all = jd_text.lower()
    for level, words in _DEGREE_WORDS.items():
        if any(w in low_all for w in words):
            degree_level = level
            break

    must: list[str] = []
    nice: list[str] = []

    def _collect(target: list[str], text: str) -> None:
        for part in re.split(r"[,،;/|•·\n]|\band\b|\bor\b", text):
            s = part.strip(" .:-")
            if (
                1 < len(s) <= 40
                and s.lower() not in _STOP
                and s.lower() not in [x.lower() for x in target]
            ):
                target.append(s)

    must_labels = (
        "required",
        "must",
        "requirement",
        "skill",
        "technolog",
        "tech stack",
        "proficient",
        "experience with",
        "knowledge of",
        "responsibilit",
        "qualification",
        "competenc",
    )
    nice_labels = (
        "nice",
        "preferred",
        "prefer",
        "bonus",
        "plus ",
        "desirable",
        "advantage",
        "good to have",
    )
    for ln in lines:
        # classify by the label BEFORE a colon ("Required skills:", "Nice-to-have:")
        mlabel = re.match(r"^\s*([^:]{1,50}):\s*(.+)$", ln)
        if mlabel:
            label = mlabel.group(1).lower()
            rest = mlabel.group(2)
            if any(w in label for w in nice_labels):
                _collect(nice, rest)
                continue
            if any(w in label for w in must_labels):
                _collect(must, rest)
                continue
        if ln.lstrip().startswith(("-", "•", "*", "·")):
            _collect(must, ln.lstrip("-•*· ").strip())

    return {
        "role_title": role,
        "must_have_skills": must[:20],
        "nice_to_have_skills": nice[:20],
        "min_years_experience": min_years,
        "_degree_level": degree_level,
    }


def _jd(jd_text: str) -> dict[str, Any]:
    p = _parse_jd(jd_text)
    return {
        "role_title": p["role_title"],
        "must_have_skills": p["must_have_skills"],
        "nice_to_have_skills": p["nice_to_have_skills"],
        "min_years_experience": p["min_years_experience"],
    }


def _rubric(jd_text: str) -> dict[str, Any]:
    p = _parse_jd(jd_text)
    reqs: list[dict[str, Any]] = []
    for s in p["must_have_skills"]:
        reqs.append({"text": s, "kind": "required", "rtype": "skill", "keywords": [s]})
    for s in p["nice_to_have_skills"]:
        reqs.append({"text": s, "kind": "preferred", "rtype": "skill", "keywords": [s]})
    if p["min_years_experience"]:
        reqs.append(
            {
                "text": f"{p['min_years_experience']}+ years of relevant experience",
                "kind": "required",
                "rtype": "experience_years",
                "min_years": p["min_years_experience"],
                "keywords": [],
            }
        )
    if p["_degree_level"]:
        reqs.append(
            {
                "text": f"{p['_degree_level'].title()} degree",
                "kind": "required",
                "rtype": "education",
                "degree_level": p["_degree_level"],
                "keywords": [p["_degree_level"]],
            }
        )
    clarity = "ok" if reqs else "vague"
    return {
        "role_title": p["role_title"] or (jd_text[:60] if jd_text else "Unknown role"),
        "clarity": clarity,
        "clarity_reason": "" if reqs else "no extractable requirements (mock parser)",
        "requirements": reqs,
    }


# --------------------------------------------------------------------------
# judge (deterministic requirement reasoning stand-in)
# --------------------------------------------------------------------------


def _judge(user_text: str) -> dict[str, Any]:
    split = re.split(r"REQUIREMENTS to judge:", user_text, maxsplit=1)
    evidence = split[0]
    reqs_block = split[1] if len(split) > 1 else ""
    low_ev = evidence.lower()

    computed_years: float | None = None
    my = re.search(r"COMPUTED TOTAL EXPERIENCE:\s*([\d.]+)\s*years", evidence, re.I)
    if my:
        try:
            computed_years = float(my.group(1))
        except ValueError:
            computed_years = None

    verdicts: list[dict[str, Any]] = []
    for ln in reqs_block.splitlines():
        # judge.py formats each line as: [id] KIND, RTYPE: "text" [extras]
        m = re.match(r'^\s*\[([^\]]+)\]\s*([a-z_]+)\s*,\s*([a-z_]+)\s*:\s*"(.*?)"(.*)$', ln, re.I)
        if not m:
            continue
        req_id, rtype, text, tail = m.group(1), m.group(3).lower(), m.group(4), m.group(5)
        verdict, quote, reason = "missing", "", "mock: no supporting evidence found"

        if rtype == "experience_years":
            ym = re.search(r"min\s*([\d.]+)\s*yr", tail, re.I)
            need = float(ym.group(1)) if ym else 0.0
            if computed_years is None:
                verdict, reason = "unverified", "mock: total experience not computable"
            elif computed_years >= need:
                verdict, reason = "met", f"mock: {computed_years:g}y >= {need:g}y required"
            elif computed_years >= 0.7 * need:
                verdict, reason = "partial", f"mock: {computed_years:g}y vs {need:g}y required"
            else:
                verdict, reason = "missing", f"mock: {computed_years:g}y below {need:g}y required"
        elif rtype == "education":
            m2 = re.search(r"degree level (\w+)", tail, re.I)
            level = m2.group(1).lower() if m2 else ""
            words = list(_DEGREE_WORDS.get(level, [])) + [
                "bachelor",
                "master",
                "phd",
                "doctorate",
                "diploma",
                "degree",
                "bsc",
                "msc",
                "mba",
                "ba",
                "ma",
                "beng",
                "meng",
                "btech",
            ]
            hit = next(
                (
                    w
                    for w in words
                    if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low_ev)
                ),
                None,
            )
            if hit:
                verdict, quote, reason = "met", hit, f"mock: degree evidence '{hit}'"
            else:
                verdict, reason = "unverified", "mock: no degree evidence in CV"
        else:
            # keyword presence: is a salient word from the requirement in the CV?
            hit = None
            for tok in _tokens(text):
                if tok.lower() in _STOP or len(tok) < 3:
                    continue
                if re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])", low_ev, re.I
                ):
                    hit = tok
                    break
            if hit:
                verdict, quote, reason = "met", hit, f"mock: '{hit}' present in CV"
            elif rtype in ("skill", "responsibility", "certification", "language"):
                verdict, reason = "missing", "mock: not present in CV"
            else:
                verdict, reason = "unverified", "mock: cannot assess from CV"

        verdicts.append(
            {
                "req_id": req_id,
                "verdict": verdict,
                "evidence_quote": quote,
                "reason": reason,
                "confidence": "medium",
            }
        )
    return {"verdicts": verdicts}


# --------------------------------------------------------------------------
# public dispatch (used by llm.chat_json / embeddings.embed / llm.ping)
# --------------------------------------------------------------------------


def chat_json(
    task: str | None, system: str, user: str, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deterministic stand-in for a JSON chat call. ``task`` selects the parser;
    when absent it is inferred from the schema/system so the mock still works if a
    new call site forgets to pass it."""
    t = (task or "").lower()
    if not t:
        props = (schema or {}).get("properties", {})
        if "verdicts" in props:
            t = "judge"
        elif "requirements" in props and "clarity" in props:
            t = "rubric"
        elif "full_name" in props or "experiences" in props:
            t = "extract"
        elif (
            "hiring requirements" in (system or "").lower()
            or "job description" in (system or "").lower()
        ):
            t = "jd"
    if t == "extract":
        return _extract(user)
    if t == "judge":
        return _judge(user)
    if t == "rubric":
        return _rubric(user)
    if t == "jd":
        return _jd(user)
    return {}


def embed(texts: list[str], dim: int = 128) -> list[list[float]]:
    """Deterministic bag-of-tokens hashing embedding (L2-normalised). Shares
    vocabulary buckets so texts with overlapping words score higher cosine -
    enough for meaningful, reproducible semantic ranking without a real model.
    Uses a stable hash (sha1) so results never depend on PYTHONHASHSEED."""
    out: list[list[float]] = []
    for t in texts or []:
        v = [0.0] * dim
        for tok in _tokens((t or "").lower()):
            v[_stable_hash(tok) % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        out.append([x / norm for x in v] if norm else v)
    return out
