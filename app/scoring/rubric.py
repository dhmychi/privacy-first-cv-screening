"""JD -> scoring rubric.

The ONLY LLM step in the scoring plane: the local model PROPOSES a structured
rubric from the free-text job description (any role, any industry, any
language); deterministic post-validation then cleans, bounds and classifies it.
Nothing here is role- or dataset-specific: every requirement, keyword and hard
minimum comes from the JD text supplied at runtime.

A JD that yields too little structure is flagged NOT SCORABLE (clarity!='ok')
instead of being scored with fabricated precision.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..pipeline import llm
from ..pipeline.anchors import normalize_digits

# Degree-level keyword map (generic education vocabulary, incl. Arabic).
DEGREE_LEVELS: dict[str, list[str]] = {
    "bachelor": [
        "bachelor",
        "bachelors",
        "bsc",
        "b.sc",
        "ba ",
        "b.a",
        "beng",
        "b.eng",
        "btech",
        "b.tech",
        "undergraduate degree",
        "بكالوريوس",
    ],
    "master": [
        "master",
        "masters",
        "msc",
        "m.sc",
        "ma ",
        "m.a",
        "mba",
        "meng",
        "m.eng",
        "postgraduate degree",
        "ماجستير",
    ],
    "phd": ["phd", "ph.d", "doctorate", "doctoral", "دكتوراه"],
    "diploma": ["diploma", "associate degree", "دبلوم"],
}

# Requirement types the engine knows how to assess from a CV. "soft" traits
# (communication, teamwork...) are NOT assessable from a CV and are excluded
# from scoring rather than pretended.
RTYPES = {
    "skill",
    "experience_years",
    "education",
    "language",
    "certification",
    "responsibility",
    "soft",
}

RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role_title": {"type": "string"},
        "clarity": {"type": "string", "enum": ["ok", "vague"]},
        "clarity_reason": {"type": "string"},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["required", "preferred"]},
                    "rtype": {"type": "string", "enum": sorted(RTYPES)},
                    "min_years": {"type": ["number", "null"]},
                    "degree_level": {"type": "string"},
                    "equivalent_experience_ok": {"type": "boolean"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "kind", "rtype", "keywords"],
            },
        },
    },
    "required": ["role_title", "clarity", "requirements"],
}

SYSTEM = (
    "You convert a job description into a structured screening rubric. Work for ANY "
    "role, industry or language.\n"
    "RULES:\n"
    "- Extract EVERY distinct requirement stated in the JD; do not invent requirements "
    "that are not in the text.\n"
    "- kind: 'required' for must-haves, 'preferred' for nice-to-haves.\n"
    "- rtype: 'experience_years' when a minimum amount of experience is demanded "
    "(set min_years); 'education' for degrees (set degree_level: bachelor/master/phd/"
    "diploma, and equivalent_experience_ok=true when the JD says 'or equivalent "
    "experience'); 'certification' for licenses/certificates; 'language' for spoken/"
    "written languages; 'skill' for concrete tools/technologies/methods; "
    "'responsibility' for activities like 'designing X'/'managing Y'; 'soft' for "
    "personality traits (communication, teamwork, problem-solving).\n"
    "- keywords: the concrete terms from the JD that would satisfy this requirement, "
    "including each stated alternative as its own keyword (e.g. 'Java, Go, Python or "
    "C#' -> four keywords) plus 1-3 widely-used synonyms/abbreviations of those exact "
    "terms (e.g. Kubernetes -> K8s). Keywords must stay faithful to the JD - never "
    "add different technologies.\n"
    "- CVs are often written in English even when the JD is not: for a non-English "
    "JD, give each keyword BOTH in the JD's language AND as its standard English "
    "equivalent (e.g. 'إعداد الخطط الدراسية' -> also 'lesson planning', "
    "'curriculum planning').\n"
    "- clarity: 'vague' ONLY if the JD has no extractable requirements (e.g. a bare "
    "job title or marketing fluff); explain in clarity_reason.\n"
    "- role_title: the job title being hired for, from the JD."
)


class RubricError(RuntimeError):
    pass


def jd_hash(jd_text: str) -> str:
    return hashlib.sha256((jd_text or "").strip().encode("utf-8")).hexdigest()[:16]


def _clean_keywords(req_text: str, kws: Any) -> list[str]:
    out: list[str] = []
    seen = set()

    def add(k: str) -> None:
        k = re.sub(r"\s+", " ", str(k or "")).strip().strip(".,;:")
        if 1 < len(k) <= 60 and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)

    if isinstance(kws, list):
        for k in kws:
            add(k)
    # Acronym sub-keywords: a multiword keyword like 'Building ETL' would miss a
    # CV that just says 'ETL'. All-caps tokens are evidence-strong terms, and
    # word-boundary matching keeps them precise. Generic, not domain-specific.
    for k in list(out):
        if " " in k:
            for tok in re.findall(r"\b[A-Z][A-Z0-9+#]{1,7}\b", k):
                add(tok)
    if not out:
        # deterministic fallback: salient capitalised / technical tokens of the text
        for tok in re.findall(r"[A-Za-z0-9+#./-]{3,30}", req_text or ""):
            if not tok.islower():
                add(tok)
    return out[:14]


def _short_label(req: dict[str, Any]) -> str:
    """Deterministic <=4-word display label for table cells."""
    if req["rtype"] == "experience_years" and req.get("min_years"):
        yrs = req["min_years"]
        yrs = int(yrs) if float(yrs).is_integer() else yrs
        return f"{yrs}+ yrs experience"
    if req["rtype"] == "education" and req.get("degree_level"):
        return f"{req['degree_level'].capitalize()} degree"

    def _cut(s: str) -> str:
        """Truncate at a word boundary (never mid-word: 'maintenance pla')."""
        if len(s) <= 34:
            return s
        cut = s[:34]
        if " " in cut[20:]:
            cut = cut[: cut.rfind(" ")]
        return cut + "…"

    kws = req.get("keywords") or []
    if kws:
        lab = (
            "/".join(kws[:2])
            if len(kws) > 1 and len(kws[1]) >= 3 and len(kws[0]) + len(kws[1]) < 18
            else kws[0]
        )
        return _cut(lab)
    words = re.sub(r"\s+", " ", req.get("text", "")).split()
    return _cut(" ".join(words[:4]))


def _dedupe_requirements(reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge requirements that collapse to the SAME display label.

    The JD parser (LLM) is non-deterministic and occasionally OVER-SPLITS one
    requirement into several near-identical rows - most often a single
    "Bachelor's degree in Business, Marketing, or a related field" becoming two
    or three `education` entries that ALL render as the label 'Bachelor degree'.
    Two rows the customer reads as the SAME line must score as ONE requirement:
    otherwise (a) the denominator inflates and every candidate's score is
    diluted toward the same value, (b) the same label can appear as BOTH matched
    and missing when a reasoning judge gives the duplicates different verdicts,
    and (c) the gaps counter double-counts ("missing for 6 of 5 candidates").

    Generic and role-agnostic: the merge key is the deterministic display label
    within an rtype - never a hard-coded word, role or profession. Merging
    unions the keywords and keeps the strongest signal (required over preferred,
    first stated min_years / degree_level, richest text)."""
    merged: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for r in reqs:
        key = (r["rtype"], re.sub(r"\s+", " ", str(r.get("label") or "")).strip().lower())
        if not key[1]:  # never merge on an empty label
            key = (r["rtype"], f"__id_{len(order)}")
        if key not in merged:
            merged[key] = dict(r)
            merged[key]["keywords"] = list(r.get("keywords") or [])
            order.append(key)
            continue
        m = merged[key]
        if r.get("kind") == "required":  # strongest kind wins
            m["kind"] = "required"
        seen = {k.lower() for k in m["keywords"]}
        for k in r.get("keywords") or []:  # union keywords, order-preserving
            if k.lower() not in seen:
                seen.add(k.lower())
                m["keywords"].append(k)
        m["min_years"] = m.get("min_years") or r.get("min_years")
        m["degree_level"] = m.get("degree_level") or r.get("degree_level")
        m["equivalent_experience_ok"] = bool(
            m.get("equivalent_experience_ok") or r.get("equivalent_experience_ok")
        )
        if len(str(r.get("text") or "")) > len(str(m.get("text") or "")):
            m["text"] = r["text"]
    out: list[dict[str, Any]] = []
    for i, key in enumerate(order, 1):
        m = merged[key]
        m["id"] = f"r{i}"
        m["keywords"] = m["keywords"][:14]
        out.append(m)
    return out


def build(settings, jd_text: str) -> dict[str, Any]:
    """LLM proposes -> code validates. Raises RubricError only on infra failure;
    an unusable JD returns clarity!='ok' (a result, not an error)."""
    jd_text = (jd_text or "").strip()
    if len(jd_text) < 30:
        return {
            "role_title": jd_text[:60] or "Unknown role",
            "clarity": "vague",
            "clarity_reason": "job description too short to extract requirements",
            "requirements": [],
            "jd_hash": jd_hash(jd_text),
        }
    jd_digits = normalize_digits(jd_text)
    try:
        raw = llm.chat_json(
            settings,
            SYSTEM,
            jd_text,
            schema=RUBRIC_SCHEMA,
            timeout=240,
            num_ctx=8192,
            task="rubric",
        )
    except llm.LLMError as e:
        raise RubricError(f"rubric model unavailable: {e}") from e

    # Ollama does not reliably enforce the schema; the local model often uses
    # natural synonym keys ('rubric' for the list, 'requirement' for text,
    # 'synonyms' next to 'keywords'). Parse tolerantly - never lose a
    # requirement to a key alias.
    def _find_items(raw_obj):
        for k in ("requirements", "rubric", "criteria", "items", "requirements_list"):
            v = raw_obj.get(k)
            if isinstance(v, list) and v:
                return v
        return None

    items = _find_items(raw)
    if items is None:
        # The model sometimes ignores the JSON format entirely (observed: a
        # markdown-table rubric for a non-English JD). ONE reinforced retry
        # with an explicit JSON-only instruction; still temp 0.
        try:
            raw = llm.chat_json(
                settings,
                SYSTEM + "\nCRITICAL: respond with ONLY one valid JSON object "
                "matching the schema. No prose, no markdown, no tables.",
                jd_text,
                schema=RUBRIC_SCHEMA,
                timeout=240,
                num_ctx=8192,
                task="rubric",
            )
            items = _find_items(raw)
        except llm.LLMError:
            items = None
    reqs: list[dict[str, Any]] = []
    for r in (items or [])[:40]:
        if not isinstance(r, dict):
            continue
        low = {str(k).lower(): v for k, v in r.items()}

        def pick(*keys, default="", low=low):  # bind this row's `low` explicitly
            for k in keys:
                v = low.get(k)
                if v not in (None, "", []):
                    return v
            return default

        rtype = str(pick("rtype", "type", "category") or "skill").lower()
        if rtype not in RTYPES:
            rtype = "skill"
        # An experience requirement WITHOUT a stated years floor is a DOMAIN
        # experience requirement ("litigation experience"), not a tenure gate -
        # reclassify so it is matched by its keywords/evidence, never blanket-
        # satisfied by any tenure. (Root-caused on real-world JDs.)
        _my_probe = pick("min_years", "minimum_years", "years", default=None)
        try:
            _my_probe = float(_my_probe) if _my_probe is not None else None
        except (TypeError, ValueError):
            _my_probe = None
        _req_text_probe = normalize_digits(
            str(pick("text", "requirement", "description", "name", "title"))
        )

        # A tenure floor is REAL only if its number appears in the ORIGINAL JD
        # (digit-normalized) - the model may echo terse text without the number
        # ('accounting experience', min=5) or FABRICATE a floor the JD never
        # stated ('ground handling knowledge', min=1). Bilingual, deterministic,
        # anchored to the JD text itself.
        def _floor_real(val) -> bool:
            if not val:
                return False
            num = int(val) if float(val).is_integer() else val
            return bool(re.search(rf"(?<!\d){num}(?!\d)", jd_digits))

        _exp_word = re.search(r"experience|خبرة", _req_text_probe, re.I)
        if rtype == "experience_years":
            if not _floor_real(_my_probe):
                rtype = "responsibility"  # domain experience, not a tenure gate
                _my_probe = None
        elif _exp_word:
            # Inverse coercion: the requirement's own text states a years floor
            # ("8+ years...", "خبرة لا تقل عن 3 سنوات") or the model put a
            # JD-corroborated floor in min_years but misclassified the type.
            _yrs_m = re.search(
                r"(\d{1,2})\s*\+?\s*(?:years?|yrs?|سنوات|سنة|عام|أعوام)", _req_text_probe, re.I
            )
            if _yrs_m:
                rtype = "experience_years"
                _my_probe = float(_yrs_m.group(1))
            elif _floor_real(_my_probe):
                rtype = "experience_years"
        kws_raw: list[Any] = []
        for k in ("keywords", "synonyms", "alternatives", "terms", "aliases"):
            v = low.get(k)
            if isinstance(v, list):
                kws_raw.extend(v)
            elif isinstance(v, str) and v:
                kws_raw.append(v)
        text = re.sub(
            r"\s+", " ", str(pick("text", "requirement", "description", "name", "title"))
        ).strip()
        if not text:
            # synthesize deterministically from what the model did return
            if kws_raw:
                text = ", ".join(str(k) for k in kws_raw[:4])
            elif rtype == "experience_years" and pick("min_years", "years", default=None):
                text = f"{pick('min_years', 'years')}+ years experience"
            elif rtype == "education" and pick("degree_level", "degree", default=None):
                text = f"{pick('degree_level', 'degree')} degree"
        if len(text) < 3:
            continue
        kind_s = str(pick("kind", "priority", "importance") or "required").lower()
        kind = (
            "preferred" if kind_s.startswith(("pref", "nice", "optional", "bonus")) else "required"
        )
        my = _my_probe  # validated above (incl. bilingual text-floor coercion)
        if my is not None and not (0 < my <= 40):
            my = None
        dl = str(pick("degree_level", "degree") or "").lower().strip()
        if dl not in DEGREE_LEVELS:
            dl = ""
        req = {
            "id": f"r{len(reqs) + 1}",
            "text": text[:220],
            "kind": kind,
            "rtype": rtype,
            "min_years": my,
            "degree_level": dl,
            "equivalent_experience_ok": bool(
                pick("equivalent_experience_ok", "equivalent_ok", default=False)
            )
            or "equivalent" in text.lower(),
            "keywords": _clean_keywords(text, kws_raw),
        }
        req["label"] = _short_label(req)
        reqs.append(req)

    # Collapse LLM over-splits: two rows the customer reads as the SAME label
    # must score as ONE requirement (prevents denominator inflation, met-and-
    # missing contradictions under the reasoning judge, and gap double-counts).
    reqs = _dedupe_requirements(reqs)

    clarity = str(raw.get("clarity") or "ok")
    if clarity not in ("ok", "vague"):
        clarity = "ok" if reqs else "vague"
    reason = str(raw.get("clarity_reason") or raw.get("reason") or "")
    assessable = [r for r in reqs if r["rtype"] != "soft"]
    if len(assessable) < 2:
        clarity = "vague"
        reason = reason or (
            "the job description does not state enough concrete, "
            "CV-assessable requirements to score candidates reliably"
        )
    return {
        "role_title": re.sub(r"\s+", " ", str(raw.get("role_title") or "")).strip()[:80]
        or "Unknown role",
        "clarity": clarity,
        "clarity_reason": reason,
        "requirements": reqs,
        "jd_hash": jd_hash(jd_text),
    }
