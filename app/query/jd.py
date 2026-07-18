"""Parse a job description into structured hiring criteria. Used when HR pastes a
real JD; for a short role phrase ('best candidate for HR Specialist') the engine
builds criteria directly without an LLM call."""

from __future__ import annotations

from typing import Any

from ..pipeline import llm
from ..pipeline.extract import _pick  # alias-tolerant getter

JD_SYSTEM = (
    "Extract hiring requirements from the job description. Return ONLY JSON with "
    "keys: role_title (string), must_have_skills (array of short skill strings), "
    "nice_to_have_skills (array of strings), min_years_experience (number or null). "
    "Job-relevant only. NEVER include gender, age, nationality, religion, marital "
    "status, or other protected attributes. Keep each skill 1-4 words. Do not "
    "invent requirements that are not in the text."
)


def _as_str_list(v: Any) -> list[str]:
    out: list[str] = []
    if isinstance(v, list):
        for x in v:
            s = ((x.get("name") or "") if isinstance(x, dict) else str(x)).strip() if x else ""
            if s and s.lower() not in [o.lower() for o in out]:
                out.append(s)
    return out


def parse_jd(text: str, settings) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        # plain JSON (Ollama does not reliably enforce a passed schema; the prompt
        # describes the keys and we parse alias/nesting tolerantly)
        out = llm.chat_json(settings, JD_SYSTEM, text, task="jd")
    except llm.LLMError:
        out = {}
    # flatten a "requirements"/"required" wrapper if the model nested under it
    for wrap in ("requirements", "required", "criteria"):
        if isinstance(out.get(wrap), dict):
            merged = dict(out)
            merged.update(out[wrap])
            out = merged
    must = _as_str_list(
        _pick(
            out,
            ["must_have_skills", "must_have", "required_skills", "mandatory_skills", "skills"],
            default=[],
        )
    )
    nice = _as_str_list(
        _pick(out, ["nice_to_have_skills", "nice_to_have", "preferred_skills"], default=[])
    )
    miny = _pick(
        out,
        [
            "min_years_experience",
            "minimum_experience_years",
            "min_years",
            "minimum_years",
            "years_required",
            "experience_years",
        ],
        default=None,
    )
    role = str(
        _pick(out, ["role_title", "job_title", "title", "role", "position"], default="")
    ).strip()
    return {
        "role_text": role or text[:120],
        "must_have": must,
        "nice_to_have": nice,
        "min_years": int(miny) if isinstance(miny, int | float) else None,
        "source": "jd",
    }
