"""Lightweight pydantic models for request validation / OpenAPI docs.
Analyze/query responses are plain dicts (the canonical candidate-profile schema
lives in the pipeline)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    llm_mode: str
    ocr: dict[str, Any]
    models: dict[str, Any]
    sessions: dict[str, Any]
    time: str


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]


class QueryRequest(BaseModel):
    question: str
    options: dict[str, Any] | None = None


class ScoreRequest(BaseModel):
    jd_text: str
    lang: str | None = "en"  # report language: en | ar
    top_n: int | None = None  # table rows shown (None = policy default)


def parse_options(raw: str | None) -> dict[str, Any]:
    import json

    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}
