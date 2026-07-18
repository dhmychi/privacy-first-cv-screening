"""Semantic similarity via the local bge-m3 model (1024-dim, multilingual) on
the existing Ollama. Used to match a query/JD/role to candidate skills even when
wording differs ('SAP Finance' ~ 'FICO'). Never used as the sole basis for a
claim — always paired with the literal-evidence skills."""

from __future__ import annotations

import math

import httpx


def embed(settings, texts: list[str], timeout: int = 60) -> list[list[float]]:
    """Batch-embed texts. Returns [] on failure so callers degrade to literal
    matching rather than crashing."""
    if not texts or settings is None:
        return []
    if getattr(settings, "llm_mode", "ollama") == "mock":
        from .. import mock

        return mock.embed(texts)
    url = settings.ollama_url.rstrip("/") + "/api/embed"
    try:
        r = httpx.post(url, json={"model": settings.embed_model, "input": texts}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("embeddings") or []
    except (httpx.HTTPError, ValueError):
        return []


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
