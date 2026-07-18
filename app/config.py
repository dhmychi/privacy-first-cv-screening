"""Runtime configuration, sourced entirely from environment variables so the
package is portable (no machine-specific values baked in). Read at call time so
the service and tests always see current env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import __version__


@dataclass(frozen=True)
class Settings:
    # --- auth / service ---
    api_key: str

    # --- OCR (Tesseract-only in MVP; vision hook present but OFF) ---
    ocr_mode: str  # off | auto | force
    ocr_min_confidence: float  # below this a page is flagged LOW_OCR_CONFIDENCE
    ocr_high_confidence: float  # at/above this OCR is treated as reliable
    ocr_dpi: int
    ocr_langs: str
    ocr_text_min: int  # min chars for a page's text layer to be "sufficient"
    vision_ocr_url: str  # MUST stay local (or blank); off-box refused at startup
    allow_cloud_ocr: bool

    # --- ingest caps ---
    max_files: int
    max_pages_per_doc: int
    max_upload_mb: int
    max_zip_entries: int

    # --- VLM escalation (LOCAL vision model for unreadable pages) ---
    vlm_enabled: bool
    vlm_model: str  # e.g. qwen2.5vl:7b (must be on the local Ollama)
    vlm_timeout: int  # seconds per page transcription
    vlm_max_pages_per_doc: int  # latency cap per document

    # --- session (ephemeral, per chat_id) ---
    session_ttl_minutes: int
    extract_workers: int  # concurrent LLM extraction calls (semaphore)
    acquire_workers: int  # concurrent CPU acquire (OCR/layout) workers
    data_dir: str  # session persistence dir ('' disables)

    # --- local model backends (Ollama, on-box) ---
    ollama_url: str
    llm_model: str
    embed_model: str
    llm_mode: str  # 'mock' (default, zero-dependency) | 'ollama'

    # --- scoring matcher: 'legacy' (keyword+cosine) | 'judge' (grounded LLM reasoning) ---
    matcher: str
    score_timeout: int

    version: str


def get_settings() -> Settings:
    g = os.environ.get
    return Settings(
        api_key=g("CV_API_KEY", ""),
        ocr_mode=g("CV_OCR_MODE", "auto"),
        ocr_min_confidence=float(g("CV_OCR_MIN_CONFIDENCE", "60")),
        ocr_high_confidence=float(g("CV_OCR_HIGH_CONFIDENCE", "80")),
        ocr_dpi=int(g("CV_OCR_DPI", "300")),
        ocr_langs=g("CV_OCR_LANGS", "ara+eng"),
        ocr_text_min=int(g("CV_OCR_TEXT_MIN", "20")),
        vision_ocr_url=g("CV_VISION_OCR_URL", ""),
        allow_cloud_ocr=g("CV_ALLOW_CLOUD_OCR", "false").lower() == "true",
        max_files=int(g("CV_MAX_FILES", "300")),
        max_pages_per_doc=int(g("CV_MAX_PAGES_PER_DOC", "30")),
        max_upload_mb=int(g("CV_MAX_UPLOAD_MB", "50")),
        max_zip_entries=int(g("CV_MAX_ZIP_ENTRIES", "500")),
        vlm_enabled=g("CV_VLM_ENABLED", "true").lower() == "true",
        vlm_model=g("CV_VLM_MODEL", "qwen2.5vl:7b"),
        vlm_timeout=int(g("CV_VLM_TIMEOUT", "120")),
        vlm_max_pages_per_doc=int(g("CV_VLM_MAX_PAGES_PER_DOC", "6")),
        session_ttl_minutes=int(g("CV_SESSION_TTL_MINUTES", "120")),
        extract_workers=int(g("CV_EXTRACT_WORKERS", "4")),
        acquire_workers=int(g("CV_ACQUIRE_WORKERS", "4")),
        data_dir=g("CV_DATA_DIR", "/app/data"),
        ollama_url=g("CV_OLLAMA_URL", "http://host.docker.internal:11434"),
        llm_model=g("CV_LLM_MODEL", "qwen3.6:35b"),
        embed_model=g("CV_EMBED_MODEL", "bge-m3:latest"),
        # 'mock' = deterministic, no-model backend (the PUBLIC DEFAULT, so the
        # service runs with zero dependencies: no Ollama, no model, no network).
        # 'ollama' = call the local models configured above.
        llm_mode=g("CV_LLM_MODE", "mock").strip().lower(),
        # DEFAULT = judge (grounded LLM reasoning). Instant rollback with NO
        # rebuild: set env CV_MATCHER=legacy to fall back to the keyword+cosine
        # matcher, which stays fully intact.
        matcher=g("CV_MATCHER", "judge").strip().lower(),
        score_timeout=int(g("CV_SCORE_TIMEOUT", "240")),
        version=__version__,
    )


def vision_is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "host.docker.internal", "::1"))
