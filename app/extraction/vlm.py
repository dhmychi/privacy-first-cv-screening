"""Local VLM escalation for pages Tesseract cannot read reliably.

When a page is image-only or its OCR confidence is below the minimum, the page
image is transcribed by a LOCAL vision model (Ollama, on-box — same privacy
envelope as everything else). The VLM is used as a TRANSCRIBER only:

  * transcription-only prompt — it must never describe the person, photos,
    or appearance (protected-attribute hygiene), and never summarise;
  * output is plain page text carried with source='vlm' provenance, so
    downstream evidence citations can mark it 'verify at interview';
  * the escalation is capped per document and degrades silently to the OCR
    text when the model is unavailable, slow, or returns nothing useful.

Flag-gated by CV_VLM_ENABLED; model via CV_VLM_MODEL.
"""

from __future__ import annotations

import base64

import httpx

PROMPT = (
    "Transcribe ALL text on this CV/resume page exactly as written. "
    "If the layout has multiple columns, transcribe one column at a time, "
    "top to bottom (right column first if the page is Arabic). "
    "Preserve headings, bullet lines and dates as plain text lines. "
    "Output ONLY the transcribed text - no commentary, no markdown, no "
    "translation. Do NOT describe photos, logos, graphics, or the person's "
    "appearance; skip them entirely."
)


def available(settings) -> bool:
    return bool(settings.vlm_enabled and settings.vlm_model)


def transcribe_png(settings, png_bytes: bytes, timeout: int | None = None) -> str:
    """Return transcribed page text, or '' on any failure (caller keeps OCR)."""
    if getattr(settings, "llm_mode", "ollama") == "mock":
        return ""  # no external model calls in mock mode
    if not available(settings) or not png_bytes:
        return ""
    body = {
        "model": settings.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(png_bytes).decode("ascii")],
            }
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    try:
        r = httpx.post(
            settings.ollama_url.rstrip("/") + "/api/chat",
            json=body,
            timeout=timeout or settings.vlm_timeout,
        )
        r.raise_for_status()
        text = ((r.json().get("message") or {}).get("content") or "").strip()
    except (httpx.HTTPError, ValueError):
        return ""
    # A refusal / meta answer is short and text-free pages return nothing useful.
    if len(text) < 40:
        return ""
    return text
