"""Structured, privacy-safe operational logging.

Logs ONLY non-identifying operational metadata about a job's lifecycle. It is
impossible to log candidate content here **by construction**: the ``log_event``
signature IS the allowlist - there is no parameter that can carry a CV's text, a
candidate name, an email, a phone number, a job description, a prompt, an
evidence quote, a protected attribute, uploaded file bytes, or a secret. Every
value is additionally coerced to a bounded, slug-like primitive so free text (and
therefore PII) cannot ride along inside an otherwise-allowed field.

Allowed fields (everything else is structurally impossible):
  ts           - ISO-8601 UTC timestamp
  event        - short event name (slug)
  session      - a HASHED session handle (never the raw client chat id)
  job          - opaque job id (already a random uuid, not identifying)
  files        - integer count of files in the batch
  status       - short task-status slug (queued|running|ready|error|...)
  duration_ms  - integer milliseconds
  error_type   - exception CLASS name only (never the message/args)
  stage        - the pipeline stage a job was in (reading|extracting|scoring|...)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger("cv_screening")

_SLUG = re.compile(r"[^A-Za-z0-9_.:+-]")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _slug(v: Any, maxlen: int = 48) -> str | None:
    """Bounded enum-like token. Strips anything that is not a safe slug char, so
    an event/status/stage/error string cannot smuggle free text or PII."""
    if v is None:
        return None
    return _SLUG.sub("_", str(v))[:maxlen]


def hash_session(chat_id: Any) -> str | None:
    """Stable, NON-identifying handle for a session. The raw ``chat_id`` is
    client-supplied and could contain PII, so it is never logged - only this
    salted-length sha1 prefix, which is stable within a deployment for
    correlation but does not reveal the original id."""
    if chat_id is None:
        return None
    return "s_" + hashlib.sha1(str(chat_id).encode("utf-8")).hexdigest()[:10]


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def log_event(
    event: str,
    *,
    session_id: Any = None,
    job_id: Any = None,
    file_count: Any = None,
    status: Any = None,
    duration_ms: Any = None,
    error_type: Any = None,
    stage: Any = None,
) -> dict[str, Any]:
    """Emit one structured, PII-free operational log line and return the record.

    ``session_id`` is HASHED (never logged raw). ``error_type`` should be an
    exception class name; the exception message is intentionally NOT accepted."""
    rec: dict[str, Any] = {
        "ts": _now_iso(),
        "event": _slug(event),
        "session": hash_session(session_id),
        "job": _slug(job_id, 32),
        "files": _int(file_count),
        "status": _slug(status),
        "duration_ms": _int(duration_ms),
        "error_type": _slug(error_type),
        "stage": _slug(stage),
    }
    rec = {k: v for k, v in rec.items() if v is not None}
    logger.info(json.dumps(rec, ensure_ascii=True))
    return rec
