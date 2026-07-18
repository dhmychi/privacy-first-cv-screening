"""Serialize all access to pypdfium2 / PDFium.

PDFium — the C library behind pypdfium2 — is **not thread-safe**: it keeps
process-global state, so calling into it from more than one thread at a time
can abort or segfault the interpreter. The batch pipeline inspects and renders
PDFs from a thread pool, so every pypdfium2 operation must be guarded by this
single, process-wide lock.

It is an ``RLock`` so a helper that already holds it can call another guarded
helper without deadlocking.
"""

from __future__ import annotations

import threading

PDFIUM_LOCK = threading.RLock()
