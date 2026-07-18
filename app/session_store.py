"""Ephemeral per-chat session store — the heart of multi-turn conversational
analysis. Keyed by a client-supplied ``chat_id``: the batch is analyzed ONCE and the
extracted candidate profiles + the evolving view-state live here for the life of
the conversation (sliding TTL), then auto-expire. No long-term PII persistence:
the TTL policy is unchanged — expiry deletes the on-disk copy too.

Thread-safe (the FastAPI app may touch a session from a request thread and from a
background analysis thread). READY sessions are additionally persisted to a local
volume (atomic JSON per chat) so a container restart does not lose an analyzed
batch mid-conversation; persistence stays inside the same on-box privacy
envelope and honors the TTL on load.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _now() -> float:
    return time.time()


@dataclass
class Session:
    chat_id: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "pending"  # pending | queued | running | ready | error
    progress: dict[str, Any] = field(default_factory=lambda: {"done": 0, "total": 0, "stage": ""})
    error: str | None = None
    client_signature: str = ""  # signature of the uploaded batch (set by the caller)

    # Inputs / intermediate
    documents: list[dict[str, Any]] = field(default_factory=list)  # per-file acquisition result

    # System of record (set when status -> ready)
    profiles: list[dict[str, Any]] = field(default_factory=list)  # CandidateProfile dicts
    roster_order: list[str] = field(default_factory=list)  # stable candidate_id order
    summary: dict[str, Any] = field(default_factory=dict)
    last_added: int = 0  # candidates added by the most recent append
    processed_hashes: set = field(default_factory=set)  # content hashes of ingested upload files

    # Conversation memory (Section 14 of the plan)
    view_state: dict[str, Any] = field(
        default_factory=lambda: {
            "current_set": [],  # candidate_ids after the latest filter/exclude
            "last_ranking": [],  # [{candidate_id, score}, ...]
            "last_shortlist": [],  # candidate_ids
            "last_result": [],  # referents of the most recent answer ("them"/"those")
            "history": [],  # ordered ops, enables undo / "go back"
        }
    )

    created: float = field(default_factory=_now)
    last_access: float = field(default_factory=_now)

    def touch(self) -> None:
        self.last_access = _now()

    def public_status(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "client_signature": self.client_signature,
            "candidate_count": len(self.profiles),
            "last_added": self.last_added,
            "needs_review": sum(
                1
                for p in self.profiles
                if (p.get("extraction") or {}).get("status") == "NEEDS_REVIEW"
            ),
            "age_seconds": int(_now() - self.created),
        }


# Fields worth surviving a restart (documents/intermediate state are omitted).
_PERSIST_FIELDS = (
    "chat_id",
    "job_id",
    "status",
    "progress",
    "error",
    "client_signature",
    "profiles",
    "roster_order",
    "summary",
    "last_added",
    "view_state",
    "created",
    "last_access",
)


class SessionStore:
    def __init__(self, ttl_minutes: int = 120, data_dir: str = ""):
        self._ttl = ttl_minutes * 60
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._dir = ""
        if data_dir:
            try:
                d = os.path.join(data_dir, "sessions")
                os.makedirs(d, exist_ok=True)
                self._dir = d
            except OSError:
                self._dir = ""  # persistence unavailable -> memory-only
        self._load_all()

    # ---- lifecycle ----
    def create(self, chat_id: str) -> Session:
        """Create a fresh session for chat_id, replacing any existing one."""
        with self._lock:
            self._sweep_locked()
            s = Session(chat_id=chat_id)
            self._sessions[chat_id] = s
            return s

    def get(self, chat_id: str) -> Session | None:
        with self._lock:
            self._sweep_locked()
            s = self._sessions.get(chat_id)
            if s is not None:
                s.touch()
            return s

    def reset(self, chat_id: str) -> bool:
        with self._lock:
            self._delete_file(chat_id)
            return self._sessions.pop(chat_id, None) is not None

    def count(self) -> int:
        with self._lock:
            self._sweep_locked()
            return len(self._sessions)

    # ---- persistence (atomic JSON per chat; same TTL policy as memory) ----
    def _path(self, chat_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._:-]", "_", chat_id)[:128]
        return os.path.join(self._dir, safe + ".json")

    def persist(self, session: Session) -> None:
        """Write a READY session to disk (atomic). No-op when disabled/unready."""
        if not self._dir or session.status != "ready":
            return
        data = {k: getattr(session, k) for k in _PERSIST_FIELDS}
        data["processed_hashes"] = sorted(session.processed_hashes)
        path = self._path(session.chat_id)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError):
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _delete_file(self, chat_id: str) -> None:
        if not self._dir:
            return
        try:
            os.remove(self._path(chat_id))
        except OSError:
            pass

    def _load_all(self) -> None:
        if not self._dir:
            return
        cutoff = _now() - self._ttl
        try:
            names = os.listdir(self._dir)
        except OSError:
            return
        for fn in names:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self._dir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if float(data.get("last_access", 0)) < cutoff:
                    os.remove(path)
                    continue
                s = Session(chat_id=data["chat_id"])
                for k in _PERSIST_FIELDS:
                    if k in data:
                        setattr(s, k, data[k])
                s.processed_hashes = set(data.get("processed_hashes") or [])
                self._sessions[s.chat_id] = s
            except (OSError, ValueError, KeyError, TypeError):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ---- expiry ----
    def _sweep_locked(self) -> None:
        cutoff = _now() - self._ttl
        stale = [cid for cid, s in self._sessions.items() if s.last_access < cutoff]
        for cid in stale:
            self._sessions.pop(cid, None)
            self._delete_file(cid)
