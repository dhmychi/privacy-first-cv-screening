"""Within-batch duplicate detection (ephemeral — only across this upload).

Two candidates are duplicates only if they share a *real* contact key (normalized
email or last-9 phone digits) AND a similar name. Template/placeholder contacts —
shared by many candidates (e.g. Canva samples with hello@reallygreatsite.com or
+123-456-7890) — are NOT treated as identity: any key that appears in 3+ candidates
is ignored, and known placeholder patterns are ignored outright. This prevents the
false "everyone is a duplicate" collapse seen on template CV batches.
"""

from __future__ import annotations

import re
from typing import Any

from . import anchors

_PLACEHOLDER_EMAIL = (
    "reallygreatsite",
    "example.com",
    "example.org",
    "email.com",
    "yourname",
    "your.email",
    "hello@",
    "name@",
    "company.com",
    "domain.com",
    "sample",
    "test@",
    "youremail",
    "mail.com",
    "website.com",
    "abc@",
    "xyz@",
)


def _placeholder_email(e: str) -> bool:
    el = (e or "").lower()
    return any(p in el for p in _PLACEHOLDER_EMAIL)


def _placeholder_phone(k: str) -> bool:
    if not k:
        return True
    if len(set(k)) <= 2:  # 000000000, 111111111
        return True
    if k in ("123456789", "234567890", "012345678", "987654321"):
        return True
    if re.match(r"^012345|^123456", k):
        return True
    return False


def _name_tokens(p: dict[str, Any]) -> set:
    nm = (p.get("identity", {}).get("full_name", {}) or {}).get("value") or ""
    return {t for t in re.split(r"\s+", nm.lower().strip()) if len(t) > 1}


def _keys(p: dict[str, Any]) -> set:
    ident = p.get("identity", {})
    out = set()
    for e in ident.get("emails", []):
        if not _placeholder_email(e):
            out.add("e:" + e.lower())
    for ph in ident.get("phones", []):
        k = anchors.phone_key(ph)
        if k and not _placeholder_phone(k):
            out.add("p:" + k)
    return out


def mark_duplicates(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # frequency of each key across the batch; keys shared by 3+ candidates are
    # template artifacts, not a real shared identity.
    freq: dict[str, int] = {}
    cand_keys: dict[str, set] = {}
    for p in profiles:
        ks = _keys(p)
        cand_keys[p["candidate_id"]] = ks
        for k in ks:
            freq[k] = freq.get(k, 0) + 1

    seen: dict[str, str] = {}  # valid key -> first candidate_id
    for p in profiles:
        ks = {k for k in cand_keys[p["candidate_id"]] if freq.get(k, 0) <= 2}
        dup_of = None
        for k in ks:
            if k in seen:
                # confirm with a light name check (same person, not just shared key)
                first = next(x for x in profiles if x["candidate_id"] == seen[k])
                ta, tb = _name_tokens(p), _name_tokens(first)
                if not ta or not tb or (ta & tb):
                    dup_of = seen[k]
                    break
        if dup_of:
            first = next(x for x in profiles if x["candidate_id"] == dup_of)
            # Same identity ≠ same document. Near-identical content is a true
            # duplicate; materially DIFFERENT content is a CONFLICT (two CV
            # versions / possibly two people sharing a contact) that a human
            # must resolve — never silently kept-first as a "duplicate".
            if _content_similarity(p, first) >= CONTENT_DUP_JACCARD:
                _mark_dup(p, dup_of)
            else:
                _mark_conflict(p, first)
        else:
            for k in ks:
                seen.setdefault(k, p["candidate_id"])

    # Fallback: catch identical CVs that share only PLACEHOLDER contacts (template
    # samples) — same valid full name AND near-identical skills. Skill overlap guards
    # against merging two DIFFERENT people who happen to share a name.
    def _name(p):
        v = (p.get("identity", {}).get("full_name", {}) or {}).get("value") or ""
        return re.sub(r"\s+", " ", v.strip().lower())

    def _skills(p):
        return {s["name"].lower() for s in p.get("skills", [])}

    for i, p in enumerate(profiles):
        if p.get("duplicate_of"):
            continue
        np_, sp = _name(p), _skills(p)
        if len(np_) < 5 or " " not in np_:  # need a real full name (first + last)
            continue
        for q in profiles[:i]:
            if q.get("duplicate_of"):
                continue
            if _name(q) == np_:
                sq = _skills(q)
                jac = (len(sp & sq) / len(sp | sq)) if (sp | sq) else 0.0
                if jac >= 0.7:
                    # Same name + overlapping skills is NOT enough to call it a
                    # duplicate: two different people in the same profession share
                    # a skill vocabulary. Only near-identical DOCUMENT CONTENT is a
                    # true duplicate (a re-uploaded template); materially different
                    # content (different employers/dates/tenure) is a CONFLICT for a
                    # human to resolve - never a silent exclusion. Mirrors the
                    # primary-key path so both routes decide dup-vs-conflict the same.
                    if _content_similarity(p, q) >= CONTENT_DUP_JACCARD:
                        _mark_dup(p, q["candidate_id"])
                    else:
                        _mark_conflict(p, q)
                    break
    return profiles


# Token-set Jaccard above this = the same document content (re-uploaded /
# re-rendered copy). Below it, two CVs sharing an identity anchor differ
# materially -> conflicting versions, a human decision, not a duplicate.
CONTENT_DUP_JACCARD = 0.80


def _tokens(p: dict[str, Any]) -> set:
    txt = " ".join((pg.get("text") or "") for pg in p.get("_pages") or [])
    return set(re.findall(r"[a-z0-9؀-ۿ]{2,}", txt.lower()))


def _content_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        # No raw text to compare (e.g. legacy profiles) — preserve the previous
        # duplicate behavior rather than inventing conflicts.
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _mark_conflict(p, first):
    reasons = p.setdefault("extraction", {}).setdefault("reasons", [])
    if "CONFLICTING_VERSIONS" not in reasons:
        reasons.append("CONFLICTING_VERSIONS")
    p["conflict_with"] = first["candidate_id"]
    if p["extraction"].get("status") == "OK":
        p["extraction"]["status"] = "NEEDS_REVIEW"
    # The kept (first) profile carries a verification note but stays scored —
    # visible to the reviewer on every surface, excluded from nothing.
    first.setdefault("verification_notes", [])
    note = (
        f"another CV in this batch (#{p.get('display_no')}) shares the same "
        f"contact details with different content — verify which is current"
    )
    if note not in first["verification_notes"]:
        first["verification_notes"].append(note)


def _mark_dup(p, dup_of):
    p["duplicate_of"] = dup_of
    reasons = p.setdefault("extraction", {}).setdefault("reasons", [])
    if "DUPLICATE" not in reasons:
        reasons.append("DUPLICATE")
    if p["extraction"].get("status") == "OK":
        p["extraction"]["status"] = "NEEDS_REVIEW"
