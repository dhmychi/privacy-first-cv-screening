"""Turn acquired documents into candidate units.

MVP rule: 1 document = 1 candidate (the most reliable separation). When a single
PDF appears to contain MULTIPLE distinct people (more than one primary email
anchor across its pages), we do NOT silently merge or mis-split — we keep it as
one candidate but flag ``multi_cv`` so it is routed to human review. Full
boundary-splitting of multi-CV PDFs is deferred to the production phase.
"""

from __future__ import annotations

from typing import Any

from . import anchors


def documents_to_candidates(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for doc in documents:
        pages = doc.get("pages", [])
        full_text = "\n".join((p.get("text") or "") for p in pages)

        # distinct email anchors across pages -> possible multiple CVs in one file
        distinct_emails = []
        for p in pages:
            for e in anchors.find_emails(p.get("text")):
                if e not in distinct_emails:
                    distinct_emails.append(e)
        multi_cv = len(distinct_emails) > 1 and len(pages) > 1

        page_nums = [p["page"] for p in pages] or [0]
        candidates.append(
            {
                "document": doc.get("file"),
                "container": doc.get("container"),
                "kind": doc.get("kind"),
                "pages": pages,
                "page_range": [min(page_nums), max(page_nums)],
                "full_text": full_text,
                "text_chars": doc.get("text_chars", len(full_text)),
                "page_count": len(pages),
                "unreadable": bool(doc.get("unreadable")),
                "ocr_used": doc.get("ocr_page_count", 0) > 0,
                "vlm_used": doc.get("vlm_page_count", 0) > 0,
                "mean_ocr_confidence": doc.get("mean_ocr_confidence"),
                "multi_cv": multi_cv,
                "distinct_emails": distinct_emails,
            }
        )
    return candidates
