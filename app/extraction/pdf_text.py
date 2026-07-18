"""Generic text + table extraction from a PDF's text layer (pdfplumber primary,
pypdf fallback), now LAYOUT-AWARE: multi-column and sidebar pages are detected
from word coordinates (a persistent vertical gutter) and each column is read
separately in the correct order, so two-column/designer CVs no longer interleave
lines from unrelated columns.

Detection is geometric and template-free: no section names, no CV assumptions.
It is deliberately conservative — a page is only split when a clear gutter
separates two substantial word groups that overlap vertically; anything
ambiguous falls back to pdfplumber's default reading order. Column order is
left-to-right, or right-to-left when the page is predominantly Arabic (RTL).
"""

from __future__ import annotations

from typing import Any

# --- conservative column-detection policy (geometry, not templates) ---
_MIN_WORDS = 40  # too few words -> not worth splitting
_MIN_GUTTER_PT = 16  # min horizontal white gap between columns (points)
_MIN_SIDE_FRACTION = 0.12  # each column must hold >= this fraction of words
_MIN_Y_OVERLAP = 0.35  # columns must coexist vertically (else it's header/footer)


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arab = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    letters = sum(1 for ch in text if ch.isalpha())
    return arab / letters if letters else 0.0


def _find_gutter(words: list[dict[str, Any]], page_width: float) -> float | None:
    """Return the x of a vertical gutter separating two substantial, vertically
    co-existing word groups, else None. Pure geometry."""
    if len(words) < _MIN_WORDS or not page_width:
        return None
    # merge word x-intervals into occupied spans
    spans: list[list[float]] = []
    for w in sorted(words, key=lambda w: w["x0"]):
        x0, x1 = float(w["x0"]), float(w["x1"])
        if spans and x0 <= spans[-1][1] + 2:  # 2pt slack
            spans[-1][1] = max(spans[-1][1], x1)
        else:
            spans.append([x0, x1])
    if len(spans) < 2:
        return None
    # widest gap between occupied spans, away from page edges
    best_gap, best_x = 0.0, None
    for a, b in zip(spans, spans[1:], strict=False):
        gap = b[0] - a[1]
        mid = (a[1] + b[0]) / 2.0
        if gap > best_gap and 0.15 * page_width < mid < 0.85 * page_width:
            best_gap, best_x = gap, mid
    if best_x is None or best_gap < _MIN_GUTTER_PT:
        return None
    left = [w for w in words if float(w["x1"]) <= best_x]
    right = [w for w in words if float(w["x0"]) >= best_x]
    n = len(words)
    if len(left) < n * _MIN_SIDE_FRACTION or len(right) < n * _MIN_SIDE_FRACTION:
        return None
    if len(left) + len(right) < n * 0.9:  # too many words straddle the gutter
        return None

    # the two sides must overlap vertically (a real column pair, not a banner)
    def _yspan(ws):
        return min(float(w["top"]) for w in ws), max(float(w["bottom"]) for w in ws)

    lt, lb = _yspan(left)
    rt, rb = _yspan(right)
    overlap = min(lb, rb) - max(lt, rt)
    shorter = min(lb - lt, rb - rt)
    if shorter <= 0 or overlap < shorter * _MIN_Y_OVERLAP:
        return None
    return best_x


def _extract_columns(page) -> tuple[str, int]:
    """Best-effort layout-aware text for one pdfplumber page.
    Returns (text, columns_used). Any anomaly -> default extraction."""
    try:
        default_text = page.extract_text() or ""
    except Exception:
        default_text = ""
    try:
        words = page.extract_words() or []
        gutter = _find_gutter(words, float(page.width))
        if gutter is None:
            return default_text, 1
        left = page.crop((0, 0, gutter, page.height))
        right = page.crop((gutter, 0, page.width, page.height))
        lt = (left.extract_text() or "").strip()
        rt = (right.extract_text() or "").strip()
        if not lt or not rt:
            return default_text, 1
        first, second = (rt, lt) if _arabic_ratio(default_text) > 0.30 else (lt, rt)
        return first + "\n\n" + second, 2
    except Exception:
        return default_text, 1


def read_pdf(path: str) -> list[dict[str, Any]]:
    """Return one dict per page: {page, text, tables, columns}. Never raises on
    a bad page; a failed page yields empty text/tables so callers can flag it."""
    pages: list[dict[str, Any]] = []
    try:
        import pdfplumber
    except Exception:  # pragma: no cover - import guard
        return _pypdf_only(path)

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text, cols = _extract_columns(page)
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                pages.append({"page": i + 1, "text": text, "tables": tables, "columns": cols})
    except Exception:
        # Corrupt / non-PDF bytes must never abort a whole batch: try pypdf,
        # and let a total failure surface as an empty (unreadable) document.
        return _pypdf_only(path)
    return pages


def _pypdf_only(path: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append({"page": i + 1, "text": text, "tables": [], "columns": 1})
    except Exception:
        pass
    return pages
