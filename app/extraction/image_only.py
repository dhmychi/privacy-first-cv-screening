"""Detect image-only (scanned) pages so they can be routed to OCR instead of
being silently treated as empty."""

from __future__ import annotations

from typing import Any

_MIN_TEXT_CHARS = 15


def page_image_info(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        import pypdfium2 as pdfium
        from pypdfium2 import raw as pdfium_c
    except Exception:
        # Without the PDF backend we cannot inspect images; assume text pages.
        return out

    try:
        pdf = pdfium.PdfDocument(path)
    except Exception:
        return out
    try:
        for i in range(len(pdf)):
            try:
                page = pdf[i]
            except Exception:
                continue
            text_page = None
            try:
                text_page = page.get_textpage()
                txt = text_page.get_text_range() or ""
            except Exception:
                txt = ""
            try:
                image_count = sum(
                    1
                    for obj in page.get_objects()
                    if getattr(obj, "type", None) == pdfium_c.FPDF_PAGEOBJ_IMAGE
                )
            except Exception:
                image_count = 0
            if text_page is not None:
                text_page.close()
            page.close()
            stripped = len(txt.strip())
            out.append(
                {
                    "page": i + 1,
                    "text_len": stripped,
                    "image_count": image_count,
                    "image_only": stripped < _MIN_TEXT_CHARS and image_count > 0,
                }
            )
    finally:
        pdf.close()
    return out
