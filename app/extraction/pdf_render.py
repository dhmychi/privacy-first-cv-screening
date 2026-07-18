"""Render a PDF page to PNG bytes using pypdfium2 (no Poppler dependency)."""

from __future__ import annotations

import io

from app.extraction._pdfium import PDFIUM_LOCK


def render_png(path: str, page_index: int, dpi: int = 300) -> bytes:
    import pypdfium2 as pdfium

    # PDFium is not thread-safe; serialize every call (see app/extraction/_pdfium.py).
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(path)
        try:
            page = pdf[page_index]
            bitmap = page.render(scale=dpi / 72.0)
            image = bitmap.to_pil()  # copies the pixel data out of the bitmap
            bitmap.close()
            page.close()
        finally:
            pdf.close()
    # The PIL image owns its pixels now, so PNG encoding runs outside the lock.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
