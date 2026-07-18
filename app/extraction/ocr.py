"""OCR layer. MVP: local Tesseract (ara+eng) only, used for evidence/assist.

Hard rules enforced here:
  * OCR output is NEVER authoritative on its own (callers must reconcile/flag).
  * No cloud OCR endpoint is contacted unless explicitly allowed.
  * A disabled local-vision cross-check hook is wired for a later phase.
"""

from __future__ import annotations

import shutil


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_langs() -> list:
    if not tesseract_available():
        return []
    try:
        import subprocess

        out = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10
        )
        langs = [ln.strip() for ln in out.stdout.splitlines()[1:] if ln.strip()]
        return langs
    except Exception:
        return []


def _osd_rotation(img) -> int:
    """Detect page rotation (0/90/180/270) via Tesseract OSD. 0 on failure."""
    import re

    import pytesseract

    try:
        osd = pytesseract.image_to_osd(img)
        m = re.search(r"Rotate:\s*(\d+)", osd)
        return int(m.group(1)) % 360 if m else 0
    except Exception:
        return 0


def _skew_angle(img) -> float:
    """Estimate page skew (degrees) via horizontal projection-profile sharpness —
    content-agnostic, no dependence on OCR word grouping (which fails unpredictably
    on some pages). Returns the rotation that best aligns text rows; 0 if negligible.
    Uses Pillow 'resize width->1' to get the per-row projection (fast, pure Pillow)."""
    from PIL import Image, ImageOps

    try:
        g = ImageOps.grayscale(img)
        w, h = g.size
        if w > 1000:  # downscale for speed; angle is scale-invariant
            g = g.resize((1000, max(1, int(h * 1000 / w))))
        g = ImageOps.invert(ImageOps.autocontrast(g))  # text -> bright on dark

        def score(a: float) -> float:
            r = g.rotate(a, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)
            # 'L'-mode single-column pixels; tobytes() avoids the deprecated
            # Image.getdata() and is equivalent for 8-bit grayscale.
            vals = list(r.resize((1, r.height), Image.Resampling.BILINEAR).tobytes())
            return sum((vals[i + 1] - vals[i]) ** 2 for i in range(len(vals) - 1))

        coarse = max(range(-8, 9), key=lambda a: score(float(a)))  # 1 deg grid
        best = max((coarse + 0.2 * k for k in range(-4, 5)), key=score)  # refine to 0.2 deg
        return best if abs(best) >= 0.3 else 0.0
    except Exception:
        return 0.0


def _variants(img):
    """Yield candidate images for OCR, cheapest/least-destructive first.
    Plain grayscale is best for clean renders; the enhanced variant only wins on
    faded/low-quality scans. Upscaling helps genuinely low-resolution pages. An
    illumination-normalised variant is added first when the page has uneven lighting."""
    from PIL import Image, ImageChops, ImageFilter, ImageOps

    g = ImageOps.grayscale(img)
    w, h = g.size
    if max(w, h) < 2200:  # upscale low-res scans toward ~300dpi
        s = 2200.0 / max(w, h)
        g = g.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    try:  # uneven illumination (shadows / gradient)?
        lo, hi = g.filter(ImageFilter.GaussianBlur(60)).getextrema()
        if hi - lo > 110:  # flat-field: remove the blurred background
            bg = g.filter(ImageFilter.GaussianBlur(30))
            yield ImageOps.autocontrast(ImageChops.invert(ImageChops.subtract(bg, g)), cutoff=1)
    except Exception:
        pass
    yield g  # plain grayscale (clean pages)
    e = ImageOps.autocontrast(g, cutoff=2)  # faded/blurred: contrast + unsharp deblur
    yield e.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))


def _image_to_lines(proc, langs: str, config: str):
    """Run image_to_data and rebuild LINE structure from word geometry.
    Returns (text, mean_confidence, n_words)."""
    import pytesseract

    data = pytesseract.image_to_data(
        proc, lang=langs, config=config, output_type=pytesseract.Output.DICT
    )
    lines: dict = {}
    confs = []
    n = len(data.get("text", []))
    page_nums = data.get("page_num", [0] * n)
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        key = (page_nums[i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        try:
            c = float(data["conf"][i])
            if c >= 0:
                confs.append(c)
        except (TypeError, ValueError):
            pass
    text = "\n".join(" ".join(lines[k]) for k in sorted(lines))
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf, len(confs)


def correct_orientation_and_skew(img):
    """Rotate an image upright (OSD 90/180/270) then deskew it (fine angle). Geometry-only
    correction so downstream sees upright text — essential for photographed / rotated pages."""
    try:
        rot = _osd_rotation(img)
        if rot:
            img = img.rotate(-rot, expand=True)  # OSD 'Rotate' is clockwise
    except Exception:
        pass
    try:
        ang = _skew_angle(img.convert("L"))  # fine deskew for slightly-skewed scans
        if 0.4 < abs(ang) < 15:
            img = img.rotate(ang, expand=True, fillcolor=255)
    except Exception:
        pass
    return img


def ocr_png(png_bytes: bytes, langs: str = "ara+eng") -> tuple[str, float]:
    """Return (text, mean_confidence 0-100). Auto-corrects page orientation and
    enhances the image (Pillow only) before OCR, then reconstructs the original
    LINE structure. Picks the best of variants by confident word count x confidence,
    so dense / multi-column CVs are read fully."""
    import io

    from PIL import Image

    img = correct_orientation_and_skew(Image.open(io.BytesIO(png_bytes)))
    try:
        cands = list(_variants(img))
    except Exception:
        cands = [img.convert("L") if img.mode != "L" else img]

    best = ("", 0.0, -1.0)
    for proc in cands:
        try:
            text, conf, words = _image_to_lines(proc, langs, "--oem 1 --psm 3")
        except Exception:
            continue
        score = words * (conf / 100.0)  # favour many confident words
        if score > best[2]:
            best = (text, conf, score)
        if conf >= 85 and words >= 40:  # only skip extra passes when plain is clearly clean
            break
    return best[0], best[1]


def ocr_pages(path: str, page_indices, langs: str = "ara+eng", dpi: int = 300) -> dict:
    """OCR only the requested page indices (0-based). Returns
    {page_index: (text, mean_confidence)}. Renders with pypdfium2 (no Poppler)."""
    from . import pdf_render

    out = {}
    for idx in page_indices:
        try:
            png = pdf_render.render_png(path, idx, dpi=dpi)
            text, conf = ocr_png(png, langs=langs)
            out[idx] = (text, conf)
        except Exception:
            out[idx] = ("", 0.0)
    return out


def ocr_image_file(path: str, langs: str = "ara+eng") -> tuple[str, float]:
    """OCR a standalone image file (photographed CV). Returns (text, mean_confidence)."""
    try:
        with open(path, "rb") as fh:
            return ocr_png(fh.read(), langs=langs)
    except Exception:
        return "", 0.0


def vision_cross_check(png_bytes: bytes, url: str, allow_cloud: bool) -> str | None:
    """Disabled in MVP (url blank). Returns None unless a *local* vision endpoint is
    configured and permitted. Never used to authorise a result. Phase-2 hook only."""
    if not url:
        return None
    if not allow_cloud and not _is_local(url):
        # Refuse to send candidate PII off-box without explicit approval.
        return None
    return None  # Phase 2: POST image + structured prompt to a LOCAL vision model.


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "host.docker.internal", "::1"))
