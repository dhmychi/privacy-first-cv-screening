"""Customer-facing phrasing for extraction/review flag codes (en/ar).

One shared vocabulary so the screening facts, the fit report and any future
surface describe a flagged CV the SAME honest way. Raw codes like SPARSE_TEXT
are internal; a customer must never see them — and 'could not be read' must be
reserved for files that were actually unreadable, never for a readable-but-thin
or non-CV document.
"""

from __future__ import annotations

from collections.abc import Iterable

REASON_PHRASES: dict[str, dict[str, str]] = {
    "UNREADABLE_FILE": {
        "en": "the file could not be read",
        "ar": "تعذرت قراءة الملف",
    },
    "LOW_OCR_CONFIDENCE": {
        "en": "scan quality too low to read reliably",
        "ar": "جودة المسح الضوئي منخفضة جدًا للقراءة الموثوقة",
    },
    "SPARSE_TEXT": {
        "en": "readable but contains very little content",
        "ar": "قابلة للقراءة لكنها تحتوي على محتوى قليل جدًا",
    },
    "SPARSE_EXTRACTION": {
        "en": "readable but too little career information could be extracted",
        "ar": "قابلة للقراءة لكن تعذر استخراج معلومات مهنية كافية",
    },
    "NO_IDENTITY_ANCHOR": {
        "en": "no name or contact details found — may not be a CV",
        "ar": "لا يوجد اسم أو بيانات تواصل — قد لا تكون سيرة ذاتية",
    },
    "MULTIPLE_CVS_IN_FILE": {
        "en": "the file may contain more than one CV",
        "ar": "قد يحتوي الملف على أكثر من سيرة ذاتية",
    },
    "LLM_UNAVAILABLE": {
        "en": "the extraction engine was unavailable for this file",
        "ar": "محرك الاستخراج لم يكن متاحًا لهذا الملف",
    },
    "DUPLICATE": {
        "en": "duplicate of another CV in this batch",
        "ar": "نسخة مكررة من سيرة ذاتية أخرى في هذه الدفعة",
    },
    "CONFLICTING_VERSIONS": {
        "en": "shares the same contact details as another CV but with different "
        "content — verify which version is current",
        "ar": "تحمل بيانات تواصل مطابقة لسيرة أخرى لكن بمحتوى مختلف — تحقق من النسخة الأحدث",
    },
    "INJECTION_SUSPECTED": {
        "en": "the CV contains text that appears aimed at influencing the "
        "screening system (it was ignored); manual review required",
        "ar": "تحتوي السيرة على نص يبدو موجّهًا للتأثير على نظام الفرز (تم "
        "تجاهله)؛ مطلوب مراجعة يدوية",
    },
}

_FALLBACK = {
    "en": "low extraction confidence — review manually",
    "ar": "ثقة الاستخراج منخفضة — تحتاج مراجعة يدوية",
}


def describe(flags: Iterable[str], lang: str = "en") -> str:
    """Join the known flags into one honest customer-facing clause."""
    lang = lang if lang in ("en", "ar") else "en"
    out: list[str] = []
    for f in flags or []:
        p = REASON_PHRASES.get(str(f).upper())
        if p and p[lang] not in out:
            out.append(p[lang])
    if not out:
        return _FALLBACK[lang]
    joiner = "؛ " if lang == "ar" else "; "
    return joiner.join(out)
