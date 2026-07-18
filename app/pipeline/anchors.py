"""Deterministic, high-precision extraction (regex + date arithmetic).

These are the fields we must NOT trust an LLM to invent: emails, phones, links,
and total years of experience. The LLM may *point* at dated roles, but the year
math is done here so no number is hallucinated (the financial_calculator
principle). Also exposes per-page identity anchors used for candidate
segmentation and within-batch dedup.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_LINK_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]]+", re.I)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_PRESENT_RE = re.compile(
    r"present|current|now|till\s*date|to\s*date|ongoing|till\s*now|الحالي|الآن|حتى\s*الآن|حالي",
    re.I,
)

# International (+/00 prefix), local 0-leading, and keyword-led phone numbers.
_PHONE_INTL_RE = re.compile(r"(?:\+|00)\s?\d[\d\s().\-]{6,16}\d")
_PHONE_LOCAL_RE = re.compile(r"(?<!\d)0\d[\d\s().\-]{7,12}\d(?!\d)")
_PHONE_KEYED_RE = re.compile(
    r"(?:tel|mobile|phone|cell|fax|whats\s*app|whatsapp|جوال|هاتف|موبايل|تليفون|الجوال|الهاتف)"
    r"[:\s.]*([+\d][\d\s().\-]{6,16}\d)",
    re.I,
)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(s: str | None) -> str:
    return (s or "").translate(_ARABIC_DIGITS)


# Placeholder emails are not real contacts: RFC-2606 reserved domains and
# universal template local-parts ("your.email@..."). Filtering them keeps
# identity anchors, dedup keys, and the roster honest. Generic by standard /
# template vocabulary - not tied to any dataset.
_PLACEHOLDER_DOMAIN_RE = re.compile(
    r"(^|\.)example\.(com|net|org)$|\.(test|invalid|localhost|example)$", re.I
)
_PLACEHOLDER_LOCAL_RE = re.compile(
    r"^(your[._-]?(e?mail|name)(here)?|e?mail[._-]?here|first[._-]?name[._-]?"
    r"last[._-]?name|firstname[._-]?lastname|name[._-]?surname|sample[._-]?e?mail)$",
    re.I,
)


def is_placeholder_email(email: str) -> bool:
    local, _, domain = (email or "").partition("@")
    if not domain:
        return True
    return bool(_PLACEHOLDER_DOMAIN_RE.search(domain) or _PLACEHOLDER_LOCAL_RE.match(local))


def find_emails(text: str | None) -> list[str]:
    out: list[str] = []
    for m in _EMAIL_RE.findall(text or ""):
        e = m.strip().strip(".").lower()
        if e and e not in out and not is_placeholder_email(e):
            out.append(e)
    return out


def find_links(text: str | None) -> list[str]:
    out: list[str] = []
    for m in _LINK_RE.findall(text or ""):
        u = m.rstrip(".,);:]")
        if u and u not in out:
            out.append(u)
    return out


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def find_phones(text: str | None) -> list[str]:
    """Precision-first: only accept international (+/00), local 0-leading, or
    keyword-led numbers, and require a sane digit count — so year ranges like
    '2018-2023' are not misread as phone numbers."""
    t = normalize_digits(text or "")
    found: list[str] = []
    keys: list[str] = []

    def _add(raw: str, lo: int, hi: int) -> None:
        raw = raw.strip()
        d = _digits(raw)
        if not (lo <= len(d) <= hi):
            return
        k = d[-9:]
        if k in keys:
            return
        keys.append(k)
        found.append(raw)

    for m in _PHONE_KEYED_RE.findall(t):
        _add(m, 8, 15)
    for m in _PHONE_INTL_RE.findall(t):
        _add(m, 9, 15)
    for m in _PHONE_LOCAL_RE.findall(t):
        _add(m, 9, 11)
    return found


def phone_key(phone: str) -> str:
    """Stable dedup key: the last 9 significant digits."""
    return _digits(normalize_digits(phone))[-9:]


def primary_anchor(text: str | None) -> str | None:
    """A page/candidate's primary identity anchor: first email, else first phone."""
    emails = find_emails(text)
    if emails:
        return "email:" + emails[0]
    phones = find_phones(text)
    if phones:
        return "phone:" + phone_key(phones[0])
    return None


# --------------------------------------------------------------- year math
def parse_year_token(s: str | None) -> int | None:
    s = normalize_digits(s)
    if not s:
        return None
    if _PRESENT_RE.search(s):
        return _dt.date.today().year
    m = _YEAR_RE.search(s)
    return int(m.group()) if m else None


# Month vocabulary (generic English + Arabic Gregorian month names/abbrevs).
_MONTHS = {}
for _i, _names in enumerate(
    [
        ("january", "jan", "يناير", "كانون الثاني"),
        ("february", "feb", "فبراير", "شباط"),
        ("march", "mar", "مارس", "آذار"),
        ("april", "apr", "أبريل", "ابريل", "نيسان"),
        ("may", "مايو", "أيار"),
        ("june", "jun", "يونيو", "حزيران"),
        ("july", "jul", "يوليو", "تموز"),
        ("august", "aug", "أغسطس", "اغسطس", "آب"),
        ("september", "sep", "sept", "سبتمبر", "أيلول"),
        ("october", "oct", "أكتوبر", "اكتوبر", "تشرين الأول"),
        ("november", "nov", "نوفمبر", "تشرين الثاني"),
        ("december", "dec", "ديسمبر", "كانون الأول"),
    ],
    start=1,
):
    for _n in _names:
        _MONTHS[_n] = _i

_MONTH_WORD_RE = re.compile(r"[A-Za-z؀-ۿ]{3,}")
_MM_YYYY_RE = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)\d{2})(?!\d)")


def parse_date_token(s: str | None) -> tuple[int | None, int | None]:
    """(year, month|None). 'Present'-family tokens -> today's year AND month, so
    contiguous careers compute to the real current date. Month is read from a
    month name (en/ar) or a numeric MM/YYYY; a bare year keeps month=None."""
    s = normalize_digits(s)
    if not s:
        return None, None
    if _PRESENT_RE.search(s):
        today = _dt.date.today()
        return today.year, today.month
    m = _YEAR_RE.search(s)
    if not m:
        return None, None
    year = int(m.group())
    mm = _MM_YYYY_RE.search(s)
    if mm:
        return year, int(mm.group(1))
    low = s.lower()
    for w in _MONTH_WORD_RE.findall(low):
        if w in _MONTHS:
            return year, _MONTHS[w]
    return year, None


def compute_total_years(experiences: list[dict[str, Any]]) -> tuple[float | None, int]:
    """Union of dated employment intervals -> total years (no double counting of
    overlapping roles). MONTH-AWARE when the CV states months ('June 2015',
    '03/2022'): a Dec 2018 -> Jan 2019 role change stays contiguous and totals
    match real tenure (year-only dates keep the previous year arithmetic
    exactly, so existing outputs do not drift). Returns (years, n_roles_used)."""
    today = _dt.date.today()
    horizon = (today.year + 1) * 12 + today.month
    intervals: list[tuple[int, int, bool]] = []  # (start_idx, end_idx, months_known)
    for e in experiences or []:
        ay, am = parse_date_token(str(e.get("start", "")))
        by, bm = parse_date_token(str(e.get("end", "")))
        if not (ay and by) or by < ay or not (1950 <= ay):
            continue
        # Months count only when BOTH endpoints state one ('June 2015 - Dec 2018',
        # 'March 2022 - Present'); a bare-year endpoint keeps the whole interval
        # on the historical year arithmetic so existing outputs do not drift.
        months_known = am is not None and bm is not None
        if not months_known:
            am = bm = None
        a_idx = ay * 12 + ((am - 1) if am else 0)
        b_idx = by * 12 + ((bm - 1) if bm else 0)
        if b_idx < a_idx or a_idx > horizon or b_idx > horizon:
            continue
        intervals.append((a_idx, b_idx, months_known))
    if not intervals:
        return None, 0
    intervals.sort()
    # Merge: adjacent months (gap <= 1 month) are contiguous employment.
    merged: list[list[Any]] = []
    for st, en, mk in intervals:
        if merged and st <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], en)
            merged[-1][2] = merged[-1][2] or mk
        else:
            merged.append([st, en, mk])
    total_months = 0
    for st, en, mk in merged:
        # month-known spans count inclusively (Jun 2015..Dec 2018 = 43 months);
        # year-only spans keep the historical (end_year - start_year) arithmetic.
        total_months += (en - st + 1) if mk else (en - st)
    return round(total_months / 12.0, 1), len(intervals)


_PRESENT_WORDS = r"present|current|now|ongoing|till\s*date|to\s*date|الحالي|الآن|حتى\s*الآن"
# "2018 - Present", "2014–2018", "2015 to 2020" (dash variants + words)
_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:[-–—~/]|to|until|till|through|إلى|حتى)\s*"
    r"((?:19|20)\d{2}|" + _PRESENT_WORDS + r")",
    re.I,
)
# looser: two years separated by up to 4 non-digit chars (encoding-tolerant dash)
_RANGE2_RE = re.compile(r"((?:19|20)\d{2})[^\d\n]{1,4}((?:19|20)\d{2})")
# "12 years", "8+ yrs", "10 years of experience", "خبرة 7 سنوات"
_PROSE_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?|سنوات|سنة|عام|أعوام)", re.I)


def stated_years_claim(text: str | None) -> float | None:
    """The candidate's own explicit total-experience claim ('with 8 years of
    experience', 'خبرة 7 سنوات') from the head of the CV, if any. Used ONLY to
    surface a verification note when it contradicts the dated-role arithmetic —
    it never changes the computed value (Present-=-today policy stands)."""
    text = normalize_digits(text or "")
    if not text:
        return None
    best = None
    for m in _PROSE_YEARS_RE.finditer(text[:900]):
        v = int(m.group(1))
        if 1 <= v <= 45 and (best is None or v > best):
            best = v
    return float(best) if best is not None else None


def years_from_text(text: str | None) -> float | None:
    """Deterministic fallback for total years when the LLM did not populate dated
    role fields. (1) Union of YYYY–YYYY / YYYY–Present ranges anywhere in the raw
    text (dash-variant and encoding tolerant); else (2) an explicit 'N years'
    statement near the top. Year math stays deterministic — no LLM, no hallucination."""
    text = normalize_digits(text or "")
    if not text:
        return None
    this_year = _dt.date.today().year
    intervals: list[tuple[int, int]] = []
    for rx in (_RANGE_RE, _RANGE2_RE):
        for m in rx.finditer(text):
            a = parse_year_token(m.group(1))
            b = parse_year_token(m.group(2))
            if a and b and b >= a and 1950 <= a <= this_year + 1 and b <= this_year + 1:
                intervals.append((a, b))
    if intervals:
        intervals.sort()
        merged: list[list[int]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        total = sum(e - s for s, e in merged)
        if 0 < total <= 55:
            return float(total)
    best = None
    for m in _PROSE_YEARS_RE.finditer(text[:900]):
        v = int(m.group(1))
        if 1 <= v <= 45 and (best is None or v > best):
            best = v
    return float(best) if best is not None else None
