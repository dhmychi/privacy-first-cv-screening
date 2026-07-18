"""Code-rendered fit-scoring report. NO model in this path: every heading,
row, KPI, caption and sentence is produced by deterministic templates over the
engine's structured results, so reasoning leaks are structurally impossible and
the same inputs render byte-identical output.

The layout mirrors the established customer-facing format (title, KPI line,
Top-N caption, ranked table with score bars, evidence, gaps/risks, shortlist,
review lane, disclaimer) in English or Arabic.
"""

from __future__ import annotations

from typing import Any

from ..reasons import describe as describe_reasons

LEVELS = {
    "en": {
        "strong": "\U0001f7e2 Strong Fit",
        "good": "\U0001f7e1 Good Fit",
        "partial": "\U0001f7e0 Partial Fit",
        "weak": "\U0001f534 Weak Fit",
    },
    "ar": {
        "strong": "\U0001f7e2 ملاءمة قوية",
        "good": "\U0001f7e1 ملاءمة جيدة",
        "partial": "\U0001f7e0 ملاءمة جزئية",
        "weak": "\U0001f534 ملاءمة ضعيفة",
    },
}

T: dict[str, dict[str, Any]] = {
    "en": {
        "title": "# HR Fit Scoring Results",
        "summary": "**Role: {role} - {n} candidates scored.** Top match: **{top_name}** - {top_score}% ({top_level}).",
        "summary_tie": "**Role: {role} - {n} candidates scored.** {k} candidates tie at the "
        "top with {top_score}% ({top_level}): {names}.",
        "kpi": "_Scored: {scored} · Needs review (not scored): {review} · Duplicates excluded: {dups}._",
        "tie_note": "_Candidates with equal scores share a rank; their order in the table "
        "follows requirements met, then verified years, then CV # — not merit._",
        "table_h": "## Ranked Fit Table",
        "caption": "_Top {shown} of {n} scored candidates (duplicates and unreadable CVs excluded)._",
        "hidden_note": "_Additional lower-ranked candidates are not shown in this summary._",
        "cols": "| Rank | CV # | Name | Fit Score | Level | Matched | Missing |",
        "sep": "|:--:|--:|---|:--:|:--:|---|---|",
        "none": "None",
        "bands": "_Fit bands: Strong 80-100 · Good 60-79 · Partial 40-59 · Weak below 40. "
        "≈ marks a partially-met requirement. Scores are computed by a "
        "deterministic engine from evidenced requirements._",
        "evidence_h": "## Evidence - Top Candidates",
        "gaps_h": "## Gaps / Risks",
        "short_h": "## Interview Shortlist",
        "review_h": "## Needs Review - Not Scored",
        "verify": "verify at interview",
        "ocr_note": "OCR-read",
        "yrs": "{y:g} years experience",
        "gap_missing": "**{label}:** not evidenced by {c} of {n} scored candidates.",
        "gap_unverified": "**Unverifiable fields:** {c} candidate(s) had requirements that "
        "could not be verified from the CV (marked unverified, excluded "
        "from their score denominator).",
        "gap_soft": "**Not assessable from CVs:** {items} - assess at interview.",
        "gap_hardcap": "**Hard-minimum shortfalls:** {c} candidate(s) miss a stated hard "
        "requirement (score capped).",
        "short_one": "**{name} (CV #{no})** ranks first at {score}% ({level_word}), meeting "
        "{met} of {tot} assessable requirements{yrs_part}.{verify_part}",
        "short_tie": "**{names}** tie at {score}% ({level_word}) — treat them as equal "
        "candidates; the table order is a deterministic tie-break, not merit."
        "{verify_part}",
        "short_more": " **{name} (CV #{no})** follows at {score}% with {met} of {tot} "
        "requirements evidenced.",
        "short_verify": " Verify at interview: {items}.",
        "short_none": "No candidate could be scored against this job description with "
        "sufficient evidence. Review the batch manually or refine the JD.",
        "short_weak": "**No candidate reaches an interview-ready fit for this role** — the "
        "best result is {name} (CV #{no}) at {score}% (Weak Fit). An interview "
        "shortlist is not recommended from this batch; consider widening the "
        "pool or revising the requirements.",
        "short_partial": "**{name} (CV #{no})** is the closest match at {score}% (Partial "
        "Fit). Proceed to interview only if the pool cannot be widened, and "
        "verify the missing requirements first: {missing}.",
        "review_row": "- **{name}** (CV #{no}) - {reason}.",
        "review_reason_ie": "insufficient verifiable evidence to score fairly "
        "({pct}% of required items verifiable)",
        "disclaimer": "_Decision support for HR - not a final hiring decision._",
        "level_words": {
            "strong": "Strong Fit",
            "good": "Good Fit",
            "partial": "Partial Fit",
            "weak": "Weak Fit",
        },
        "not_scorable": (
            "The job description could not be scored reliably: {reason}\n\n"
            "Please provide a job description with concrete requirements "
            "(skills, experience, education) and I will produce a full "
            "fit-scoring report."
        ),
    },
    "ar": {
        "title": "# نتائج تقييم الملاءمة الوظيفية (Fit Scoring)",
        "summary": "**الوظيفة: {role} - تم تقييم {n} مرشحًا.** أفضل مرشح: **{top_name}** - {top_score}% ({top_level}).",
        "summary_tie": "**الوظيفة: {role} - تم تقييم {n} مرشحًا.** يتعادل {k} مرشحين في الصدارة "
        "بنسبة {top_score}% ({top_level}): {names}.",
        "kpi": "_تم التقييم: {scored} · بحاجة لمراجعة (لم يتم تقييمهم): {review} · نسخ مكررة مستبعدة: {dups}._",
        "tie_note": "_المرشحون المتساوون في الدرجة يتشاركون الترتيب نفسه؛ ترتيبهم داخل الجدول يتبع "
        "عدد المتطلبات المحققة ثم سنوات الخبرة ثم رقم السيرة — وليس الأفضلية._",
        "table_h": "## جدول الترتيب (Ranked Fit Table)",
        "caption": "_أفضل {shown} من أصل {n} مرشحًا تم تقييمهم (تم استبعاد النسخ المكررة والسير غير المقروءة)._",
        "hidden_note": "_لا تُعرض بقية السير الذاتية الأدنى ترتيبًا في هذا الملخص._",
        "cols": "| الترتيب (Rank) | رقم السيرة (CV #) | الاسم (Name) | الدرجة (Fit Score) | المستوى (Level) | المطابق (Matched) | الناقص (Missing) |",
        "sep": "|:--:|--:|---|:--:|:--:|---|---|",
        "none": "لا يوجد",
        "bands": "_نطاقات الملاءمة: قوية 80-100 · جيدة 60-79 · جزئية 40-59 · ضعيفة أقل من 40. "
        "علامة ≈ تعني متطلبًا محققًا جزئيًا. تُحسب الدرجات بمحرك حتمي اعتمادًا على "
        "المتطلبات المدعومة بالأدلة._",
        "evidence_h": "## الأدلة - أفضل المرشحين",
        "gaps_h": "## الفجوات والمخاطر",
        "short_h": "## القائمة المختصرة للمقابلة",
        "review_h": "## بحاجة إلى مراجعة - لم يتم التقييم",
        "verify": "يُتحقق منها في المقابلة",
        "ocr_note": "قراءة OCR",
        "yrs": "خبرة {y:g} سنة",
        "gap_missing": "**{label}:** غير مثبت لدى {c} من {n} مرشحًا تم تقييمهم.",
        "gap_unverified": "**حقول غير قابلة للتحقق:** {c} مرشح لديه متطلبات تعذر التحقق منها من "
        "السيرة الذاتية (مستبعدة من مقام الدرجة).",
        "gap_soft": "**غير قابلة للتقييم من السيرة الذاتية:** {items} - تُقيَّم في المقابلة.",
        "gap_hardcap": "**نقص في الحد الأدنى الإلزامي:** {c} مرشح لا يحقق متطلبًا إلزاميًا "
        "(تم تقييد الدرجة).",
        "short_one": "**{name} (السيرة #{no})** في المرتبة الأولى بنسبة {score}% ({level_word})، "
        "محققًا {met} من {tot} من المتطلبات القابلة للتقييم{yrs_part}.{verify_part}",
        "short_tie": "**{names}** يتعادلون بنسبة {score}% ({level_word}) — يُعاملون كمرشحين "
        "متساوين؛ ترتيب الجدول كسرُ تعادلٍ حتمي وليس أفضلية.{verify_part}",
        "short_more": " **{name} (السيرة #{no})** يليه بنسبة {score}% مع إثبات {met} من {tot} "
        "من المتطلبات.",
        "short_verify": " يُتحقق في المقابلة من: {items}.",
        "short_none": "تعذر تقييم أي مرشح مقابل هذا الوصف الوظيفي بأدلة كافية. "
        "راجع الدفعة يدويًا أو وضّح الوصف الوظيفي.",
        "short_weak": "**لا يوجد مرشح يصل إلى مستوى ملاءمة يؤهله للمقابلة لهذه الوظيفة** — "
        "أفضل نتيجة هي {name} (السيرة #{no}) بنسبة {score}% (ملاءمة ضعيفة). "
        "لا يُنصح بقائمة مقابلات من هذه الدفعة؛ يُقترح توسيع البحث أو مراجعة المتطلبات.",
        "short_partial": "**{name} (السيرة #{no})** هو الأقرب بنسبة {score}% (ملاءمة جزئية). "
        "لا يُنصح بالمقابلة إلا إذا تعذر توسيع الدفعة، مع التحقق أولًا من "
        "المتطلبات الناقصة: {missing}.",
        "review_row": "- **{name}** (السيرة #{no}) - {reason}.",
        "review_reason_ie": "أدلة قابلة للتحقق غير كافية لتقييم عادل "
        "({pct}% من المتطلبات قابلة للتحقق)",
        "disclaimer": "_دعم لقرار الموارد البشرية — وليس قرارًا نهائيًا للتوظيف._",
        "level_words": {
            "strong": "ملاءمة قوية",
            "good": "ملاءمة جيدة",
            "partial": "ملاءمة جزئية",
            "weak": "ملاءمة ضعيفة",
        },
        "not_scorable": (
            "تعذر تقييم الوصف الوظيفي بشكل موثوق: {reason}\n\n"
            "يرجى تقديم وصف وظيفي بمتطلبات محددة (مهارات، خبرة، تعليم) "
            "وسأقدم تقرير تقييم ملاءمة كاملًا."
        ),
    },
}


def _bar(score: int) -> str:
    filled = max(0, min(10, int(round(score / 10.0))))
    return "`" + "█" * filled + "░" * (10 - filled) + "`"


def _cells(items: list[str], cap: int, none_word: str) -> str:
    seen = set()
    deduped = []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            deduped.append(i)
    items = deduped
    if not items:
        return none_word
    out = items[:cap]
    if len(items) > cap:
        out.append("…")
    return ", ".join(out)


MAX_TABLE_ROWS = 50  # completeness-first policy: show ALL scored candidates
# up to this ceiling, then Top-50 with a hidden-rows note


def top_n_for(total_scored: int, top_n: int | None) -> int:
    if isinstance(top_n, int) and top_n > 0:
        return min(top_n, total_scored)
    return min(total_scored, MAX_TABLE_ROWS)


def render_not_scorable(reason: str, lang: str = "en") -> str:
    t = T.get(lang, T["en"])
    return t["not_scorable"].format(reason=reason or "no concrete requirements found")


def render(
    rubric: dict[str, Any],
    ranked: list[dict[str, Any]],
    review_lane: list[dict[str, Any]],
    lang: str = "en",
    top_n: int | None = None,
    dup_count: int = 0,
) -> str:
    """ranked: engine.rank() output augmented with display_no/name/years/
    field_coverage. review_lane: [{name, display_no, kind: needs_review|
    insufficient, flags, verifiable_fraction}]."""
    t = T.get(lang, T["en"])
    lv = LEVELS.get(lang, LEVELS["en"])
    lines: list[str] = []
    n = len(ranked)

    if not ranked:
        lines.append(t["title"])
        lines.append("")
        lines.append(t["short_none"])
        lines.extend(_review_section(review_lane, t, lang))
        lines.extend(["", t["disclaimer"]])
        return "\n".join(lines)

    top = ranked[0]
    top_tied = [r for r in ranked if r["score"] == top["score"]]
    lines.append(t["title"])
    if len(top_tied) > 1:
        names = " · ".join(f"**{r['name']}** (CV #{r['display_no']})" for r in top_tied[:4])
        lines.append(
            t["summary_tie"].format(
                role=rubric.get("role_title", ""),
                n=n,
                k=len(top_tied),
                top_score=top["score"],
                top_level=lv[top["level"]],
                names=names,
            )
        )
    else:
        lines.append(
            t["summary"].format(
                role=rubric.get("role_title", ""),
                n=n,
                top_name=top["name"],
                top_score=top["score"],
                top_level=lv[top["level"]],
            )
        )
    lines.append(t["kpi"].format(scored=n, review=len(review_lane), dups=dup_count))
    lines.append("")
    lines.append(t["table_h"])
    shown = top_n_for(n, top_n)
    if shown < n:
        lines.append(t["caption"].format(shown=shown, n=n))
    lines.append("")
    lines.append(t["cols"])
    lines.append(t["sep"])
    any_tie = False
    for r in ranked[:shown]:
        # partial credits are honestly marked (≈), never presented as full matches
        matched = _cells(
            [
                ("≈ " + m["label"]) if m["verdict"] == "partial" else m["label"]
                for m in r.get("matched", [])
            ],
            6,
            t["none"],
        )
        missing = _cells(r.get("missing_required", []), 4, t["none"])
        rk = f"{r['rank']}=" if r.get("tied") else f"{r['rank']}"
        any_tie = any_tie or bool(r.get("tied"))
        lines.append(
            f"| {rk} | {r['display_no']} | **{r['name']}** | "
            f"{_bar(r['score'])} **{r['score']}%** | {lv[r['level']]} | "
            f"{matched} | {missing} |"
        )
    if shown < n:
        lines.append("")
        lines.append(t["hidden_note"])
    lines.append("")
    lines.append(t["bands"])
    if any_tie:
        lines.append(t["tie_note"])
    lines.append("")
    lines.append("---")

    # Evidence - top candidates (citations come from verified evidence only)
    lines.append(t["evidence_h"])
    for r in ranked[: min(3, shown)]:
        bits: list[str] = []
        pages: list[str] = []
        needs_verify = False
        for m in r.get("matched", [])[:6]:
            # keep the same ≈ partial marker the table uses, so the Evidence
            # section never presents a partially-met item as a full match.
            bits.append(("≈ " + m["label"]) if m.get("verdict") == "partial" else m["label"])
            ev = m.get("evidence") or {}
            if ev.get("page"):
                pg = f"p.{ev['page']}"
                if ev.get("source") in ("ocr", "vlm"):
                    needs_verify = True
                if pg not in pages:
                    pages.append(pg)
        yrs = r.get("years")
        yrs_txt = (t["yrs"].format(y=yrs) + ". ") if isinstance(yrs, int | float) else ""
        cite = (
            (
                " ("
                + ", ".join(pages[:3])
                + (f"; {t['ocr_note']} - {t['verify']}" if needs_verify else "")
                + ")."
            )
            if pages
            else "."
        )
        lines.append(f"**{r['name']} (CV #{r['display_no']})** - {yrs_txt}{', '.join(bits)}{cite}")
    lines.append("")

    # Gaps / risks - computed, not narrated
    lines.append(t["gaps_h"])
    gap_counts: dict[str, int] = {}
    unverified_c = 0
    hardcap_c = 0
    for r in ranked:
        for lab in r.get("missing_required", []):
            gap_counts[lab] = gap_counts.get(lab, 0) + 1
        if r.get("unverified"):
            unverified_c += 1
        if any(c.get("cap") == "hard_minimum_missing" for c in r.get("caps", [])):
            hardcap_c += 1
    for lab, c in sorted(gap_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
        lines.append("*   " + t["gap_missing"].format(label=lab, c=c, n=n))
    if hardcap_c:
        lines.append("*   " + t["gap_hardcap"].format(c=hardcap_c))
    if unverified_c:
        lines.append("*   " + t["gap_unverified"].format(c=unverified_c))
    soft = [r["label"] for r in rubric.get("requirements", []) if r["rtype"] == "soft"]
    if soft:
        lines.append("*   " + t["gap_soft"].format(items=", ".join(soft[:3])))
    lines.append("")

    # Interview shortlist - template sentences from data. The wording is
    # LEVEL-AWARE: a Weak-Fit pool is never presented as an interview
    # recommendation, and a top-score tie is presented as a tie.
    lines.append(t["short_h"])
    first = ranked[0]
    yrs = first.get("years")
    yrs_part = (
        ("، " if lang == "ar" else ", ") + t["yrs"].format(y=yrs)
        if isinstance(yrs, int | float)
        else ""
    )
    verify_items = first.get("unverified", [])[:3]
    verify_part = t["short_verify"].format(items=", ".join(verify_items)) if verify_items else ""
    if first["level"] == "weak":
        short = t["short_weak"].format(
            name=first["name"], no=first["display_no"], score=first["score"]
        )
    elif first["level"] == "partial":
        short = t["short_partial"].format(
            name=first["name"],
            no=first["display_no"],
            score=first["score"],
            missing=", ".join(first.get("missing_required", [])[:3]) or t["none"],
        )
    elif len(top_tied) > 1:
        names = ", ".join(f"{r['name']} (CV #{r['display_no']})" for r in top_tied[:4])
        short = t["short_tie"].format(
            names=names,
            score=first["score"],
            level_word=t["level_words"][first["level"]],
            verify_part=verify_part,
        )
    else:
        short = t["short_one"].format(
            name=first["name"],
            no=first["display_no"],
            score=first["score"],
            level_word=t["level_words"][first["level"]],
            met=first.get("req_met_count", 0),
            tot=first.get("total_assessable", 0),
            yrs_part=yrs_part,
            verify_part=verify_part,
        )
    if first["level"] in ("strong", "good"):
        for r in ranked[1 : min(3, shown)]:
            if r["level"] in ("strong", "good") and r["score"] != first["score"]:
                short += t["short_more"].format(
                    name=r["name"],
                    no=r["display_no"],
                    score=r["score"],
                    met=r.get("req_met_count", 0),
                    tot=r.get("total_assessable", 0),
                )
    lines.append(short)

    lines.extend(_review_section(review_lane, t, lang))
    lines.extend(["", t["disclaimer"]])
    return "\n".join(lines)


def _review_section(
    review_lane: list[dict[str, Any]], t: dict[str, Any], lang: str = "en"
) -> list[str]:
    if not review_lane:
        return []
    out = ["", t["review_h"]]
    for c in review_lane[:12]:
        if c.get("kind") == "insufficient":
            reason = t["review_reason_ie"].format(
                pct=int(100 * (c.get("verifiable_fraction") or 0))
            )
        else:
            # honest, code-mapped phrasing — raw flag codes never reach the
            # customer, and 'could not be read' is reserved for unreadable files
            reason = describe_reasons(c.get("flags") or [], lang)
        out.append(
            t["review_row"].format(
                name=c.get("name") or "(name not detected)", no=c.get("display_no"), reason=reason
            )
        )
    return out


def _evidenced_term(m: dict[str, Any]) -> str:
    """The CONCRETE skill/degree the candidate was actually evidenced with, so a
    follow-up answer cites the real evidence — never the requirement's
    alternatives-label. A requirement labelled 'Python/Java' that a candidate
    met by listing only Python must be reported as 'Python', so the model can't
    claim the candidate 'has both Python and Java'."""
    detail, label = (m.get("detail") or ""), (m.get("label") or "")
    marker = "evidenced: "
    if marker in detail:  # 'evidenced: Python' / 'degree evidenced: BSc CS'
        term = detail.split(marker, 1)[1].strip()
        if term:
            return term
    return label


def facts_appendix(
    rubric: dict[str, Any], ranked: list[dict[str, Any]], review_lane: list[dict[str, Any]]
) -> str:
    """Compact authoritative block injected into follow-up turns so the model's
    answers stay consistent with the deterministic scores."""
    lines = [
        "===== LAST FIT SCORING (deterministic engine - AUTHORITATIVE; never "
        "recompute these scores) =====",
        f"Role: {rubric.get('role_title', '')}",
        "When explaining a match, cite ONLY the specific evidenced item listed "
        "below; never claim a candidate has a skill that is not in their facts.",
    ]
    if ranked and all(r["level"] == "weak" for r in ranked):
        lines.append(
            "NOTE: EVERY scored candidate is a WEAK FIT for this role - if asked "
            "for a shortlist or interview recommendation, say clearly that no "
            "candidate meets the bar; never present the least-weak as a "
            "recommendation."
        )
    for r in ranked:
        rk = f"{r['rank']}=" if r.get("tied") else f"{r['rank']}"
        matched = (
            ", ".join(
                (("~" if m.get("verdict") == "partial" else "") + _evidenced_term(m))
                for m in r.get("matched", [])[:6]
            )
            or "none"
        )
        lines.append(
            f"{rk}. {r['name']} (CV #{r['display_no']}) - {r['score']}% "
            f"{r['level']}{' (TIED score)' if r.get('tied') else ''}; "
            f"evidenced: {matched}; "
            f"missing: {', '.join(r.get('missing_required', [])[:4]) or 'none'}"
        )
    for c in review_lane:
        lines.append(f"NOT SCORED: {c.get('name')} (CV #{c.get('display_no')}) - {c.get('kind')}")
    return "\n".join(lines)
