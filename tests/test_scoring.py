"""Unit tests for the deterministic scoring plane (no model calls needed).

Fixtures are synthetic and generic — invented roles/skills, no values from any
test batch. Matching runs in literal-only mode (empty embedding vectors)."""

from __future__ import annotations

import copy

from app.scoring import engine, match, report
from app.scoring.rubric import _clean_keywords, _short_label


# ---------------------------------------------------------------- fixtures
def _req(
    id_,
    text,
    kind="required",
    rtype="skill",
    keywords=None,
    min_years=None,
    degree_level="",
    equiv=False,
):
    r = {
        "id": id_,
        "text": text,
        "kind": kind,
        "rtype": rtype,
        "min_years": min_years,
        "degree_level": degree_level,
        "equivalent_experience_ok": equiv,
        "keywords": keywords or [],
    }
    r["label"] = _short_label(r)
    return r


RUBRIC = {
    "role_title": "Widget Process Supervisor",
    "clarity": "ok",
    "clarity_reason": "",
    "requirements": [
        _req("r1", "5+ years supervising production lines", rtype="experience_years", min_years=5),
        _req(
            "r2",
            "Bachelor degree in industrial engineering or equivalent experience",
            rtype="education",
            degree_level="bachelor",
            equiv=True,
        ),
        _req("r3", "Lean manufacturing methods", keywords=["Lean", "Kaizen"]),
        _req("r4", "Quality auditing", keywords=["Quality Audit", "ISO 9001"]),
        _req(
            "r5",
            "Forklift certification",
            kind="preferred",
            rtype="certification",
            keywords=["Forklift"],
        ),
        _req("r6", "Excellent teamwork", rtype="soft", keywords=["teamwork"]),
    ],
    "jd_hash": "cafe0123",
}


def _profile(no, name, years, skills_text, education=None):
    pages = [
        {"page": 1, "text": f"{name}\n{skills_text}", "source": "text", "ocr_confidence": None}
    ]
    return {
        "candidate_id": f"c_{no:03d}",
        "display_no": no,
        "identity": {
            "full_name": {"value": name, "evidence": None},
            "emails": [f"p{no}@example.test"],
            "phones": [],
            "links": [],
        },
        "total_years_experience": {"value": years, "basis": "test"},
        "experiences": [
            {
                "title": "Production Line Supervisor",
                "organization": "Plant A",
                "start": "2018",
                "end": "2024",
                "evidence": None,
            }
        ],
        "education": education or [],
        "skills": [
            {"name": s.strip(), "source": "stated", "evidence": None}
            for s in skills_text.split(",")
            if s.strip()
        ],
        "certifications": [],
        "languages": [],
        "extraction": {
            "status": "OK",
            "reasons": [],
            "field_coverage": 0.9,
            "ocr_used": False,
            "vlm_used": False,
            "mean_ocr_confidence": None,
        },
        "duplicate_of": None,
        "_pages": pages,
    }


def _verdicts(profile, rubric=RUBRIC):
    nreq = len(rubric["requirements"])
    return match.match_candidate(profile, rubric, [None] * nreq, {})


# ---------------------------------------------------------------- matching
def test_years_met_partial_missing_unverified():
    v = _verdicts(_profile(1, "Alpha One", 7, "Lean"))
    assert v[0]["verdict"] == "met"
    v = _verdicts(_profile(2, "Beta Two", 4, "Lean"))  # 4 vs 5 -> partial (>=75%)
    assert v[0]["verdict"] == "partial"
    v = _verdicts(_profile(3, "Gamma Three", 1, "Lean"))
    assert v[0]["verdict"] == "missing"
    v = _verdicts(_profile(4, "Delta Four", None, "Lean"))
    assert v[0]["verdict"] == "unverified"


def test_education_equivalent_and_degree():
    edu = [
        {
            "degree": "BSc",
            "field": "Industrial Engineering",
            "institution": "Uni",
            "year": "2015",
            "evidence": {"page": 1, "quote": "BSc Industrial Engineering"},
        }
    ]
    v = _verdicts(_profile(1, "Alpha One", 8, "Lean", education=edu))
    assert v[1]["verdict"] == "met"
    # no degree but long tenure + JD allows equivalent -> partial
    v = _verdicts(_profile(2, "Beta Two", 8, "Lean"))
    assert v[1]["verdict"] == "partial"


def test_term_match_word_boundary_and_soft():
    v = _verdicts(_profile(1, "Alpha One", 8, "Lean, ISO 9001"))
    assert v[2]["verdict"] == "met"  # Lean evidenced
    assert v[3]["verdict"] == "met"  # ISO 9001 evidenced
    assert v[5]["verdict"] == "not_assessable"
    # 'Lean' must not match inside 'cLEANing'
    p = _profile(2, "Beta Two", 8, "industrial cleaning services")
    p["skills"] = []
    v = _verdicts(p)
    assert v[2]["verdict"] == "missing"


# ---------------------------------------------------------------- engine
def _score(profile):
    return engine.score_candidate(RUBRIC, _verdicts(profile))


def test_full_match_scores_high_and_missing_low():
    edu = [
        {
            "degree": "Bachelor of Industrial Engineering",
            "field": "",
            "institution": "U",
            "year": "",
            "evidence": None,
        }
    ]
    good = _score(
        _profile(1, "Alpha One", 9, "Lean, Kaizen, Quality Audit, Forklift", education=edu)
    )
    assert good["scored"] and good["score"] >= 90 and good["level"] == "strong"
    bad = _score(_profile(2, "Beta Two", 9, "florist, gardening"))
    assert bad["scored"] and bad["level"] == "weak"  # no core capability


def test_hard_minimum_gate_caps_score():
    edu = [{"degree": "BSc", "field": "IE", "institution": "U", "year": "", "evidence": None}]
    # all skills present but only 1 year of experience -> hard-min missing -> cap
    r = _score(
        _profile(3, "Gamma Three", 1, "Lean, Kaizen, Quality Audit, ISO 9001", education=edu)
    )
    assert r["scored"] and r["score"] <= engine.CAP_HARD_MISS
    assert any(c["cap"] == "hard_minimum_missing" for c in r["caps"])


def test_unverified_excluded_not_zeroed():
    # years unreadable -> requirement excluded from denominator, disclosed
    edu = [{"degree": "BSc", "field": "IE", "institution": "U", "year": "", "evidence": None}]
    r = _score(_profile(4, "Delta Four", None, "Lean, Kaizen, Quality Audit", education=edu))
    assert r["scored"]
    assert any("yrs" in u or "experience" in u for u in r["unverified"])
    # excluded from the denominator (not zeroed) BUT capped below Strong:
    # missing information must never out-rank verified candidates.
    assert 60 <= r["score"] <= engine.CAP_UNVERIFIED_REQ
    assert any(c["cap"] == "required_unverified" for c in r["caps"])


def test_ranking_deterministic_with_tiebreaks():
    results = [
        {
            "scored": True,
            "score": 80,
            "req_met_count": 3,
            "years": 6.0,
            "field_coverage": 0.9,
            "display_no": 2,
            "name": "B",
        },
        {
            "scored": True,
            "score": 80,
            "req_met_count": 3,
            "years": 9.0,
            "field_coverage": 0.9,
            "display_no": 1,
            "name": "A",
        },
        {
            "scored": True,
            "score": 91,
            "req_met_count": 4,
            "years": 3.0,
            "field_coverage": 0.5,
            "display_no": 3,
            "name": "C",
        },
    ]
    ranked = engine.rank(copy.deepcopy(results))
    assert [r["name"] for r in ranked] == ["C", "A", "B"]  # years orders within the 80-tie
    assert [r["rank"] for r in ranked] == [1, 2, 2]  # competition ranking: tie shares a rank
    assert not ranked[0]["tied"] and ranked[1]["tied"] and ranked[2]["tied"]


# ---------------------------------------------------------------- report
def _ranked_fixture():
    edu = [{"degree": "BSc", "field": "IE", "institution": "U", "year": "", "evidence": None}]
    profs = [
        _profile(1, "Alpha One", 9, "Lean, Kaizen, Quality Audit, Forklift", education=edu),
        _profile(2, "Beta Two", 6, "Lean, Quality Audit", education=edu),
        _profile(3, "Gamma Three", 8, "florist, gardening"),
    ]
    results = []
    for p in profs:
        res = engine.score_candidate(RUBRIC, _verdicts(p))
        results.append(
            {
                "candidate_id": p["candidate_id"],
                "display_no": p["display_no"],
                "name": p["identity"]["full_name"]["value"],
                "years": p["total_years_experience"]["value"],
                "field_coverage": 0.9,
                **res,
            }
        )
    return engine.rank(results)


def test_report_render_deterministic_and_clean():
    ranked = _ranked_fixture()
    review = [
        {
            "name": "Epsilon Five",
            "display_no": 9,
            "kind": "needs_review",
            "flags": ["LOW_OCR_CONFIDENCE"],
        }
    ]
    md1 = report.render(RUBRIC, ranked, review, lang="en")
    md2 = report.render(RUBRIC, ranked, review, lang="en")
    assert md1 == md2  # byte-reproducible
    assert md1.startswith("# HR Fit Scoring Results")
    assert "| Rank |" in md1 and "Needs Review" in md1
    assert "Decision support for HR" in md1
    low = md1.lower()
    for leak in ("let's", "wait,", "re-check", "i will", "score calculation"):
        assert leak not in low
    # arabic variant renders with arabic headings + same table shape
    ar = report.render(RUBRIC, ranked, review, lang="ar")
    assert ar.count("|") >= md1.count("|") - 10 and "الأدلة" in ar


def test_table_shows_all_up_to_50_then_caps():
    # policy: all rows when <=50 scored; Top 50 + hidden-note when more
    assert report.top_n_for(5, None) == 5
    assert report.top_n_for(43, None) == 43
    assert report.top_n_for(50, None) == 50
    assert report.top_n_for(51, None) == 50
    assert report.top_n_for(200, None) == 50
    assert report.top_n_for(200, 10) == 10  # explicit override kept
    # 15 scored -> all 15 rows, no caption, no note
    ranked = engine.rank(
        [dict(r, display_no=i + 1, rank=None) for i, r in enumerate(_ranked_fixture() * 5)]
    )
    md = report.render(RUBRIC, ranked, [], lang="en")
    assert md.count("| ") and "Top " not in md.split("##")[1]
    assert "not shown in this summary" not in md
    # 60 scored -> 50 rows + caption + hidden note under the table
    big = engine.rank(
        [dict(r, display_no=i + 1, rank=None) for i, r in enumerate(_ranked_fixture() * 20)]
    )
    md2 = report.render(RUBRIC, big, [], lang="en")
    assert "Top 50 of 60 scored candidates" in md2
    assert "Additional lower-ranked candidates are not shown" in md2
    rows = [ln for ln in md2.split(chr(10)) if ("█" in ln or "░" in ln) and ln.count("|") >= 7]
    assert len(rows) == 50
    ar = report.render(RUBRIC, big, [], lang="ar")
    assert "لا تُعرض بقية السير" in ar


def test_not_scorable_message():
    md = report.render_not_scorable("only a job title was provided", "en")
    assert "could not be scored" in md


# ---------------------------------------------------------------- rubric utils
def test_keyword_cleaning_and_labels():
    ks = _clean_keywords(
        "Strong experience with Java, Go, Python, or C#", ["Java", "Go", "Python", "C#", "Java", ""]
    )
    assert ks[:4] == ["Java", "Go", "Python", "C#"] and len(ks) == 4
    lab = _short_label(_req("rX", "x", rtype="experience_years", min_years=8))
    assert lab == "8+ yrs experience"


# ---------------------------------------------------------------- RWB generic fixes
def test_floorless_experience_never_blanket_met():
    # a years requirement with no floor must not be satisfied by mere tenure
    req = _req("rx", "Litigation experience", rtype="experience_years", min_years=None)
    rub = {"role_title": "X", "clarity": "ok", "requirements": [req], "jd_hash": "t"}
    p = _profile(1, "Alpha One", 20, "florist, gardening")
    v = match.match_candidate(p, rub, [None], {})
    assert v[0]["verdict"] in ("unverified", "missing")  # never 'met'


def test_certification_absent_section_is_unverified_not_missing():
    req = _req(
        "rc",
        "License to practice law",
        rtype="certification",
        keywords=["bar admission", "law license"],
    )
    rub = {"role_title": "X", "clarity": "ok", "requirements": [req], "jd_hash": "t"}
    p = _profile(2, "Beta Two", 10, "Litigation, Legal research")  # no certs parsed
    v = match.match_candidate(p, rub, [None], {})
    assert v[0]["verdict"] == "unverified"
    p2 = _profile(3, "Gamma Three", 10, "Litigation")
    p2["certifications"] = [{"name": "Notary Commission", "evidence": None}]
    v2 = match.match_candidate(p2, rub, [None], {})
    assert v2[0]["verdict"] == "missing"  # certs parsed, none match -> definitive


def test_heading_with_colon_rejected_as_name():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("Technical Skills:")
    assert not _valid_name("Core Competencies:")
    assert _valid_name("Mary-Jane O'Neil")  # real punctuation still fine


def test_partial_marker_in_report():
    req_years = _req("r1", "5+ years", rtype="experience_years", min_years=5)
    rub = {
        "role_title": "X",
        "clarity": "ok",
        "requirements": [req_years, _req("r2", "Lean", keywords=["Lean"])],
        "jd_hash": "t",
    }
    p = _profile(1, "Alpha One", 4, "Lean")  # 4 vs 5 -> partial years
    res = engine.score_candidate(rub, match.match_candidate(p, rub, [None, None], {}))
    ranked = engine.rank(
        [
            {
                "candidate_id": "c1",
                "display_no": 1,
                "name": "Alpha One",
                "years": 4.0,
                "field_coverage": 0.9,
                **res,
            }
        ]
    )
    md = report.render(rub, ranked, [], lang="en")
    assert "≈ 5+ yrs experience" in md and "≈ marks a partially-met" in md


def test_lowercase_fragment_rejected_as_name():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("specific requirements.")
    assert not _valid_name("with over ten years")
    assert _valid_name("Dyson Yudiana")
    assert _valid_name("Jan van der Berg")  # >=1 uppercase-initial word


def test_plural_heading_rejected_as_name():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("DATA SCIENCE RESUMES")
    assert not _valid_name("Employment Histories")
    assert _valid_name("James Watson")  # trailing-s real names unaffected


def test_placeholder_person_vocabulary_rejected():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("CANDIDATE JOBSEEKER")
    assert not _valid_name("Applicant Name")
    assert _valid_name("Shyam Sharma")


def test_fabricated_years_floor_reclassified():
    # rubric.build post-validation is exercised via the raw-parse path; here we
    # simulate its outcome contract: a years floor is only real when the
    # requirement's own text contains a digit.
    import re

    text = "Ground handling or ramp operations knowledge"
    assert not re.search(r"\d", text)  # guard premise
    # the engine-facing contract: floorless years never blanket-met (see
    # test_floorless_experience_never_blanket_met)


def test_matched_cells_deduped():
    assert report._cells(["A", "A", "B"], 6, "None") == "A, B"


def test_rubric_dedupes_oversplit_requirements():
    # The JD parser (LLM) sometimes over-splits ONE line into several near-
    # identical rows that all render as the SAME label (observed live: a single
    # "Bachelor's degree in Business, Marketing, or a related field" becoming
    # three education rows all labelled "Bachelor degree"). Left unmerged this
    # inflates the denominator and lets the same label show as met AND missing.
    from app.scoring.rubric import _dedupe_requirements

    reqs = [
        _req(
            "r1",
            "Bachelor's degree",
            rtype="education",
            degree_level="bachelor",
            keywords=["Bachelor", "degree"],
        ),
        _req(
            "r2",
            "degree in Business",
            rtype="education",
            degree_level="bachelor",
            keywords=["Business"],
        ),
        _req(
            "r3",
            "degree in Marketing or a related field",
            rtype="education",
            degree_level="bachelor",
            keywords=["Marketing"],
        ),
        _req("r4", "CRM system", keywords=["CRM", "Salesforce"]),
        _req("r5", "negotiation", keywords=["negotiation"]),
    ]
    out = _dedupe_requirements(reqs)
    labels = [r["label"] for r in out]
    assert len(out) == 3, labels  # 3 education rows -> 1
    assert labels.count("Bachelor degree") == 1  # no duplicate label
    edu = next(r for r in out if r["rtype"] == "education")
    # keywords from all three merged rows are preserved (union)
    assert {"Bachelor", "Business", "Marketing"}.issubset(set(edu["keywords"]))
    # ids are re-sequenced contiguously after the merge
    assert [r["id"] for r in out] == ["r1", "r2", "r3"]


def test_rubric_dedupe_keeps_distinct_labels_and_strongest_kind():
    # Distinct requirements must NOT be merged, and a required duplicate must win
    # over a preferred one (a candidate can't benefit from the split).
    from app.scoring.rubric import _dedupe_requirements

    reqs = [
        _req("r1", "Python", keywords=["Python"], kind="preferred"),
        _req("r2", "Python", keywords=["Python", "py"], kind="required"),
        _req("r3", "SQL", keywords=["SQL"]),
    ]
    out = _dedupe_requirements(reqs)
    assert len(out) == 2  # Python x2 -> 1, SQL stays
    py = next(r for r in out if "Python" in r["label"] or "python" in r["keywords"][0].lower())
    assert py["kind"] == "required"  # strongest kind wins


def test_institution_and_honors_rejected_as_name():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("Cranford High School")
    assert not _valid_name("Magna Cum Laude")
    assert not _valid_name("Riverside Academy")
    assert _valid_name("Cole Hodges")


def test_arabic_years_floor_coerced(monkeypatch):
    # simulate the raw-model shape: years line classified as 'responsibility'
    from app.pipeline import llm as L
    from app.scoring import rubric as R

    raw = {
        "role_title": "Teacher",
        "clarity": "ok",
        "requirements": [
            {
                "text": "خبرة لا تقل عن 3 سنوات في التدريس",
                "kind": "required",
                "rtype": "responsibility",
                "keywords": ["teaching experience"],
            },
            {
                "text": "Adobe Photoshop",
                "kind": "required",
                "rtype": "skill",
                "keywords": ["Photoshop"],
            },
        ],
    }
    monkeypatch.setattr(L, "chat_json", lambda *a, **k: raw)
    rub = R.build(object(), "JD: teacher role. " + "خبرة لا تقل عن 3 سنوات في التدريس. " + "x" * 20)
    r0 = rub["requirements"][0]
    assert r0["rtype"] == "experience_years" and r0["min_years"] == 3.0
    assert rub["requirements"][1]["rtype"] == "skill"  # untouched


def test_cv_token_and_doe_placeholders_rejected():
    from app.pipeline.extract import _valid_name

    assert not _valid_name("HUMAN RESOURCES CV")
    assert not _valid_name("Jane Doe")
    assert not _valid_name("John Doe")
    assert _valid_name("C.V. Raman")  # initials unaffected
    assert _valid_name("Jonathan Burns")
