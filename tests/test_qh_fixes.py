"""QH quality-hardening regression tests — one test per confirmed defect class.

All offline (no LLM / embedder): semantic paths receive empty vectors and the
literal-deterministic behavior is what's asserted.
"""

import datetime as dt

from app.pipeline import anchors, dedup
from app.pipeline.extract import error_profile, verify_term
from app.reasons import describe
from app.scoring import engine, report
from app.scoring.match import (
    _degree_rank,
    _domain_terms,
    _edu_field_terms,
    _match_education,
    _match_years,
)

THIS_YEAR = dt.date.today().year


# ---------------------------------------------------------------- F2: degree word-boundary
def test_diploma_is_not_a_masters():
    assert _degree_rank("Diploma in Culinary Arts") == 0  # was 2 (master!)
    assert _degree_rank("Diploma") == 0
    assert _degree_rank("BA History") == 1
    assert _degree_rank("MBA, Finance") == 2
    assert _degree_rank("Alabama Institute certificate") is None  # 'ba' inside a word
    assert _degree_rank("Ph.D in Physics") == 3


# ---------------------------------------------------------------- F3: education field
def _edu_profile(degree, field=""):
    return {
        "education": [
            {
                "degree": degree,
                "field": field,
                "institution": "U",
                "evidence": {"page": 1, "quote": degree},
            }
        ],
        "total_years_experience": {"value": 7.0},
        "_pages": [],
    }


CS_REQ = {
    "degree_level": "bachelor",
    "equivalent_experience_ok": False,
    "text": "Bachelor's degree in Computer Science or a closely related field",
    "keywords": ["Bachelor's degree in Computer Science", "Computer Science", "CS"],
}


def test_education_field_terms_extracted():
    terms = _edu_field_terms(CS_REQ)
    assert any("computer science" == t.lower() for t in terms)


def test_wrong_field_degree_is_partial_not_met():
    v = _match_education(_edu_profile("Bachelor of Arts in History"), CS_REQ)
    assert v["verdict"] == "partial"
    assert "field" in v["detail"].lower()


def test_right_field_degree_is_met():
    v = _match_education(_edu_profile("BSc Computer Science"), CS_REQ)
    assert v["verdict"] == "met"


def test_level_only_requirement_unchanged():
    req = {
        "degree_level": "bachelor",
        "equivalent_experience_ok": False,
        "text": "Bachelor's degree",
        "keywords": ["Bachelor's degree"],
    }
    v = _match_education(_edu_profile("Bachelor of Arts in History"), req)
    assert v["verdict"] == "met"  # no field named by the JD -> level satisfies


def test_diploma_below_bachelor_is_missing():
    v = _match_education(_edu_profile("Diploma"), CS_REQ)
    assert v["verdict"] == "missing"


# ---------------------------------------------------------------- F4: domain-qualified years
YRS_REQ = {
    "min_years": 4.0,
    "text": "4+ years of professional software development experience",
    "keywords": ["professional software development experience"],
}


def _exp_profile(total, titles):
    return {
        "total_years_experience": {"value": total},
        "experiences": [{"title": t, "start": "2019", "end": "Present"} for t in titles],
    }


def test_domain_terms_extracted():
    assert any("software development" in t.lower() for t in _domain_terms(YRS_REQ))


def test_sales_tenure_does_not_meet_software_years():
    v = _match_years(_exp_profile(11.0, ["Sales Manager", "Sales Executive"]), YRS_REQ)
    assert v["verdict"] == "partial"  # honest partial, never full met
    assert "total" in v["detail"]


def test_software_titles_fully_meet_software_years():
    v = _match_years(_exp_profile(7.0, ["Software Developer"]), YRS_REQ)
    assert v["verdict"] == "met"


def test_unqualified_tenure_requirement_unchanged():
    req = {"min_years": 5.0, "text": "5+ years of experience", "keywords": []}
    v = _match_years(_exp_profile(8.0, ["Anything"]), req)
    assert v["verdict"] == "met"


# ---------------------------------------------------------------- F12: plural-tolerant terms
def test_verify_term_plural_tolerance():
    pages = [{"page": 1, "text": "Participated in code reviews and testing."}]
    assert verify_term("Code review", pages) is not None
    assert verify_term("code reviews", [{"page": 1, "text": "weekly code review"}]) is not None
    # boundaries stay strict
    assert verify_term("Java", [{"page": 1, "text": "JavaScript only"}]) is None


# ---------------------------------------------------------------- F10: month-aware years
def test_month_boundary_role_change_is_contiguous():
    ty, n = anchors.compute_total_years(
        [
            {"start": "June 2015", "end": "December 2018"},
            {"start": "January 2019", "end": "Present"},
        ]
    )
    assert n == 2
    expected = ((THIS_YEAR * 12 + dt.date.today().month) - (2015 * 12 + 6) + 1) / 12.0
    assert abs(ty - round(expected, 1)) < 0.15  # ~11.2 in July 2026, not 10


def test_year_only_math_unchanged():
    ty, _ = anchors.compute_total_years([{"start": "2014", "end": "2019"}])
    assert ty == 5.0
    ty2, _ = anchors.compute_total_years(
        [{"start": "2010", "end": "2014"}, {"start": "2018", "end": "2022"}]
    )
    assert ty2 == 8.0


def test_month_overlap_union():
    ty, _ = anchors.compute_total_years(
        [
            {"start": "March 2022", "end": "Present"},
            {"start": "September 2021", "end": "February 2022"},
        ]
    )
    expected = ((THIS_YEAR * 12 + dt.date.today().month) - (2021 * 12 + 9) + 1) / 12.0
    assert abs(ty - round(expected, 1)) < 0.15  # ~4.9 — below a 5-year floor


# ---------------------------------------------------------------- F11: stated-vs-dated note
def test_stated_years_claim():
    assert anchors.stated_years_claim("Senior QA Engineer with 8 years of experience") == 8.0
    assert anchors.stated_years_claim("no numbers here") is None


# ---------------------------------------------------------------- F7: conflict vs duplicate
def _dprof(cid, no, name, email, text):
    return {
        "candidate_id": cid,
        "display_no": no,
        "identity": {"full_name": {"value": name}, "emails": [email], "phones": []},
        "skills": [],
        "extraction": {"status": "OK", "reasons": []},
        "_pages": [{"page": 1, "text": text}],
    }


def test_conflicting_versions_flagged_not_collapsed():
    a = _dprof(
        "c_001",
        1,
        "Rami Khoury",
        "rami@acmecorp-x.com",
        "Network engineer Cisco routing firewalls VPN monitoring enterprise",
    )
    b = _dprof(
        "c_002",
        2,
        "Rami Khoury",
        "rami@acmecorp-x.com",
        "Senior accountant IFRS reconciliation QuickBooks financial reporting close",
    )
    out = dedup.mark_duplicates([a, b])
    assert out[1].get("duplicate_of") is None
    assert "CONFLICTING_VERSIONS" in out[1]["extraction"]["reasons"]
    assert out[0].get("verification_notes")


def test_true_duplicates_still_collapse():
    text = "Software engineer Python SQL REST APIs Git Docker backend development services"
    a = _dprof("c_001", 1, "Yousef Nabil", "y@acmecorp-x.com", text)
    b = _dprof("c_002", 2, "Yousef Nabil", "y@acmecorp-x.com", text)
    out = dedup.mark_duplicates([a, b])
    assert out[1].get("duplicate_of") == "c_001"


# AV-DEFECT-1: the PLACEHOLDER-contact fallback (same name + overlapping skills,
# no real shared key) must also distinguish duplicate from conflict by CONTENT.
def _sales(cid, no, text):
    p = _dprof(
        cid, no, "Yara Sethi", "candidate016@example.com", text
    )  # example.com = placeholder -> no real key
    p["skills"] = [
        {"name": s} for s in ["Prospecting", "Negotiation", "B2B Sales", "Quota Attainment"]
    ]
    return p


def test_fallback_same_name_different_content_is_conflict_not_duplicate():
    a = _sales(
        "c_001",
        1,
        "Sales Executive Larkspur Foods 2025 present closed 18M SAR "
        "pipeline 60 qualified quota 120 percent Vertex Labs Auralink seven years",
    )
    b = _sales(
        "c_002",
        2,
        "Sales Executive Ironwood Manufacturing 2022 present Cobalt Industries "
        "twelve years managed 120 key accounts closed 3M SAR led team",
    )
    out = dedup.mark_duplicates([a, b])
    assert out[1].get("duplicate_of") is None  # NOT excluded as a duplicate
    assert "CONFLICTING_VERSIONS" in out[1]["extraction"]["reasons"]  # flagged for human review
    assert out[0].get("verification_notes")  # kept profile carries a note


def test_fallback_same_name_identical_content_still_duplicate():
    text = (
        "Sales Executive Larkspur Foods 2025 present closed 18M SAR pipeline 60 qualified "
        "quota 120 percent Vertex Labs Auralink Cedarfield Group seven years measurable impact"
    )
    a = _sales("c_001", 1, text)
    b = _sales("c_002", 2, text)  # a genuine re-upload of the same document
    out = dedup.mark_duplicates([a, b])
    assert out[1].get("duplicate_of") == "c_001"


# ---------------------------------------------------------------- F9: benefit-from-missing cap
RUB = {
    "requirements": [
        {"id": "r1", "kind": "required", "rtype": "experience_years", "label": "5+ yrs"},
        {"id": "r2", "kind": "required", "rtype": "skill", "label": "Recruitment"},
        {"id": "r3", "kind": "required", "rtype": "skill", "label": "HRIS"},
    ]
}


def test_required_unverified_capped_below_strong():
    va = [
        {"req_id": "r1", "verdict": "unverified"},
        {"req_id": "r2", "verdict": "met"},
        {"req_id": "r3", "verdict": "met"},
    ]
    res = engine.score_candidate(RUB, va)
    assert res["scored"] and res["score"] <= engine.CAP_UNVERIFIED_REQ
    assert res["level"] != "strong"
    assert any(c["cap"] == "required_unverified" for c in res["caps"])


# ---------------------------------------------------------------- F6: competition tie ranks
def _mk(no, score, met=3):
    return {
        "candidate_id": f"c_{no:03d}",
        "display_no": no,
        "name": f"P{no}",
        "years": 5.0,
        "field_coverage": 0.8,
        "scored": True,
        "score": score,
        "level": engine.level_for(score),
        "matched": [],
        "missing_required": [],
        "unverified": [],
        "caps": [],
        "req_met_count": met,
        "included_requirements": 3,
        "total_assessable": 3,
        "verifiable_fraction": 1.0,
    }


def test_tied_scores_share_rank():
    ranked = engine.rank([_mk(1, 84), _mk(2, 84), _mk(3, 70)])
    assert [r["rank"] for r in ranked] == [1, 1, 3]
    assert ranked[0]["tied"] and ranked[1]["tied"] and not ranked[2]["tied"]


def test_report_declares_tie_and_no_unique_top():
    ranked = engine.rank([_mk(1, 84), _mk(2, 84)])
    md = report.render({"role_title": "X", "requirements": []}, ranked, [], "en")
    assert "tie at the top" in md
    assert "| 1= |" in md
    assert "Top match: **P1**" not in md


# ---------------------------------------------------------------- F5: weak pool shortlist
def test_weak_pool_not_recommended_for_interview():
    ranked = engine.rank([_mk(1, 22), _mk(2, 15)])
    md = report.render({"role_title": "X", "requirements": []}, ranked, [], "en")
    section = md.split("## Interview Shortlist")[1]
    assert "No candidate reaches an interview-ready fit" in section
    assert "ranks first" not in section


# ---------------------------------------------------------------- F8: honest review phrasing
def test_reason_phrases_map():
    assert "readable" in describe(["SPARSE_TEXT"], "en")
    assert "could not be read" in describe(["UNREADABLE_FILE"], "en")
    assert "may not be a CV" in describe(["NO_IDENTITY_ANCHOR"], "en")
    assert "SPARSE_TEXT" not in describe(["SPARSE_TEXT"], "en")  # no raw codes


def test_review_section_uses_phrases_and_kpi_line_renders():
    ranked = engine.rank([_mk(1, 90)])
    lane = [{"name": "Hani", "display_no": 7, "kind": "needs_review", "flags": ["SPARSE_TEXT"]}]
    md = report.render({"role_title": "X", "requirements": []}, ranked, lane, "en", dup_count=2)
    assert "very little content" in md and "SPARSE_TEXT" not in md
    assert (
        "Scored: 1" in md
        and "Needs review (not scored): 1" in md
        and "Duplicates excluded: 2" in md
    )


# ---------------------------------------------------------------- F1: crash isolation
def test_error_profile_is_reviewable():
    p = error_profile({"document": "x.pdf", "page_range": [0, 0]}, 3)
    assert p["extraction"]["status"] == "NEEDS_REVIEW"
    assert p["extraction"]["reasons"] == ["UNREADABLE_FILE"]
    assert p["candidate_id"] == "c_003"


# AV-DEFECT-2: follow-up facts must cite the SPECIFIC evidenced skill, never the
# requirement's alternatives-label (model claimed "Python and Java" when only
# Python was evidenced).
def test_facts_appendix_cites_evidenced_skill_not_label():
    rub = {"role_title": "Software Engineer", "requirements": []}
    ranked = [
        {
            "rank": 1,
            "display_no": 3,
            "name": "Mason Vance",
            "score": 69,
            "level": "good",
            "tied": False,
            "req_met_count": 5,
            "total_assessable": 10,
            "missing_required": ["Design backend services"],
            "matched": [
                {"label": "Python/Java", "verdict": "met", "detail": "evidenced: Python"},
                {"label": "SQL databases/SQL", "verdict": "met", "detail": "evidenced: SQL"},
                {
                    "label": "Bachelor degree",
                    "verdict": "met",
                    "detail": "degree evidenced: BA in Computer Science",
                },
            ],
        }
    ]
    txt = report.facts_appendix(rub, ranked, [])
    assert "evidenced: Python," in txt  # the concrete skill
    assert "Python/Java" not in txt  # NOT the alternatives-label
    assert "BA in Computer Science" in txt  # concrete degree, not "Bachelor degree"


# AV2-DEFECT-1: a certification must require LITERAL evidence — never a semantic
# role-proximity partial ("CPA" must not partial-match the title "Accountant").
def test_certification_never_semantic_partial():
    from app.scoring.match import match_candidate

    rub = {
        "requirements": [
            {
                "id": "r1",
                "kind": "preferred",
                "rtype": "certification",
                "label": "CPA",
                "text": "CPA certification",
                "keywords": ["CPA", "Certified Public Accountant"],
            }
        ]
    }
    prof = {
        "skills": [{"name": "Accountant", "source": "stated"}],
        "experiences": [{"title": "Accountant"}],
        "education": [],
        "certifications": [],
        "_pages": [
            {"page": 1, "text": "Senior Accountant with reconciliation and payroll experience"}
        ],
    }
    # strong semantic vectors that WOULD partial-match Accountant~CPA if allowed
    req_vecs = [[[1.0, 0.0]]]
    phrase_vecs = {"accountant": [0.99, 0.14]}  # cos ~0.99 with the req vector
    v = match_candidate(prof, rub, req_vecs, phrase_vecs)[0]
    assert v["verdict"] != "partial"  # NOT a semantic partial
    assert v["verdict"] in ("unverified", "missing")


def test_certification_literal_still_matches():
    from app.scoring.match import match_candidate

    rub = {
        "requirements": [
            {
                "id": "r1",
                "kind": "preferred",
                "rtype": "certification",
                "label": "CPA",
                "text": "CPA",
                "keywords": ["CPA"],
            }
        ]
    }
    prof = {
        "skills": [],
        "experiences": [],
        "education": [],
        "certifications": [{"name": "CPA"}],
        "_pages": [{"page": 1, "text": "Certifications: CPA (2018)"}],
    }
    v = match_candidate(prof, rub, [None], {})[0]
    assert v["verdict"] == "met"  # literal CPA still matches


# AV2-DEFECT-2: display names normalized to consistent case (JESSICA CLAIRE -> Jessica Claire)
def test_name_case_normalization():
    from app.pipeline.extract import _normalize_name_case

    assert _normalize_name_case("JESSICA CLAIRE") == "Jessica Claire"
    assert _normalize_name_case("jessica claire") == "Jessica Claire"
    assert _normalize_name_case("ahmed al-otaibi") == "Ahmed Al-Otaibi"
    assert _normalize_name_case("Jessica Claire") == "Jessica Claire"  # already fine
    # mixed-case words are NOT mangled
    assert _normalize_name_case("Ronan McDonald") == "Ronan McDonald"
    assert _normalize_name_case("Sarah O'Brien") == "Sarah O'Brien"
