"""Offline tests for the query layer: reference resolution, query parsing, and
the intents that need no model (count, compare, years-filter, exclude)."""

from app.query import engine, references


class FakeSession:
    def __init__(self, profiles):
        self.profiles = profiles
        self.roster_order = [p["candidate_id"] for p in profiles]
        self.view_state = {
            "current_set": list(self.roster_order),
            "last_ranking": [],
            "last_shortlist": [],
            "last_result": [],
            "history": [],
        }
        self.summary = {}


def mk(no, name, years, skills, status="OK", dup=None):
    return {
        "candidate_id": f"c_{no:03d}",
        "display_no": no,
        "identity": {
            "full_name": {"value": name, "evidence": None},
            "emails": [name.split()[0].lower() + "@x.com"],
            "phones": [],
            "links": [],
        },
        "headline": "",
        "summary": "",
        "total_years_experience": {"value": years, "basis": "x"},
        "experiences": [],
        "education": [],
        "skills": [
            {"name": s, "source": "stated", "evidence": {"page": 1, "quote": s}} for s in skills
        ],
        "certifications": [],
        "languages": [],
        "extraction": {
            "ocr_used": False,
            "mean_ocr_confidence": None,
            "field_coverage": 0.8,
            "status": status,
            "reasons": [] if status == "OK" else ["LOW_OCR_CONFIDENCE"],
        },
        "duplicate_of": dup,
    }


def _session():
    return FakeSession(
        [
            mk(1, "Ahmed Ali", 8, ["Recruitment", "SAP SuccessFactors"]),
            mk(2, "Sara Khan", 11, ["Recruitment", "SAP HCM"]),
            mk(3, "Omar Nasr", 3, ["Payroll"], status="NEEDS_REVIEW"),
        ]
    )


def test_resolve_explicit_and_ordinal():
    s = _session()
    assert references.resolve_candidate_refs("compare candidate 2 and 5", s) == ["c_002"]
    assert references.resolve_candidate_refs("the second candidate", s) == ["c_002"]


def test_resolve_by_name_and_anaphora():
    s = _session()
    assert references.resolve_candidate_refs("compare Ahmed and Sara", s) == ["c_001", "c_002"]
    s.view_state["last_result"] = ["c_002", "c_003"]
    assert references.resolve_candidate_refs("tell me about those candidates", s) == [
        "c_002",
        "c_003",
    ]


def test_year_and_topn_parsing():
    assert references.parse_years_threshold("more than 5 years") == ("gte", 5)
    assert references.parse_years_threshold("less than 5 years") == ("lt", 5)
    assert references.parse_years_threshold("5+ years") == ("gte", 5)
    assert references.parse_top_n("give me the top 3", default=10) == 3


def test_skill_query_and_criteria_extraction():
    assert "sap" in " ".join(engine._extract_skill_terms("Who has SAP experience?")).lower()
    assert (
        "recruitment"
        in " ".join(engine._extract_skill_terms("who is strong in recruitment")).lower()
    )
    crit = engine._criteria_from_query("Recommend the best candidates for HR Specialist")
    assert "hr specialist" in crit["role_text"].lower()


def test_multi_term_skill_and_noise_stripping():
    terms = engine._extract_skill_terms(
        "Who has software engineering, web development, or UX/design experience? Show evidence."
    )
    joined = " ".join(terms)
    assert "software engineering" in joined and "web development" in joined
    assert "ux" in joined or "design" in joined
    assert "evidence" not in joined  # instruction noise stripped
    crit = engine._criteria_from_query(
        "Rank the top 5 candidates for a Software Engineer role. Explain reasons and missing requirements."
    )
    assert "software engineer" in crit["role_text"]
    assert "explain" not in crit["role_text"] and "missing" not in crit["role_text"]


def test_what_about_followup_is_who_has():
    s = _session()
    r = engine.answer(s, "What about the UX candidates?", settings=None)
    assert r["intent"] == "who_has"


def test_count_intent():
    s = _session()
    r = engine.answer(s, "how many candidates are in this file?", settings=None)
    assert r["intent"] == "count"
    assert "3 candidate" in r["answer"]


def test_compare_intent_offline():
    s = _session()
    r = engine.answer(s, "compare candidate 1 and 2", settings=None)
    assert r["intent"] == "compare"
    assert "Ahmed Ali" in r["answer"] and "Sara Khan" in r["answer"]


def test_years_filter_offline():
    s = _session()
    r = engine.answer(s, "who has more than 5 years of experience?", settings=None)
    assert r["intent"] == "years_filter"
    assert set(r["candidate_ids"]) == {"c_001", "c_002"}  # 8y, 11y; Omar (3y) excluded


def test_exclude_years_offline():
    s = _session()
    r = engine.answer(s, "exclude candidates with less than 5 years", settings=None)
    assert r["intent"] == "exclude"
    assert "c_003" not in r["candidate_ids"]
    assert set(r["candidate_ids"]) == {"c_001", "c_002"}


def test_exclude_weak_offline():
    s = _session()
    r = engine.answer(s, "exclude the weak ones", settings=None)
    assert r["intent"] == "exclude"
    assert "c_003" not in r["candidate_ids"]  # Omar is NEEDS_REVIEW


def test_facts_block_shared_name_notice():
    # Three distinct people share one name (real-world template resumes). The
    # facts block MUST tell the model they are distinct and name the specific
    # candidates, so it never blends their histories or calls them 'the same
    # person'. Different-named candidates must NOT trigger the notice.
    from app.query import render

    profiles = [
        mk(1, "Jessica Claire", 2, ["GAAP", "Tax"]),
        mk(2, "Julie McFederal", 0, ["Excel"]),
        mk(3, "Jessica Claire", 9, ["Python", "C++"]),
        mk(5, "jessica  claire", 6, ["JavaScript"]),  # case/space variant
    ]
    fb = render.facts_block(profiles, {"candidate_count": 4})
    assert "SHARED NAMES" in fb
    assert "Jessica Claire" in fb
    # names the exact cluster (all three variants), not the distinct person
    assert "#1" in fb and "#3" in fb and "#5" in fb
    line = next(ln for ln in fb.splitlines() if "share the name" in ln)
    assert "#2" not in line  # Julie is not part of the cluster
    assert "do NOT combine" in fb  # anti-history-blend instruction


def test_facts_block_no_notice_when_names_unique():
    from app.query import render

    profiles = [mk(1, "Ahmed Ali", 8, ["HR"]), mk(2, "Sara Khan", 5, ["Payroll"])]
    fb = render.facts_block(profiles, {"candidate_count": 2})
    assert "SHARED NAMES" not in fb


def test_facts_block_duplicate_not_treated_as_shared_name():
    # A real DUPLICATE (same person, merged) must NOT trigger the distinct-people
    # notice - that would contradict the DUPLICATE status.
    from app.query import render

    profiles = [mk(1, "Ahmed Ali", 8, ["HR"]), mk(2, "Ahmed Ali", 8, ["HR"], dup="c_001")]
    fb = render.facts_block(profiles, {"candidate_count": 2})
    assert "SHARED NAMES" not in fb
