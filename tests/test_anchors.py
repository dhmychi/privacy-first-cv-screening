"""Fast offline unit tests for the deterministic layer (no LLM, no network)."""

from app.pipeline import anchors, dedup, extract


def test_find_emails():
    assert anchors.find_emails("contact ahmed.ali@acmecorp.com now") == ["ahmed.ali@acmecorp.com"]
    assert anchors.find_emails("none here") == []


def test_find_phones_international():
    phones = anchors.find_phones("Mobile: +966 50 123 4567")
    assert phones and "966" in phones[0].replace(" ", "")


def test_phone_does_not_match_year_range():
    # a date range must never be read as a phone number
    assert anchors.find_phones("Worked 2018-2023 at the company") == []


def test_arabic_indic_digits_normalize():
    assert anchors.normalize_digits("٠٥٠١٢٣") == "050123"


def test_compute_total_years_merges_overlap():
    exps = [{"start": "2018", "end": "2021"}, {"start": "2021", "end": "present"}]
    years, n = anchors.compute_total_years(exps)
    assert n == 2 and years is not None and years >= 5


def test_quote_verification():
    pages = [{"page": 1, "text": "Administered SAP SuccessFactors for performance management."}]
    ev = extract.verify_quote("SAP SuccessFactors", pages)
    assert ev and ev["page"] == 1
    assert extract.verify_quote("a fabricated quote not present", pages) is None
    assert extract.verify_quote("short", pages) is None  # below min length


def test_dedup_by_email_case_insensitive():
    profiles = [
        {
            "candidate_id": "c_001",
            "identity": {"emails": ["a@x.com"], "phones": []},
            "duplicate_of": None,
            "extraction": {"status": "OK", "reasons": []},
        },
        {
            "candidate_id": "c_002",
            "identity": {"emails": ["A@X.com"], "phones": []},
            "duplicate_of": None,
            "extraction": {"status": "OK", "reasons": []},
        },
    ]
    dedup.mark_duplicates(profiles)
    assert profiles[1]["duplicate_of"] == "c_001"
    assert "DUPLICATE" in profiles[1]["extraction"]["reasons"]


def test_protected_scrub_removes_stray_keys():
    p = extract._scrub_protected(
        {"identity": {"full_name": "x", "nationality": "Y"}, "_excluded_by_policy": ["nationality"]}
    )
    assert "nationality" not in p["identity"]
    assert p["_excluded_by_policy"] == ["nationality"]


def test_placeholder_emails_filtered():
    from app.pipeline import anchors

    txt = (
        "Contact: resumesample@example.com, your.email@gmail.com, "
        "real.person@company.com, firstname.lastname@corp.test, anna@novoresume.com"
    )
    emails = anchors.find_emails(txt)
    assert "real.person@company.com" in emails
    assert "anna@novoresume.com" in emails  # real-looking domains stay
    assert "resumesample@example.com" not in emails  # RFC-2606 domain
    assert "your.email@gmail.com" not in emails  # placeholder local-part
    assert "firstname.lastname@corp.test" not in emails


def test_placeholder_names_rejected():
    from app.pipeline import extract

    assert not extract._valid_name("Full Name")
    assert not extract._valid_name("Your Name Here")
    assert not extract._valid_name("Lorem Ipsum")
    assert extract._valid_name("Anna Gunther")
    assert extract._valid_name("Carole Chun")


def test_docx_is_supported():
    from app.pipeline.ingest import SUPPORTED, kind_of

    assert kind_of("CV1_Ahmed_AlOtaibi.docx") == "docx"
    assert kind_of("resume.PDF") == "pdf"
    assert kind_of("answer_key.xlsx") == "unsupported"  # spreadsheets are not CVs
    assert ".docx" in SUPPORTED
