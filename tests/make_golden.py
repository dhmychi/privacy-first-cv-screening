"""Generate a diverse golden set of synthetic CVs for evaluation:
clean EN, junior, no-email, sparse, a same-name pair (different people),
a scanned (image-only) CV that forces OCR, and a multi-CV-in-one-PDF.
Each fixture ships with an `EXPECT` record for the metrics harness.

Usage (in container):  python tests/make_golden.py /tmp/golden
"""

import json
import os
import sys

from fpdf import FPDF


def _text_pdf(path, lines):
    pdf = FPDF(unit="pt", format="A4")
    pdf.add_page()
    y = 64
    for text, size in lines:
        if text:
            pdf.set_font("Helvetica", size=size)
            pdf.text(64, y, text)
        y += size + 6
    pdf.output(path)


def _multi_pdf(path, blocks):
    """One PDF, several people (each a page) -> should be flagged multi_cv."""
    pdf = FPDF(unit="pt", format="A4")
    for lines in blocks:
        pdf.add_page()
        y = 64
        for text, size in lines:
            if text:
                pdf.set_font("Helvetica", size=size)
                pdf.text(64, y, text)
            y += size + 6
    pdf.output(path)


def _scanned_pdf(path, lines):
    """Rasterize text to an image-only PDF (no text layer) so it MUST be OCR'd."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(img)
    try:
        base = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        fonts = {s: ImageFont.truetype(base, s) for s in (22, 30, 38)}
    except Exception:
        fonts = {s: ImageFont.load_default() for s in (22, 30, 38)}
    y = 50
    for text, size in lines:
        f = fonts.get(38 if size >= 16 else (30 if size >= 12 else 22), fonts[22])
        d.text((50, y), text, fill="black", font=f)
        y += (size * 2) + 12
    img.save(path, "PDF", resolution=200)


CLEAN = [
    ("Ahmed Ali", 18),
    ("Senior HR Specialist", 12),
    ("Email: ahmed.ali@acmecorp-hr.com  |  Mobile: +966 50 123 4567", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("Senior HR Specialist, Saudi Telecom (2021 - present)", 10),
    ("HR Officer, Almarai (2018 - 2021)", 10),
    ("", 10),
    ("EDUCATION", 12),
    ("BSc Business Administration, King Saud University, 2017", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, Talent Acquisition, SAP SuccessFactors, Onboarding, Payroll", 10),
]
NO_EMAIL = [
    ("Khalid Omar", 18),
    ("Recruiter", 12),
    ("Mobile: +966 53 444 7788  |  Dammam, Saudi Arabia", 10),
    ("", 10),
    ("PROFESSIONAL SUMMARY", 12),
    ("Recruiter with 5 years of experience in sourcing and full-cycle hiring across the Gulf.", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("Recruiter, Gulf Foods (2019 - present)", 10),
    ("- Managed end-to-end recruitment for retail and logistics roles.", 10),
    ("Talent Sourcer, Hire Co (2017 - 2019)", 10),
    ("", 10),
    ("EDUCATION", 12),
    ("BSc Management, King Fahd University, 2016", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, Sourcing, Interviewing, Onboarding, Stakeholder Management", 10),
]
SPARSE = [("Min Lee", 18), ("Looking for opportunities.", 10)]
SAMENAME_A = [
    ("Mohammed Ali", 18),
    ("HR Manager", 12),
    ("Email: mohammed.ali1@acmecorp-hr.com", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("HR Manager, Acme (2016 - present)", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, SAP HCM, Compensation, Payroll", 10),
]
SAMENAME_B = [
    ("Mohammed Ali", 18),
    ("Talent Acquisition Specialist", 12),
    ("Email: mohammed.ali2@acmecorp-hr.com", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("TA Specialist, Beta Corp (2020 - present)", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, Sourcing, Employer Branding", 10),
]
SCANNED = [
    ("Noura Saleh", 18),
    ("HR Analyst", 12),
    ("Email: noura.saleh@acmecorp-hr.com", 10),
    ("Mobile: +966 50 222 3344", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("HR Analyst, Tatweer (2019 - present)", 10),
    ("", 10),
    ("SKILLS", 12),
    ("HR Analytics, Recruitment, Excel, SAP SuccessFactors", 10),
]
MULTI_P1 = [
    ("Fatima Noor", 18),
    ("Payroll Officer", 12),
    ("Email: fatima.noor@acmecorp-hr.com", 10),
    ("Payroll Officer, NB Co (2018 - present)", 10),
    ("Skills: Payroll, GOSI, Excel", 10),
]
MULTI_P2 = [
    ("Yousef Tariq", 18),
    ("Recruiter", 12),
    ("Email: yousef.tariq@acmecorp-hr.com", 10),
    ("Recruiter, NB Co (2020 - present)", 10),
    ("Skills: Recruitment, Sourcing", 10),
]

# expectations consumed by the eval harness
EXPECT = {
    "clean.pdf": {
        "name": "Ahmed Ali",
        "email": "ahmed.ali@acmecorp-hr.com",
        "must_skills": ["sap successfactors", "recruitment"],
        "min_years": 5,
        "status": "OK",
        "candidates": 1,
    },
    "no_email.pdf": {
        "name": "Khalid Omar",
        "phone_digits": "534447788",
        "status": "OK",
        "candidates": 1,
        "no_email": True,
    },
    "sparse.pdf": {"name": "Min Lee", "status": "NEEDS_REVIEW", "candidates": 1},
    "samename_a.pdf": {
        "name": "Mohammed Ali",
        "email": "mohammed.ali1@acmecorp-hr.com",
        "candidates": 1,
    },
    "samename_b.pdf": {
        "name": "Mohammed Ali",
        "email": "mohammed.ali2@acmecorp-hr.com",
        "candidates": 1,
    },
    "scanned.pdf": {
        "name": "Noura Saleh",
        "ocr": True,
        "must_skills": ["recruitment"],
        "candidates": 1,
    },
    "multi.pdf": {"multi_cv": True, "candidates": 1},
}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    _text_pdf(os.path.join(outdir, "clean.pdf"), CLEAN)
    _text_pdf(os.path.join(outdir, "no_email.pdf"), NO_EMAIL)
    _text_pdf(os.path.join(outdir, "sparse.pdf"), SPARSE)
    _text_pdf(os.path.join(outdir, "samename_a.pdf"), SAMENAME_A)
    _text_pdf(os.path.join(outdir, "samename_b.pdf"), SAMENAME_B)
    _scanned_pdf(os.path.join(outdir, "scanned.pdf"), SCANNED)
    _multi_pdf(os.path.join(outdir, "multi.pdf"), [MULTI_P1, MULTI_P2])
    with open(os.path.join(outdir, "expect.json"), "w", encoding="utf-8") as f:
        json.dump(EXPECT, f, ensure_ascii=False, indent=2)
    print(outdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/golden")
