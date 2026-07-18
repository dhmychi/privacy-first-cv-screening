"""Generate synthetic English CV PDFs for local testing (fpdf2 text layer).
Usage (inside the container):  python tests/make_fixtures.py /tmp/fix
Creates cv_ahmed.pdf, cv_sara.pdf, cv_ahmed_dup.pdf and cvs.zip.
"""

import os
import sys
import zipfile

from fpdf import FPDF


def _cv(path, lines):
    pdf = FPDF(unit="pt", format="A4")
    pdf.add_page()
    y = 64
    for text, size in lines:
        if text:
            pdf.set_font("Helvetica", size=size)
            pdf.text(64, y, text)
        y += size + 6
    pdf.output(path)


AHMED = [
    ("Ahmed Ali", 18),
    ("Senior HR Specialist", 12),
    ("Email: ahmed.ali@acmecorp-hr.com  |  Mobile: +966 50 123 4567", 10),
    ("Riyadh, Saudi Arabia", 10),
    ("", 10),
    ("PROFESSIONAL SUMMARY", 12),
    ("HR specialist with 6 years in talent acquisition and HR operations.", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("Senior HR Specialist, Saudi Telecom (2021 - present)", 10),
    ("- Led recruitment for IT and engineering roles.", 10),
    ("- Administered SAP SuccessFactors for performance management.", 10),
    ("HR Officer, Almarai (2018 - 2021)", 10),
    ("- Managed onboarding and employee records.", 10),
    ("", 10),
    ("EDUCATION", 12),
    ("BSc Business Administration, King Saud University, 2017", 10),
    ("", 10),
    ("SKILLS", 12),
    (
        "Recruitment, Talent Acquisition, SAP SuccessFactors, Onboarding, Payroll, Arabic, English",
        10,
    ),
]

SARA = [
    ("Sara Khan", 18),
    ("Recruitment Lead", 12),
    ("Email: sara.khan@acmecorp-hr.com  |  Mobile: +966 55 987 6543", 10),
    ("Jeddah, Saudi Arabia", 10),
    ("", 10),
    ("PROFESSIONAL SUMMARY", 12),
    ("Recruitment leader with 9 years building teams across the GCC.", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("Recruitment Lead, Jeddah Holding (2015 - present)", 10),
    ("- Built and led a recruitment team of 8.", 10),
    ("- Implemented SAP HCM recruitment module.", 10),
    ("", 10),
    ("EDUCATION", 12),
    ("MBA, King Abdulaziz University, 2014", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, Employer Branding, SAP HCM, Stakeholder Management, English", 10),
]


LAYLA = [
    ("Layla Hassan", 18),
    ("HR Coordinator", 12),
    ("Email: layla.hassan@acmecorp-hr.com  |  Mobile: +966 56 222 1188", 10),
    ("Jeddah, Saudi Arabia", 10),
    ("", 10),
    ("PROFESSIONAL SUMMARY", 12),
    ("HR coordinator with 2 years in recruitment support and onboarding.", 10),
    ("", 10),
    ("EXPERIENCE", 12),
    ("HR Coordinator, Reef Retail (2023 - present)", 10),
    ("- Coordinated interviews and used SAP SuccessFactors for tracking.", 10),
    ("", 10),
    ("EDUCATION", 12),
    ("BSc Human Resources, Effat University, 2022", 10),
    ("", 10),
    ("SKILLS", 12),
    ("Recruitment, Onboarding, SAP SuccessFactors, Scheduling, English", 10),
]


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    a = os.path.join(outdir, "cv_ahmed.pdf")
    s = os.path.join(outdir, "cv_sara.pdf")
    la = os.path.join(outdir, "cv_layla.pdf")
    dup = os.path.join(outdir, "cv_ahmed_dup.pdf")
    _cv(a, AHMED)
    _cv(s, SARA)
    _cv(la, LAYLA)
    _cv(dup, AHMED)  # same person/email -> duplicate test
    zp = os.path.join(outdir, "cvs.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(a, "cv_ahmed.pdf")
        zf.write(s, "cv_sara.pdf")
        zf.write(la, "cv_layla.pdf")
    print(outdir)
    for f in (a, s, la, dup, zp):
        print(" ", f)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fix")
