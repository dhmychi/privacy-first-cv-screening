"""Evaluation harness over the golden set. Ingests each fixture, runs the real
extraction pipeline, and measures the guarantees that matter for HR:
  * evidence-validity (every displayed quote exists in the CV)  -> target 100%
  * zero protected-attribute leakage
  * candidate count: same-name people NOT merged; multi-CV PDF flagged
  * identity / skill / status correctness; OCR path on scanned CVs
Run in container:  python tests/eval_golden.py /tmp/golden
Importable: evaluate(settings, outdir) -> metrics dict.
"""

import json
import os
import re
import sys

from app.config import get_settings
from app.pipeline import acquire, dedup, extract, segment
from tests import make_golden

PROTECTED = extract.PROTECTED_KEYS


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _profile_for(path, settings):
    name = os.path.basename(path)
    doc = acquire.acquire_document(name, path, "pdf", settings)
    cands = segment.documents_to_candidates([doc])
    return [extract.extract_candidate(c, i + 1, settings) for i, c in enumerate(cands)]


def _ptext(p):
    return "\n".join(pg["text"] for pg in p.get("_pages", []))


def _all_evidence(p):
    evs = []
    fn = p["identity"]["full_name"].get("evidence")
    if fn:
        evs.append(fn)
    for sec in ("experiences", "education", "skills", "certifications", "languages"):
        for it in p.get(sec, []):
            if it.get("evidence"):
                evs.append(it["evidence"])
    return evs


def _evidence_validity(p):
    txt = _norm(_ptext(p))
    ok = tot = 0
    for ev in _all_evidence(p):
        tot += 1
        if _norm(ev["quote"]) in txt:
            ok += 1
    return ok, tot


def _protected_leaks(obj):
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_excluded_by_policy":
                continue
            if k.lower() in PROTECTED:
                n += 1
            n += _protected_leaks(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _protected_leaks(v)
    return n


def evaluate(settings, outdir):
    make_golden.main(outdir)
    expect = json.load(open(os.path.join(outdir, "expect.json"), encoding="utf-8"))
    checks = []
    ev_ok = ev_tot = leaks = 0

    def add(name, cond):
        checks.append((name, bool(cond)))

    for fname, exp in expect.items():
        profs = _profile_for(os.path.join(outdir, fname), settings)
        add(f"{fname}: candidate_count={exp['candidates']}", len(profs) == exp["candidates"])
        p = profs[0]
        for ev_p in profs:
            o, t = _evidence_validity(ev_p)
            ev_ok += o
            ev_tot += t
            leaks += _protected_leaks(ev_p)
        if "name" in exp:
            add(f"{fname}: name", _norm(p["identity"]["full_name"]["value"]) == _norm(exp["name"]))
        if "email" in exp:
            add(f"{fname}: email", exp["email"] in p["identity"]["emails"])
        if exp.get("no_email"):
            add(
                f"{fname}: phone-anchor (no email)",
                not p["identity"]["emails"] and bool(p["identity"]["phones"]),
            )
        if "phone_digits" in exp:
            digits = "".join(c for c in "".join(p["identity"]["phones"]) if c.isdigit())
            add(f"{fname}: phone", exp["phone_digits"] in digits)
        if "status" in exp:
            add(f"{fname}: status={exp['status']}", p["extraction"]["status"] == exp["status"])
        if exp.get("ocr"):
            add(f"{fname}: OCR used", p["extraction"]["ocr_used"])
        if exp.get("multi_cv"):
            add(f"{fname}: multi_cv flagged", "MULTIPLE_CVS_IN_FILE" in p["extraction"]["reasons"])
        for sk in exp.get("must_skills", []):
            names = " ".join(s["name"].lower() for s in p["skills"])
            add(f"{fname}: skill '{sk}'", sk in names)

    # same-name people must NOT be merged or marked duplicate
    sn = _profile_for(os.path.join(outdir, "samename_a.pdf"), settings) + _profile_for(
        os.path.join(outdir, "samename_b.pdf"), settings
    )
    for i, pr in enumerate(sn):
        pr["candidate_id"] = f"c_{i + 1:03d}"
        pr["display_no"] = i + 1
    dedup.mark_duplicates(sn)
    add(
        "same-name: 2 distinct candidates, not merged",
        len(sn) == 2 and all(p["duplicate_of"] is None for p in sn),
    )

    # an exact duplicate (same email) MUST be flagged
    dup = _profile_for(os.path.join(outdir, "clean.pdf"), settings) + _profile_for(
        os.path.join(outdir, "clean.pdf"), settings
    )
    for i, pr in enumerate(dup):
        pr["candidate_id"] = f"c_{i + 1:03d}"
        pr["display_no"] = i + 1
    dedup.mark_duplicates(dup)
    add("duplicate: exact copy flagged", dup[1]["duplicate_of"] == dup[0]["candidate_id"])

    passed = sum(1 for _, ok in checks if ok)
    metrics = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "evidence_validity": (ev_ok / ev_tot if ev_tot else 1.0),
        "evidence_quotes": ev_tot,
        "protected_leaks": leaks,
        "all_passed": passed == len(checks) and leaks == 0 and (ev_ok == ev_tot),
        "details": checks,
    }
    return metrics


if __name__ == "__main__":
    s = get_settings()
    m = evaluate(s, sys.argv[1] if len(sys.argv) > 1 else "/tmp/golden")
    print("\n=== GOLDEN-SET EVALUATION ===")
    for name, ok in m["details"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  checks: {m['checks_passed']}/{m['checks_total']}")
    print(
        f"  evidence-validity: {m['evidence_validity'] * 100:.1f}%  ({m['evidence_quotes']} quotes)"
    )
    print(f"  protected leaks:   {m['protected_leaks']}")
    print(f"  OVERALL: {'PASS' if m['all_passed'] else 'FAIL'}")
    sys.exit(0 if m["all_passed"] else 1)
