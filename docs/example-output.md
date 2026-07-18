# Example output

Everything below is **real, unedited output** from the service running in the
default `mock` mode on the bundled synthetic corpus — reproduce it with:

```bash
python scripts/generate_synthetic_data.py
docker compose up
# then upload synthetic_data/bundles/demo_english.zip and POST an AI-Engineer JD
# (see the API examples in the README)
```

No models, no network, no API key. The batch contains fully synthetic candidates,
plus an exact duplicate and a CV carrying a prompt-injection line. (The corpus
also ships Arabic and bilingual CVs — the pipeline is bilingual — omitted from
this showcase for quick global readability.)

---

## 1. After upload — the analysed roster

`GET /sessions/{id}/status` returns a screening summary. Note the deterministic
years, the top skills, and the automatic **duplicate** and **injection** flags:

> ## CV Batch Analyzed — 7 candidate(s)
> _1 flagged for review, 1 duplicate._
>
> - **Candidate 1 — David Kim** (10y) · Python, FastAPI, RAG, Agents, SQL
> - **Candidate 2 — Jordan Miller** (8y) · Python, FastAPI, SQL, Docker, scikit-learn
> - **Candidate 3 — Taylor Brooks** (5y) · Python, Java, SQL, Docker
> - **Candidate 4 — Chris Evans** (6y) · Python, SQL, Pandas, Machine Learning
> - **Candidate 5 — Robin Shaw** (5y) · Excel, SQL, Reporting, Process Improvement
> - **Candidate 6 — Jordan Miller** (8y) · Python, FastAPI, SQL, Docker  ⚠️ _DUPLICATE_
> - **Candidate 7 — Morgan Lee** (7y) · Python, FastAPI, PostgreSQL, Docker  ⚠️ _INJECTION_SUSPECTED_
>
> _Decision support for HR — not a final hiring decision._

---

## 2. Fit scoring against a job description

`POST /sessions/{id}/score` with an **AI Engineer** JD (Python, FastAPI, RAG,
Agents, SQL, Docker; min 4 years; Bachelor). The report is rendered entirely by
deterministic code — the score bars, banding and ranking are reproducible:

### HR Fit Scoring Results

**Role: AI Engineer — Demo Organization — 5 candidates scored.** Top match:
**David Kim** — 100% (Strong Fit).
_Scored: 5 · Needs review (not scored): 1 · Duplicates excluded: 1._

| Rank | CV # | Name | Fit Score | Level | Matched | Missing |
|:--:|--:|---|:--|:--:|---|---|
| 1 | 1 | **David Kim** | `██████████` 100% | Strong | Python, FastAPI, RAG, Agents, SQL, Docker | None |
| 2 | 2 | **Jordan Miller** | `███████░░░` 68% | Good | Python, FastAPI, SQL, Docker, 4+ yrs, Bachelor | RAG, Agents |
| 3 | 3 | **Taylor Brooks** | `█████░░░░░` 51% | Partial | Python, SQL, Docker, 4+ yrs experience | FastAPI, RAG, Agents |
| 4 | 4 | **Chris Evans** | `████░░░░░░` 45% | Partial | Python, SQL, 4+ yrs, Bachelor | FastAPI, RAG, Agents, Docker |
| 5 | 5 | **Robin Shaw** | `███░░░░░░░` 34% | Weak | SQL, 4+ yrs, Bachelor | Python, FastAPI, RAG, Agents, … |

_Fit bands: Strong 80–100 · Good 60–79 · Partial 40–59 · Weak below 40. Scores
are computed by a deterministic engine from evidenced requirements._

### Evidence — top candidates

- **David Kim (CV #1)** — 10 years experience. Python, FastAPI, RAG, Agents, SQL, Docker (p.1).
- **Jordan Miller (CV #2)** — 8 years experience. Python, FastAPI, SQL, Docker, 4+ yrs, Bachelor (p.1).
- **Taylor Brooks (CV #3)** — 5 years experience. Python, SQL, Docker, 4+ yrs experience (p.1).

### Gaps / risks — computed, not narrated

- **Agents:** not evidenced by 4 of 5 scored candidates.
- **RAG:** not evidenced by 4 of 5 scored candidates.
- **FastAPI:** not evidenced by 3 of 5 scored candidates.
- **Unverifiable fields:** 1 candidate had a requirement that could not be verified from the CV (excluded from their score denominator, never silently zeroed).

### Needs review — not scored

- **Morgan Lee (CV #7)** — the CV contains text that appears aimed at influencing
  the screening system (it was ignored); manual review required.

> The injected CV is **not** ranked. Its adversarial instruction was redacted
> before any model saw it, and the candidate was routed to a human — the system
> never let it jump the queue.

---

## 3. Fairness audit

`POST /sessions/{id}/audit` with operator-supplied group labels returns both the
protected-leakage report and the EEOC four-fifths analysis:

```json
{
  "leakage": { "protected_leaks": 0, "clean": true },
  "adverse_impact": {
    "four_fifths_threshold": 0.8,
    "adverse_impact": true,
    "axes": { "gender": { "groups": {
      "female": { "selection_rate": 1.0, "impact_ratio": 1.0, "adverse": false },
      "male":   { "selection_rate": 0.0, "impact_ratio": 0.0, "adverse": true }
    } } }
  }
}
```

`leakage.clean = true` proves no protected attribute reached the scored output;
`impact_ratio < 0.8` flags an adverse-impact outcome for review.
