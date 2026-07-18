# Synthetic data

Everything here is **fully synthetic** — no real people, companies, documents or
contact details. It exists so anyone can exercise the full screening pipeline
(in the default `CV_LLM_MODE=mock`) without touching real candidate data.

## Generate it

```bash
python scripts/generate_synthetic_data.py            # writes ./synthetic_data/
# or choose an output directory:
python scripts/generate_synthetic_data.py /tmp/demo
```

The corpus is **deterministic** (content is hard-coded and document timestamps are
pinned), so re-running produces the same files. The generated files
(`cvs/`, `jds/`, `bundles/`, `manifest.json`) are git-ignored — the generator
script is the single source of truth and is fully readable.

## What it contains

- **17 CVs** (`cvs/`) spanning Arabic, English and mixed AR/EN, plus edge cases:
  sparse, conflicting dates, image-only (OCR path), an exact duplicate, a
  same-name-but-different-person pair, an embedded prompt-injection line, a CV
  carrying protected attributes (to prove they are excluded), and a
  fairness-equivalent group of four.
- **3 job descriptions** (`jds/`): Arabic, English, and an AI-Engineer JD
  (Python, FastAPI, RAG, Agents, SQL, Docker).
- **3 demo ZIPs** (`bundles/`) ready to upload: `demo_english.zip` (a
  fully-English showcase batch), `demo_ai_candidates.zip`, and `demo_full_batch.zip`.
- **`manifest.json`** — ground truth for the tests and evals.

## Conventions

- Names are invented; organisations are generic (`Demo Organization`,
  `Example Company`, `Demo University`).
- Emails use the RFC-2606 reserved domain `example.com`, so they reach no real
  inbox. The extractor intentionally treats these as placeholders — positive
  contact extraction is demonstrated via the (fake) phone numbers.
- Document metadata is generic (no author identity).
