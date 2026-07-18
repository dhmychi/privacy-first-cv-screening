# Design decisions

A record of the engineering choices that shaped this project, the alternatives
each was weighed against, and the consequences. The format is lightweight ADR
(Architecture Decision Record): **Context → Decision → Alternatives → Consequences**.

---

## 1. Treat the LLM as untrusted: "the model points, the code reads"

**Context.** LLMs hallucinate. In hiring, a fabricated skill or degree is an
audit and fairness failure, not a cosmetic bug.

**Decision.** The model is used only to *read prose* and *make semantic
judgments*. Every fact that must be true is produced or verified by code:
contact details and years come from deterministic anchors; the model must return
a **verbatim quote** for each field, and code locates that quote in the source
pages and assigns the page number. An unverifiable claim is dropped.

**Alternatives considered.**
- *Trust structured LLM output directly* — simplest, but unciteable and prone to
  invented facts.
- *Ask the model to also return page numbers* — rejected: the model hallucinates
  citations too. Pages are assigned by code, never by the model.

**Consequences.** Output is citable and reproducible; the model can never inject
a fact that isn't in the document. Cost: an extra verification pass and the
occasional dropped-but-real quote (a safe failure — it degrades to *unverified*,
never to a fabricated *met*).

---

## 2. A deterministic mock backend, on by default

**Context.** A reviewer, an interviewer, or CI should be able to run the *whole*
system in seconds without a GPU, a model download, or a network.

**Decision.** Ship a `mock` backend (default) that stands in for every model call
with a deterministic, rule-based parser, and select it with `CV_LLM_MODE`. It
parses the real input (so results vary by CV/JD and are testable), is clearly
labelled as demo output, and is grounded the same way the real path is.

**Alternatives considered.**
- *Require Ollama to run anything* — high barrier; CI would need a model.
- *Fixed canned responses* — rejected: not testable and misleading; the mock must
  handle different inputs logically.

**Consequences.** The full pipeline (extraction → scoring → audit) is exercised by
130 deterministic tests with no network, and the project is runnable by anyone
immediately. Cost: the mock is a demonstration of *architecture*, not semantic
quality — real judgment needs `CV_LLM_MODE=ollama`, which the docs state plainly.

---

## 3. Deterministic-first extraction

**Context.** Emails, phone numbers, and years-of-experience are exactly the
fields an LLM is most tempted to "tidy up" or miscompute.

**Decision.** Extract them with code (regex anchors + date arithmetic with
overlap merging) *before* the model runs, and never let the model's own number
override the computed one. A major conflict between the computed total and the
candidate's stated total is surfaced as a verification note, not silently
resolved.

**Alternatives.** *Let the model return years* — rejected: models routinely add
overlapping roles or miscount. Deterministic date math is both correct and free.

**Consequences.** Numbers are trustworthy and explainable; the model is freed to
do only what it's good at (prose).

---

## 4. pypdfium2 for rendering, not PyMuPDF

**Context.** The pipeline needs to rasterise PDF pages (for OCR) and detect
image-only pages. PyMuPDF is the obvious choice — but it is **AGPL-3.0**, which
conflicts with an MIT-licensed public project.

**Decision.** Use **pypdfium2** (Apache-2.0 / BSD) for page rendering and image
detection, and **fpdf2** for authoring synthetic test PDFs. The repository is
100% free of AGPL/strong-copyleft runtime dependencies (the only weak-copyleft
dependency is `python-bidi`, LGPL, used unmodified for Arabic shaping).

**Alternatives.** *Keep PyMuPDF and relicense the project AGPL* — rejected: less
attractive as an open portfolio piece and imposes network-copyleft on any user.
*Document the AGPL obligation and keep PyMuPDF* — rejected: a permissive stack is
cleaner and demonstrates license awareness.

**Consequences.** Clean MIT posture; a `pip-audit` + license review that a
reviewer can re-run and find nothing to flag.

---

## 5. On-box, ephemeral by design

**Context.** Résumés are dense PII. The safest data is the data you never store.

**Decision.** Everything runs on-box (local Ollama for the optional models).
Profiles live in a per-chat session with a sliding TTL and then auto-expire;
`reset` erases them immediately. A startup guard refuses to boot if an off-box
OCR/vision URL is configured without an explicit allow-flag.

**Alternatives.** *Cloud LLM APIs* — rejected: sends candidate PII off-box.
*Long-term profile storage* — rejected: unnecessary for the use case and a
standing liability.

**Consequences.** A tight privacy envelope that is easy to reason about. Cost:
sessions are single-node (see [Limitations](../README.md#limitations)).

---

## 6. A grounded-reasoning judge, with the deterministic matcher as a floor

**Context.** Keyword/cosine matching mislabels a lot ("CPA" ≠ "Accountant", "SQL"
≠ "PostgreSQL", a history degree for a CS role). Pure-LLM judging hallucinates.

**Decision.** Default to a **grounded judge**: the model judges each requirement
by *meaning* (is this the relevant experience? the right degree field? the
specific credential?), returns a cited quote, and code grounds that quote —
ungrounded verdicts collapse to *unverified*. Arithmetic stays in code. The
legacy **keyword+cosine matcher** remains fully intact and is used automatically
for any candidate the judge fails on, so judge mode is never worse than legacy on
error, and `CV_MATCHER=legacy` is an instant, no-rebuild rollback.

**Alternatives.** *Rules-only matcher* — brittle, endless special cases.
*Judge-only* — no safety net on model failure.

**Consequences.** Better semantic accuracy with a deterministic floor and a
one-flag rollback. In `mock` mode the deterministic matcher *is* the judge, which
is the honest deterministic stand-in.

---

## 7. Privacy-safe logging by construction, not by redaction

**Context.** Redaction filters leak — someone always logs a new field that the
scrubber didn't anticipate.

**Decision.** Make PII-in-logs *structurally impossible*: `log_event`'s signature
is the allowlist (event, hashed session id, file count, duration, error **class**,
failed stage). There is no parameter that can carry CV text, a name, an email, a
prompt, or an evidence quote, and every value is coerced to a bounded slug. A
test proves a real analysis run emits zero PII.

**Alternatives.** *Regex/redaction middleware over free-form logs* — rejected:
fails open; a missed pattern leaks.

**Consequences.** Logs are safe to ship anywhere. Cost: log call sites can only
record the allowlisted fields — which is the point.

---

## 8. The fairness audit consumes external labels; it never infers demographics

**Context.** Measuring adverse impact (EEOC four-fifths / NYC LL144) needs group
membership — but the system deliberately does not extract protected attributes.

**Decision.** Protected attributes are never extracted or scored. The four-fifths
audit takes group labels and a selection set **supplied by the operator** and
computes per-group selection rates, impact ratios, and an adverse-impact flag.
The audit also always reports whether any protected attribute leaked into the
scored output (it never does — proven by the golden harness).

**Alternatives.** *Infer demographics from the CV to self-audit* — rejected:
inferring protected attributes is exactly the harm the system avoids.

**Consequences.** The tool can be audited for fairness without ever profiling
candidates.

---

## 9. Async worker + polling for batch analysis

**Context.** A 25–200 CV batch takes real time (OCR, model calls). A synchronous
request would time out.

**Decision.** `analyze` returns `202` immediately and runs on a background thread
pool; the client polls `status`, which streams progress and then the roster.

**Alternatives.** *Synchronous* — times out at scale. *Streaming (SSE/WebSocket)*
— nicer UX but heavier; kept on the [roadmap](../README.md#roadmap).

**Consequences.** Simple, robust, framework-agnostic. Polling is the small cost.

---

## 10. One model-backend choke point

**Context.** Model calls are scattered across extraction, judging, rubric-building
and JD parsing. Branching on the backend at each site would be error-prone.

**Decision.** Funnel every LLM call through `llm.chat_json(task=...)` and every
embedding through `embeddings.embed`, and branch on `CV_LLM_MODE` *once* inside
those. Call sites pass a `task` tag so the mock backend applies the right parser.

**Alternatives.** *Per-call-site branching* — duplicated and drift-prone.

**Consequences.** Swapping or adding a backend touches one place; the mock/real
split is invisible to the rest of the code.

---

## 11. Analysis gaps are results (200 + review lane), not errors

**Context.** A thin, unreadable, or non-CV file is a normal outcome, not a system
failure.

**Decision.** Such files return a `200` *Needs Review* result with a
human-readable reason (see `app/reasons.py`), never an HTTP error. Only genuine
infra/input failures use 4xx/5xx.

**Alternatives.** *4xx on a bad CV* — rejected: a client's error handling would
mask a safe, actionable "review this" outcome and could drop the whole batch.

**Consequences.** Robust batches (one bad file never kills the rest) and honest,
non-silent handling of edge cases.

---

## 12. Configuration entirely through environment variables

**Context.** The service runs in Docker, in tests, and locally, and must bake in
no machine-specific values.

**Decision.** Every setting is a `CV_*` environment variable read at call time,
so the same image behaves correctly everywhere and tests can flip any knob
per-case.

**Consequences.** Twelve-factor portability; no config files to leak or drift.
