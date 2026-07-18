# API reference

Base URL (default): `http://127.0.0.1:8089`
Interactive docs (FastAPI/OpenAPI): `GET /docs`.

## Authentication

Every endpoint except `GET /health` requires the API key in a header:

```
X-API-Key: <CV_API_KEY>
```

A missing or wrong key returns `401`.

## Conventions

- A **session** is keyed by a client-supplied `chat_id` (path segment). A batch is
  analysed once per session; the profiles then persist (ephemerally) for scoring
  and follow-up questions.
- **Analysis gaps are `200` results**, not errors: a thin/unreadable/non-CV file
  comes back as a *Needs Review* candidate with a reason. Only genuine infra/input
  failures use 4xx/5xx (`401` auth, `413` too large, `415` no usable files, `404`
  unknown session).
- Responses that render markdown include `llm_mode`; in `mock` mode the markdown
  is prefixed with a **demo banner** so it is never mistaken for real inference.

---

## `GET /health`  (open)

Liveness, version, and the active backend mode.

```json
{
  "status": "ok",
  "service": "cv-screening",
  "version": "0.1.0",
  "llm_mode": "mock",
  "ocr": { "tesseract": true, "langs": ["ara","eng"], "mode": "auto",
           "vision_enabled": false, "cloud_ocr": false },
  "models": { "llm": "qwen3.6:35b", "embed": "bge-m3:latest",
              "ollama": "http://host.docker.internal:11434" },
  "sessions": { "active": 0, "ttl_minutes": 120 },
  "time": "2026-01-01T00:00:00+00:00"
}
```

## `POST /sessions/{chat_id}/analyze`  (auth, multipart)

Start analysis of a batch. Returns immediately with a `job_id`; poll `status`.

**Form fields**

| Field | Type | Notes |
|---|---|---|
| `files` | file[] | one or more PDF / image / `.docx`, and/or a ZIP of them |
| `mode` | string | `replace` (default) or `append` (merge into the ready batch) |
| `options` | string | reserved JSON options (optional) |
| `signature` | string | client batch signature (optional) |

**202 Accepted**

```json
{ "chat_id": "demo", "job_id": "a1b2c3d4e5f6", "status": "queued",
  "mode": "replace", "message": "analysis started; poll status for progress",
  "status_url": "/sessions/demo/status" }
```

```bash
curl -X POST $BASE/sessions/demo/analyze -H "X-API-Key: $KEY" \
     -F "files=@batch.zip"
```

## `GET /sessions/{chat_id}/status`  (auth)

Progress while running; when `ready`, the roster.

```json
{
  "chat_id": "demo", "status": "ready",
  "progress": { "done": 3, "total": 3, "stage": "ready" },
  "candidate_count": 3, "needs_review": 1,
  "llm_mode": "mock",
  "roster": [
    { "no": 1, "candidate_id": "c_001", "name": "…", "years": 8,
      "top_skills": ["Python","FastAPI"], "status": "OK", "flags": [],
      "duplicate_of": null }
  ],
  "roster_markdown": "> 🧪 Demo (mock) mode …\n\n# Screening summary …"
}
```

`status` ∈ `queued | running | ready | error`.

## `GET /sessions/{chat_id}/facts`  (auth)

The grounded candidate facts (for a client LLM to reason over) plus the roster
markdown. Same readiness semantics as `status`.

## `POST /sessions/{chat_id}/query`  (auth)

Multi-turn Q&A over the analysed batch (count, rank, compare, exclude, shortlist,
"who has X"), with reference resolution across turns.

```json
// request
{ "question": "who has FastAPI and at least 5 years?" }
```

## `POST /sessions/{chat_id}/score`  (auth)

Score the batch against a job description.

```json
// request
{ "jd_text": "AI Engineer. Required: Python, FastAPI, RAG, SQL, Docker. Min 4 years.",
  "lang": "en", "top_n": 25 }
```

```json
// response (scorable)
{
  "chat_id": "demo", "status": "ready", "scorable": true, "llm_mode": "mock",
  "rubric": { "role_title": "AI Engineer", "clarity": "ok",
              "requirements": [ { "id": "r1", "label": "Python", "rtype": "skill" } ] },
  "scored_count": 3, "review_count": 1,
  "ranking": [ { "rank": 1, "cv": 2, "name": "…", "score": 78, "level": "good" } ],
  "report_md": "…matched / missing / evidence…"
}
```

A JD with no extractable requirements returns `scorable: false` with a reason and
an explanatory report — not an error.

## `POST /sessions/{chat_id}/audit`  (auth)

Always returns the protected-leakage report. If `labels` **and** `selected` are
supplied, also returns the EEOC four-fifths / NYC LL144 analysis.

```json
// request
{ "labels":   { "c_001": {"gender":"female"}, "c_002": {"gender":"male"} },
  "selected": { "c_001": true, "c_002": false } }
```

```json
// response
{
  "chat_id": "demo",
  "leakage": { "protected_leaks": 0, "clean": true, "details": [],
               "excluded_by_policy": ["age","dob","ethnicity","gender", "…"] },
  "adverse_impact": {
    "four_fifths_threshold": 0.8, "adverse_impact": true,
    "axes": { "gender": { "groups": {
      "female": { "selection_rate": 1.0, "impact_ratio": 1.0, "adverse": false },
      "male":   { "selection_rate": 0.0, "impact_ratio": 0.0, "adverse": true } } } }
  }
}
```

## `POST /sessions/{chat_id}/reset`  (auth)

Erase the session (memory + any on-disk copy).

```json
{ "reset": true }
```

---

## Error model

| Status | When |
|---|---|
| `401` | missing/invalid `X-API-Key` |
| `404` | unknown / expired session |
| `413` | an uploaded file exceeds `CV_MAX_UPLOAD_MB` |
| `415` | no usable CV files in the upload |
| `200` + `scorable:false` / `status:"error"` / review flags | a *result*, not a failure — see conventions above |

## Configuration (`CV_*` environment variables)

Auth & mode: `CV_API_KEY`, `CV_LLM_MODE` (`mock`|`ollama`), `CV_MATCHER`
(`judge`|`legacy`). Models: `CV_OLLAMA_URL`, `CV_LLM_MODEL`, `CV_EMBED_MODEL`,
`CV_VLM_MODEL`. OCR: `CV_OCR_MODE`, `CV_OCR_LANGS`, `CV_OCR_DPI`,
`CV_OCR_MIN_CONFIDENCE`. Caps: `CV_MAX_FILES`, `CV_MAX_UPLOAD_MB`,
`CV_MAX_PAGES_PER_DOC`, `CV_MAX_ZIP_ENTRIES`. Session: `CV_SESSION_TTL_MINUTES`,
`CV_DATA_DIR` (empty disables persistence). See `.env.example` and
`docker-compose.yml` for defaults.
