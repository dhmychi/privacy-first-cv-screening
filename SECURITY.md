# Security policy

Security and privacy are core design goals of this project, not an afterthought.

## Design posture

- **On-box by default.** Candidate PII never leaves the machine. Optional models
  run on a local Ollama; a startup guard refuses to boot if an off-box OCR/vision
  URL is configured without an explicit allow-flag.
- **Ephemeral data.** Analysed profiles live in a per-session store with a sliding
  TTL and auto-expire; `POST /sessions/{id}/reset` erases them immediately.
- **Untrusted-input handling.** Résumé text is treated as adversarial: a
  prompt-injection screen redacts AI-directed instructions before any model sees
  them, and protected attributes are never extracted.
- **Privacy-safe logging.** Logs carry only non-identifying metadata; it is
  structurally impossible to log CV content (see `app/logging_safe.py`).
- **Supply chain.** Dependencies are pinned and locked; CI runs `gitleaks` and
  `pip-audit`, and Dependabot proposes updates.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** via GitHub Security
Advisories ("Report a vulnerability" on the repository's *Security* tab) rather
than opening a public issue. Include a description, reproduction steps, and the
affected version. You can expect an initial acknowledgement within a few days.

## Scope

This is a decision-support tool and a portfolio project. All bundled data is
synthetic. Deploying it against real candidate data is the operator's
responsibility; review the [privacy notes](README.md#privacy--security) first.
