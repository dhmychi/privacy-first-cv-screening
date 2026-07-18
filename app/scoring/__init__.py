"""Deterministic fit-scoring plane.

Pipeline: JD text -> rubric (LLM proposes, code validates) -> per-requirement
matching (deterministic term evidence + local embeddings; NO per-candidate LLM)
-> deterministic aggregation (weights, gates, caps, bands) -> deterministic
ranking -> code-rendered report.

Design rule: semantic components only PROPOSE (rubric structure, similarity);
deterministic code DECIDES, VERIFIES, COMPUTES and RENDERS. Same session +
same JD => byte-identical report.
"""

# Bump whenever scoring logic changes: it namespaces the per-session
# rubric/score caches, so persisted sessions never serve results computed by
# an older scoring plane.
# "11": rubric now merges LLM over-splits that collapse to the same display
# label (e.g. one "Bachelor's degree in Business, Marketing..." split into three
# 'Bachelor degree' rows), which previously inflated the denominator and let a
# label show as both matched and missing. Bump invalidates pre-fix cached
# rubrics/scores so every re-score rebuilds through the deduping build().
SCORING_REV = "11"
