"""Transparent ranking/scoring (plan section 11). A score is a weighted blend of
must-have coverage (literal + semantic evidence), years fit (deterministic), and
semantic role match (bge-m3). Every component is inspectable and every hit
carries the candidate's own evidence. NEEDS_REVIEW candidates are penalized, not
hidden. No protected attribute ever enters scoring."""

from __future__ import annotations

from typing import Any

from . import embeddings


def candidate_blob(p: dict[str, Any]) -> str:
    parts = [p.get("headline") or ""]
    parts += [s["name"] for s in p.get("skills", [])]
    parts += [e.get("title") or "" for e in p.get("experiences", [])]
    parts.append(p.get("summary") or "")
    return ", ".join(x for x in parts if x) or (p["identity"]["full_name"]["value"] or "")


class EmbedIndex:
    """One batched embedding pass per query: candidate blobs + all skill names.
    Query texts are embedded lazily and cached. Degrades to literal-only (ok=False)
    if embeddings are unavailable."""

    def __init__(self, settings, candidates: list[dict[str, Any]]):
        self.s = settings
        self.blobs = [candidate_blob(p) for p in candidates]
        flat: list[str] = []
        self._skill_owner: list[int] = []
        for ci, p in enumerate(candidates):
            for s in p.get("skills", []):
                self._skill_owner.append(ci)
                flat.append(s["name"])
        vecs = embeddings.embed(settings, self.blobs + flat)
        self.ok = bool(vecs) and len(vecs) == len(self.blobs) + len(flat)
        self.blob_vecs = vecs[: len(self.blobs)] if self.ok else []
        self.skill_vecs: dict[int, list[tuple[str, list[float]]]] = {}
        if self.ok:
            for owner, name, v in zip(
                self._skill_owner, flat, vecs[len(self.blobs) :], strict=False
            ):
                self.skill_vecs.setdefault(owner, []).append((name, v))
        self._qcache: dict[str, list[float] | None] = {}

    def qvec(self, text: str) -> list[float] | None:
        if not self.ok or not text:
            return None
        if text not in self._qcache:
            r = embeddings.embed(self.s, [text])
            self._qcache[text] = r[0] if r else None
        return self._qcache[text]

    def blob_sim(self, ci: int, text: str) -> float:
        v = self.qvec(text)
        if v and ci < len(self.blob_vecs):
            return embeddings.cosine(v, self.blob_vecs[ci])
        return 0.0

    def best_skill_sim(self, ci: int, text: str) -> tuple[float, str | None]:
        v = self.qvec(text)
        if not v:
            return 0.0, None
        best, nm = 0.0, None
        for name, sv in self.skill_vecs.get(ci, []):
            c = embeddings.cosine(v, sv)
            if c > best:
                best, nm = c, name
        return best, nm


def match_skill_in_candidate(profile, term, ci, index, sem_threshold: float = 0.62):
    """Find a candidate's skill matching `term`, literal first then semantic.
    Returns {skill, evidence, how, source} or None."""
    tl = (term or "").lower().strip()
    for s in profile.get("skills", []):
        nml = s["name"].lower()
        if tl and (tl in nml or nml in tl):
            return {
                "skill": s["name"],
                "evidence": s.get("evidence"),
                "how": "literal",
                "source": s.get("source", "inferred"),
            }
    if index and index.ok:
        sim, nm = index.best_skill_sim(ci, term)
        if sim >= sem_threshold and nm:
            s = next((x for x in profile.get("skills", []) if x["name"] == nm), None)
            return {
                "skill": nm,
                "evidence": (s or {}).get("evidence"),
                "how": f"semantic({sim:.2f})",
                "source": (s or {}).get("source", "inferred"),
            }
    return None


def score_candidate(profile, ci, criteria, index) -> dict[str, Any]:
    must = criteria.get("must_have") or []
    hits, miss = [], []
    for skill in must:
        m = match_skill_in_candidate(profile, skill, ci, index)
        if m:
            hits.append({"requirement": skill, **m})
        else:
            miss.append(skill)
    coverage = (len(hits) / len(must)) if must else None

    years = profile["total_years_experience"]["value"] or 0
    min_years = criteria.get("min_years")
    years_fit = min(years / min_years, 1.0) if min_years else min(years / 10.0, 1.0)

    role_text = criteria.get("role_text") or " ".join(must)
    sem = index.blob_sim(ci, role_text) if (index and role_text) else 0.0

    if must:
        assert coverage is not None  # non-None whenever `must` is non-empty
        base = 0.5 * coverage + 0.25 * years_fit + 0.25 * sem
    else:
        base = 0.6 * sem + 0.4 * years_fit

    needs_review = profile["extraction"]["status"] == "NEEDS_REVIEW" or bool(
        profile.get("duplicate_of")
    )
    score = base * (0.85 if needs_review else 1.0)
    return {
        "score": round(score, 4),
        "coverage": coverage,
        "hits": hits,
        "missing": miss,
        "years": years,
        "semantic": round(sem, 3),
        "needs_review": needs_review,
    }


def rank(candidates, criteria, index) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scored = [(p, score_candidate(p, ci, criteria, index)) for ci, p in enumerate(candidates)]
    scored.sort(key=lambda x: x[1]["score"], reverse=True)
    return scored
