"""Golden-set guarantees (live; opt-in CV_LIVE_LLM=1). Asserts the
non-negotiables: 100% evidence-validity, zero protected leakage, and every
golden check (same-name no-merge, multi-CV flag, OCR, identity, dedup)."""

import os

import pytest

from app.config import get_settings
from tests import eval_golden

pytestmark = pytest.mark.skipif(
    os.environ.get("CV_LIVE_LLM") != "1",
    reason="set CV_LIVE_LLM=1 to run the golden-set evaluation",
)


def test_golden_guarantees(tmp_path):
    m = eval_golden.evaluate(get_settings(), str(tmp_path))
    assert m["protected_leaks"] == 0
    assert m["evidence_validity"] == 1.0, f"evidence-validity {m['evidence_validity']:.2%}"
    failed = [name for name, ok in m["details"] if not ok]
    assert m["all_passed"], f"failed checks: {failed}"
