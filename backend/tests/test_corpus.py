"""Replay every real submission in tests/corpus through the current checker.

A case fails when the checker's verdict on a real answer changes: accepted or not, and
which findings it flags. See wellpilot.agent.corpus for adding cases.
"""

import json
from pathlib import Path

import pytest

from wellpilot.agent.corpus import CORPUS_DIR, replay, verdict

CASES = sorted(CORPUS_DIR.glob("*.json"))


def test_the_corpus_is_not_empty() -> None:
    assert CASES, f"no cases in {CORPUS_DIR}"


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_real_submission_keeps_its_verdict(path: Path) -> None:
    case = json.loads(path.read_text())

    problems = replay(case)

    assert verdict(problems) == case["expected"], (
        f"{case['note'] or 'no note'}\nproblems now:\n" + "\n".join(problems)
    )
