"""Metamorphic tests: every answer in the corpus the checker accepted, rewritten.

A real answer reworded the way models vary their writing (tests/metamorphic.py: typography,
dates in words, "about", the unit inside a declaration, commas in thousands) must still be
accepted; the same answer with one thing broken that the checker claims to judge (a number,
a date, a cause in a fact) must be rejected for that finding. The corpus grows with every
run, so these grow with it. Each false rejection found live on Oct 29 had a transform here
that would have caught it first (non-breaking hyphens, "to about 40%", a well's digits read
as a day, "choke has been at 117%").
"""

import json

import pytest
from metamorphic import BREAKING, PRESERVING

from wellpilot.agent.checker import canonical
from wellpilot.agent.corpus import CORPUS_DIR, replay

CASES = {
    p.stem: case
    for p in sorted(CORPUS_DIR.glob("*.json"))
    if (case := json.loads(p.read_text()))["expected"]["accepted"]
}


def test_canonical_keeps_every_position() -> None:
    text = "F‑12’s cut 0.97 on 2010‑07‑18: 4 050 − 1 146 — “ok” … １２"
    plain = canonical(text)
    assert len(plain) == len(text)
    assert plain == "F-12's cut 0.97 on 2010-07-18: 4 050 - 1 146 — “ok” … 12"


@pytest.mark.parametrize("name", sorted(PRESERVING))
@pytest.mark.parametrize("case_id", sorted(CASES))
def test_a_rewording_is_still_accepted(case_id: str, name: str) -> None:
    case = CASES[case_id]
    reworded = PRESERVING[name](case["submission"])
    if reworded is None:
        pytest.skip(f"{name} does not apply")
    assert replay(case | {"submission": reworded}) == []


@pytest.mark.parametrize("name", sorted(BREAKING))
@pytest.mark.parametrize("case_id", sorted(CASES))
def test_a_broken_answer_is_sent_back(case_id: str, name: str) -> None:
    case = CASES[case_id]
    broken = BREAKING[name](case)
    if broken is None:
        pytest.skip(f"{name} does not apply")
    submission, label = broken
    problems = replay(case | {"submission": submission})
    assert any(p.startswith(label) for p in problems), (label, problems)
