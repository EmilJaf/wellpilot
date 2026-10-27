"""The system prompt's promises about the checker, as executable cases.

The prompt tells the model what it may write without declaring; the checker must accept
exactly that. Each clause of the prompt has a case here, so editing one without the other
fails the build. Drift between the two caused most of the week's false rejections.
"""

import re

import pytest

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import SYSTEM_PROMPT


def _tool() -> None:
    """Stands in for a tool function."""


EVENTS = {
    "events": [
        {
            "kind": "regime_change",
            "start": "2010-07-18",
            "details": {"before": 4050.37, "after": 1145.72, "relative_change": -0.7171},
        }
    ],
    "first_oil": "2008-02-12",
}


def check(claim: str, numbers: list[DeclaredNumber], question: str = "") -> list[str]:
    log = EvidenceLog()
    log.add("detect_events", {"well": "F-12"}, EVENTS, _tool)
    finding = Finding(claim=claim, evidence_ids=["E1"], confidence="fact", numbers=numbers)
    return check_submission(Submission(summary="Done.", findings=[finding]), log, question)


def test_the_prompts_declaration_examples_pass() -> None:
    examples = re.findall(r'\{\{"text": "([^"]+)", "value": ([\d.]+)', SYSTEM_PROMPT)
    assert len(examples) == 2
    numbers = [DeclaredNumber(text=t, value=float(v), evidence_id="E1") for t, v in examples]
    claim = " and ".join(t for t, _ in examples)
    assert check(f"The rate fell {claim}.", numbers) == []


# (clause the prompt states, a claim that relies on it, the question asked)
CLAUSES = [
    ("Write dates as YYYY-MM-DD", "The rate stepped down on 2010-07-18.", ""),
    ("years", "The rate stepped down in 2010.", ""),
    ("well names", "15/9-F-12 stepped down.", ""),
    ("integers up to 10", "F-12 is one of 3 producers checked in 2 steps.", ""),
    ("window lengths", "The 30-day median and the next 12 months both show it.", ""),
    ("numbers from the question", "Over the 400 days after 2010-07-18 it fell.", "the 400 days"),
]


@pytest.mark.parametrize("clause,claim,question", CLAUSES, ids=[c[0] for c in CLAUSES])
def test_each_exemption_the_prompt_states_holds(clause: str, claim: str, question: str) -> None:
    assert clause in SYSTEM_PROMPT, f"the prompt no longer says {clause!r}; update this case"
    assert check(claim, [], question) == []


def test_the_prompts_rounding_examples_pass() -> None:
    """The prompt asks for readable precision; the checker must accept the examples it gives."""
    log = EvidenceLog()
    log.add("fit", {"well": "F-12"}, {"di_per_year": 0.4461, "oil_sm3": 1133053}, _tool)
    assert "0.45 per year" in SYSTEM_PROMPT and "1.13 million Sm3" in SYSTEM_PROMPT
    claim = "F-12 declined at 0.45 per year and produced 1.13 million Sm3 meanwhile."
    numbers = [
        DeclaredNumber(text="0.45", value=0.4461, evidence_id="E1"),
        DeclaredNumber(text="1.13 million", value=1133053, evidence_id="E1"),
    ]
    finding = Finding(claim=claim, evidence_ids=["E1"], confidence="fact", numbers=numbers)
    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []
