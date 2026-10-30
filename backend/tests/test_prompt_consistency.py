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
        },
        # The prompt's declaration examples ("2 870", "38%") use made-up values.
        {
            "kind": "regime_change",
            "signal": "liquid_rate",
            "start": "2010-07-18",
            "details": {"before": 2870.4, "after": 1779.6, "relative_change": -0.38},
        },
    ],
    "first_oil": "2008-02-12",
}
# The submit tool's example finding is about a made-up well, P-2.
P2_EVENTS = {
    "events": [
        {
            "kind": "regime_change",
            "signal": "oil_rate",
            "start": "2018-11-07",
            "details": {"before": 1032.4, "after": 640.2, "relative_change": -0.3798},
        }
    ],
    "first_oil": "2015-03-01",
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
    ("window lengths", "Its oil fell over the last 90 days and the 180-day shut-in count.", ""),
    ("numbers from the question", "Over the 400 days after 2010-07-18 it fell.", "the 400 days"),
]


@pytest.mark.parametrize("clause,claim,question", CLAUSES, ids=[c[0] for c in CLAUSES])
def test_each_exemption_the_prompt_states_holds(clause: str, claim: str, question: str) -> None:
    assert clause in SYSTEM_PROMPT, f"the prompt no longer says {clause!r}; update this case"
    assert check(claim, [], question) == []


def test_the_prompts_rounding_examples_pass() -> None:
    """The prompt asks for readable precision; the checker must accept the examples it gives."""
    log = EvidenceLog()
    log.add("fit", {"well": "P-2"}, {"di_per_year": 0.3127, "oil_sm3": 2468915}, _tool)
    assert "0.31 per year" in SYSTEM_PROMPT and "2.47 million Sm3" in SYSTEM_PROMPT
    claim = "P-2 declined at 0.31 per year and produced 2.47 million Sm3 meanwhile."
    numbers = [
        DeclaredNumber(text="0.31", value=0.3127, evidence_id="E1"),
        DeclaredNumber(text="2.47 million", value=2468915, evidence_id="E1"),
    ]
    finding = Finding(claim=claim, evidence_ids=["E1"], confidence="fact", numbers=numbers)
    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []


def test_a_share_written_as_a_percent_needs_no_calculation() -> None:
    """The prompt says 0.513 may be written 51.3% with the evidence's value declared."""
    log = EvidenceLog()
    log.add("events", {"well": "F-12"}, {"after": 0.5134}, _tool)
    assert "declare 51.3% with the evidence's value 0.513" in SYSTEM_PROMPT
    finding = Finding(
        claim="F-12's water cut rose to 51.3%.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[DeclaredNumber(text="51.3%", value=0.5134, evidence_id="E1")],
    )
    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []


def test_the_submit_tools_example_finding_passes() -> None:
    """The schema shows the model one finding; it must be one the checker accepts."""
    import json

    from wellpilot.agent.loop import submit_tool_schema

    description = submit_tool_schema()["function"]["description"]
    example = json.loads(description[description.index("{") : description.rindex("}") + 1])
    log = EvidenceLog()
    log.add("detect_events", {"well": "P-2"}, P2_EVENTS, _tool)
    log.add("fit_decline", {"well": "P-2"}, {"best": {"r2_log": 0.91}}, _tool)
    example["evidence_ids"] = ["E1", "E2"]  # the example's E3/E7, as this log numbers them
    for number in example["numbers"]:
        number["evidence_id"] = {"E3": "E1", "E7": "E2"}[number["evidence_id"]]

    finding = Finding.model_validate(example)
    assert finding.detail and len(finding.numbers) == 2  # the claim's; the detail's are looked up
    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []


def test_a_question_as_of_a_past_date_names_no_field_and_no_operator() -> None:
    """The model may have read about Volve; asked as of 2010, what it read about the years
    after must not be invited in. The present-day prompt names the field."""
    from datetime import date
    from pathlib import Path

    from wellpilot.agent.loop import system_prompt
    from wellpilot.data.store import ProductionStore

    processed = Path(__file__).parent / "fixtures" / "processed"
    past = system_prompt(ProductionStore(processed, as_of=date(2010, 12, 31)))
    now = system_prompt(ProductionStore(processed))

    assert "Volve" not in past and "Equinor" not in past
    assert "Volve" in now


# F-12's values and dates from the whole history: an answer the time machine gives, or a
# fact from after any cut it might be asked at. The model reads the prompt and every tool
# description in each run, so none of them may hold one (audit, Oct 31: the examples used
# the time machine's own 359,000 and F-12's 2010-07-18 change).
FULL_HISTORY = [
    "359,000", "359 007", "324,000", "470,000", "340,000", "1.13 million", "1,133,053",
    "1133053", "0.4461", "0.45 per year", "1 146", "1146", "1145.72", "4 050", "4050",
    "4207", "1888", "1,890", "0.7171", "0.8732", "0.0566", "100% by 2012", "2095", "857",
    "2010-05-02", "2010-07-18", "2010-10-12", "2012-02-14", "2014-11-26",
]  # fmt: skip


@pytest.mark.parametrize("as_of", ["2009-06-30", "2010-12-31", "2014-10-31"])
def test_nothing_the_model_reads_comes_from_after_the_cut(as_of: str) -> None:
    import json
    from datetime import date
    from pathlib import Path

    from wellpilot.agent import review, sandbox, vision
    from wellpilot.agent.loop import statements_tool_schema, submit_tool_schema, system_prompt
    from wellpilot.agent.tools import tool_schemas
    from wellpilot.data.store import ProductionStore

    cut = date.fromisoformat(as_of)
    store = ProductionStore(Path(__file__).parent / "fixtures" / "processed", as_of=cut)
    texts = [
        system_prompt(store),
        system_prompt(store, statements=True),
        json.dumps([*tool_schemas(), submit_tool_schema(), statements_tool_schema()]),
        json.dumps([vision.VISION_TOOL.schema(), sandbox.PYTHON_TOOL.schema()]),
        review.SYSTEM,
        review.REVISE,
        vision.SYSTEM,
        vision.WARNING,
        sandbox.NO_DATA,
        sandbox.OWN_CODE,
    ]
    read = "\n".join(texts)
    assert [v for v in FULL_HISTORY if v in read] == []
    # Volve's years (first oil 2008, shutdown 2016): every date of them is before the cut.
    # The examples' made-up well P-2 lives in 2015-2020 and is dated outside Volve's years.
    later = [
        d
        for d in re.findall(r"\b((?:2007|2008|2009|201[0-6])-\d{2}(?:-\d{2})?)\b", read)
        if d > as_of[: len(d)]
    ]
    assert later == []
