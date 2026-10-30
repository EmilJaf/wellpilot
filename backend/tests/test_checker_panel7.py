"""The seventh panel's edits of the frozen demo runs (a simulated AI engineer, Oct 28
night), checked like a reader's edit in "Try to fool the checker": each false version must
be sent back, each true one accepted. The ones still open are listed in the fixture, not
asserted."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog

# Real-data regression suite: CI runs it; locally `uv run pytest -m "slow or not slow"`.
pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"
PROBES = json.loads((FIXTURES / "panel7_probes.json").read_text())


def _check(probe: dict) -> dict:
    run = next(
        json.loads(p.read_text())
        for folder in ("runs", "retired_runs")
        if (p := FIXTURES / folder / f"{probe['run']}.json").exists()
    )
    log = EvidenceLog.restore(run["evidence"].values())
    if probe["index"] == "S":
        return check_edits(run["question"], run["result"], log, [], probe["text"])
    edit = (int(probe["index"]), probe["text"], probe.get("conf") or "fact")
    return check_edits(run["question"], run["result"], log, [edit], None)


@pytest.mark.parametrize("probe", PROBES["caught"], ids=lambda p: f"{p['id']}-{p['kind']}")
def test_a_false_version_is_sent_back(probe: dict) -> None:
    assert not _check(probe)["accepted"], probe["why"]


@pytest.mark.parametrize("probe", PROBES["true"], ids=lambda p: f"{p['id']}-{p['kind']}")
def test_a_true_version_is_accepted(probe: dict) -> None:
    out = _check(probe)
    assert out["accepted"], out["problems"]


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        # The change between the periods' means written as a fall at a vague date (run
        # eb43551c59a6's "fell 72% ... around mid-2010") is the same misreading as at an ISO one.
        ("F-12 oil rate fell 55% around mid-2010 to 1 888 Sm3/d, while choke was cut.", False),
        ("F-12 oil rate fell 55% around 2010-05-02 to 1 888 Sm3/d, while choke was cut.", False),
        (
            "F-12 oil rate shifted down 55% around 2010-05-02, averaging 1 888 Sm3/d after it, "
            "while choke was cut.",
            True,
        ),
    ],
)
def test_a_change_between_periods_is_no_fall_at_a_date(text: str, accepted: bool) -> None:
    probe = {"run": "69519dd0a995", "index": 2, "text": text, "conf": "fact"}
    assert _check(probe)["accepted"] is accepted


SHIFT = (
    "F-12 oil rate {} around 2010-05-02, averaging 1 888 Sm3/d after it, while choke was cut "
    "from 83% to 58%."
)
NEEDS = (
    "F-12 needs attention most: it makes 1 551 Sm3/d oil at a water cut of {}, with its"
    " downhole gauge dead since 2010-10-12."
)


@pytest.mark.parametrize(
    ("index", "text", "accepted"),
    [
        # The outside review (Oct 29): a plus sign declared with the number reads as a rise,
        # and a count of days is no percentage. Their true neighbours pass.
        (2, SHIFT.format("moved +55%"), False),
        (2, SHIFT.format("shifted down 55%"), True),
        (1, NEEDS.format("81%"), False),
        (1, NEEDS.format("69%"), True),
    ],
)
def test_a_sign_and_a_unit_written_into_the_number(index: int, text: str, accepted: bool) -> None:
    from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission

    run = json.loads((FIXTURES / "retired_runs" / "69519dd0a995.json").read_text())
    log = EvidenceLog.restore(run["evidence"].values())
    finding = run["result"]["findings"][index - 1]
    gauge = next(
        e["id"] for e in run["evidence"].values() if '"zero_days": 81' in json.dumps(e["result"])
    )
    numbers = []
    for n in finding["numbers"]:
        if n["text"] == "55%" and "+55%" in text:
            n = {**n, "text": "+55%"}
        if n["text"] == "69%" and "81%" in text:
            n = {"text": "81%", "value": 81, "evidence_id": gauge}
        numbers.append(DeclaredNumber.model_validate(n))
    cited = sorted({*finding["evidence_ids"], *(n.evidence_id for n in numbers)})
    findings = [Finding.model_validate(f) for f in run["result"]["findings"]]
    findings[index - 1] = findings[index - 1].model_copy(
        update={"claim": text, "numbers": numbers, "evidence_ids": cited}
    )
    submission = Submission(summary="Done.", findings=findings)
    problems = [
        p
        for p in check_submission(submission, log, run["question"])
        if p.startswith(f"Finding {index}")
    ]
    assert (not problems) is accepted, problems
