"""The seventh panel's edits of the frozen demo runs (a simulated AI engineer, Oct 28
night), checked like a reader's edit in "Try to fool the checker": each false version must
be sent back, each true one accepted. The ones still open are listed in the fixture, not
asserted."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog

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
            "F-12 oil rate shifted down 55% around 2010-05-02 to 1 888 Sm3/d, while choke was cut.",
            True,
        ),
    ],
)
def test_a_change_between_periods_is_no_fall_at_a_date(text: str, accepted: bool) -> None:
    probe = {"run": "69519dd0a995", "index": 2, "text": text, "conf": "fact"}
    assert _check(probe)["accepted"] is accepted
