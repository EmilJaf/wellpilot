"""The sixth panel's edits of the frozen demo runs (a simulated AI engineer, Oct 28), checked
like a reader's edit in "Try to fool the checker": each false version must be sent back,
each true one accepted. The ones still open are listed in the fixture, not asserted."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog

# Real-data regression suite: CI runs it; locally `uv run pytest -m "slow or not slow"`.
pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"
PROBES = json.loads((FIXTURES / "panel6_probes.json").read_text())


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
