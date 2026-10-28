"""The fifth judge panel's probes (Oct 28): an AI engineer's edits of the frozen runs
through "Try to fool the checker", each with what the checker now says. False edits within
what the checker claims to judge (numbers, units, dates, direction, wells, quotes, causes)
are sent back; true rephrasings pass. Edits that change only meaning ("most" for "least",
"deeper" for "shallower") are not pinned: the checker does not judge meaning, and the
README says so."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog

# Real-data regression suite: CI runs it; locally `uv run pytest -m "slow or not slow"`.
pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"
PROBES = json.loads((FIXTURES / "checker_probes_panel5.json").read_text())


def run(run_id: str) -> dict:
    for folder in ("runs", "retired_runs"):
        if (path := FIXTURES / folder / f"{run_id}.json").exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(run_id)


@pytest.mark.parametrize("probe", PROBES, ids=[p["tag"] for p in PROBES])
def test_a_probe_gets_the_verdict_it_should(probe) -> None:
    saved = run(probe["run"])
    log = EvidenceLog.restore(saved["evidence"].values())
    result = saved["result"]
    if probe["index"] == "S":
        verdict = check_edits(saved["question"], result, log, [], probe["claim"])
    else:
        confidence = probe["confidence"] or result["findings"][probe["index"] - 1]["confidence"]
        verdict = check_edits(
            saved["question"], result, log, [(probe["index"], probe["claim"], confidence)]
        )
    assert verdict["accepted"] is probe["accepted"], verdict["problems"]
