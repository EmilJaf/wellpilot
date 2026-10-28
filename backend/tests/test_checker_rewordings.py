"""True rewordings of the featured answers (tests/fixtures/rewordings.json, written by the
paraphrase fuzzer and reviewed): each keeps the verdict it should get. The accepted ones
include the nine phrasings the checker used to misread (Oct 29): "no gauge issue ..., but
from <date>", "rather than an abrupt choke cut", "a liquid rate of 5 130 oil fell", "the
next twelve months", "the fit implies", "shut off ~32% water", "gone down to about 31%"
after a dated shut-in, "97% before and 31% after in late 2014"."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.paraphrase import check_rewording

FIXTURES = Path(__file__).parent / "fixtures"
REWORDINGS = json.loads((FIXTURES / "rewordings.json").read_text())["rewordings"]
RUNS = {
    r: json.loads((FIXTURES / "runs" / f"{r}.json").read_text())
    for r in {x["run"] for x in REWORDINGS}
}
LOGS = {r: EvidenceLog.restore(run["evidence"].values()) for r, run in RUNS.items()}


@pytest.mark.parametrize(
    "item", REWORDINGS, ids=[f"{x['run']}-{x['passage']}-{n}" for n, x in enumerate(REWORDINGS)]
)
def test_a_rewording_keeps_its_verdict(item: dict) -> None:
    run = RUNS[item["run"]]
    problems = check_rewording(
        run["question"],
        run["result"],
        LOGS[item["run"]],
        item["index"],
        item["text"],
        item["numbers"],
    )
    assert (not problems) is item["accepted"], item.get("why") or problems


def test_circa_is_an_approximation() -> None:
    run = RUNS["e10e0aaf0542"]
    first = run["result"]["findings"][0]
    text = first["claim"].replace("about 174", "circa 174")
    assert text != first["claim"]
    log = LOGS["e10e0aaf0542"]
    assert check_rewording(run["question"], run["result"], log, 1, text, first["numbers"]) == []
