"""The eighth panel's findings (Oct 29): two true phrasings the checker could not accept, a
direction word it missed, a water-cut level between periods written as a jump, and a figure
reading SAM 3.1 disagreed with, written as if it were the page's."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.checker import (
    DeclaredNumber,
    Finding,
    Submission,
    canonical,
    check_submission,
)
from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.tools import execute
from wellpilot.data.store import ProductionStore

FIXTURES = Path(__file__).parent / "fixtures"


def _run(run_id: str) -> dict:
    return next(
        json.loads(p.read_text())
        for folder in ("runs", "retired_runs")
        if (p := FIXTURES / folder / f"{run_id}.json").exists()
    )


def _declared(
    run_id: str, index: int, claim: str, conf: str, numbers: list, detail: str | None = None
) -> list[str]:
    """One finding replaced, with declarations as a model writes them (no edit carry-over);
    its detail kept unless one is given."""
    run = _run(run_id)
    log = EvidenceLog.restore(run["evidence"].values())
    findings = [Finding.model_validate(f) for f in run["result"]["findings"]]
    findings[index - 1] = findings[index - 1].model_copy(
        update={
            "claim": claim,
            "confidence": conf,
            "numbers": [DeclaredNumber(text=t, value=v, evidence_id=e) for t, v, e in numbers],
            **({"detail": detail} if detail is not None else {}),
        }
    )
    submission = Submission(
        summary=run["result"]["summary"],
        findings=findings,
        summary_numbers=[
            DeclaredNumber.model_validate(n) for n in run["result"]["summary_numbers"]
        ],
    )
    return [
        p
        for p in check_submission(submission, log, run["question"])
        if p.startswith(f"Finding {index}")
    ]


def test_per_cent_in_two_words_keeps_positions() -> None:
    assert canonical("a 69 per cent water cut") == "a 69  percent water cut"


@pytest.mark.parametrize(
    ("claim", "numbers", "accepted"),
    [
        (
            "F-12 is likely to make roughly 0.36 million Sm3 of oil over 2011, with a range of "
            "0.32 to 0.47 million Sm3; the fit is poor, so treat this with caution.",
            [
                ["0.36 million", 359007, "E13"],
                ["0.32", 324192, "E13"],
                ["0.47 million", 469712, "E13"],
            ],
            True,
        ),
        (
            "F-12 is likely to make roughly 0.36 million Sm3 of oil over 2011, with a range of "
            "0.22 to 0.47 million Sm3; the fit is poor, so treat this with caution.",
            [
                ["0.36 million", 359007, "E13"],
                ["0.22", 224192, "E13"],
                ["0.47 million", 469712, "E13"],
            ],
            False,
        ),
    ],
)
def test_a_range_shares_its_scale_word(claim: str, numbers: list, accepted: bool) -> None:
    assert (_declared("69519dd0a995", 6, claim, "inference", numbers) == []) is accepted


@pytest.mark.parametrize(("cut", "accepted"), [("69", True), ("96", False)])
def test_per_cent_is_a_percent(cut: str, accepted: bool) -> None:
    claim = (
        f"F-12 needs attention most: it makes 1551 Sm3/d of oil at a {cut} per cent water cut, "
        "and its downhole gauge has been dead since October 12, 2010."
    )
    numbers = [
        ["1551", 1551, "E2"],
        [f"{cut} per cent", int(cut) / 100 + 0.0039 * (cut == "69"), "E2"],
    ]
    assert (_declared("69519dd0a995", 1, claim, "fact", numbers) == []) is accepted


@pytest.mark.parametrize(("word", "accepted"), [("up", False), ("down", True)])
def test_shifted_up_is_a_rise(word: str, accepted: bool) -> None:
    run = _run("69519dd0a995")
    log = EvidenceLog.restore(run["evidence"].values())
    text = (
        f"F-12 oil rate shifted {word} 55% around 2010-05-02, averaging 1 888 Sm3/d after it, "
        "while choke was "
        "cut from 83% to 58%."
    )
    out = check_edits(run["question"], run["result"], log, [(2, text, "fact")], None)
    assert out["accepted"] is accepted, out["problems"]


def test_a_water_cut_between_periods_is_no_jump() -> None:
    """The detector measures the water cut across a change's date too (0.22 -> 0.31 at
    F-12's 2010-02-14 change, against period means of 0.027 and 0.513): "jumped ... to
    51.3%" reads the later period's mean as a step."""
    store = ProductionStore(FIXTURES / "processed")
    log = EvidenceLog()
    result = execute(store, "detect_events", json.dumps({"well": "F-12"}))["result"]
    item = log.add("detect_events", {"well": "F-12"}, result, execute)
    change = next(e for e in result if e["signal"] == "water_cut" and e["start"] == "2010-02-14")
    assert round(change["details"]["change_across"], 2) == 0.08

    def check(claim: str) -> list[str]:
        finding = Finding(
            claim=claim,
            evidence_ids=[item.id],
            confidence="fact",
            numbers=[DeclaredNumber(text="51.3%", value=0.5134, evidence_id=item.id)],
        )
        submission = Submission(
            summary="F-12's water rose.", findings=[finding], summary_numbers=[]
        )
        return [p for p in check_submission(submission, log, "") if p.startswith("Finding 1")]

    jumped = check("F-12's water cut jumped around 2010-02-14 to 51.3%.")
    assert any("gradual shift" in p and "8 points" in p for p in jumped), jumped
    assert check("F-12's water cut averaged 51.3% in the period after 2010-02-14.") == []


def test_a_figure_reading_sam_disagrees_with_must_say_so() -> None:
    """Run 063034849654 wrote the model's 'about 2 years' off PUD Figure 4.2 as the figure's,
    though SAM 3.1 measured 19 months (the evidence records the disagreement)."""
    numbers = [["9,000", 9000.0, "E3"], ["8,400", 8400.0, "E3"], ["2 years", 2.0, "E3"]]
    claim = (
        "Figure 4.2 on the same page plots the oil rate plateau slightly below 9,000 Sm3/d at "
        "about 8,400 Sm3/d for about {} from around 2008 to 2010."
    )
    silent = _declared("063034849654", 2, claim.format("2 years"), "inference", numbers)
    assert any("SAM 3.1's measurement" in p for p in silent), silent
    said = claim.format(
        "2 years by the model's reading, which SAM 3.1's measurement does not confirm"
    )
    assert _declared("063034849654", 2, said, "inference", numbers, detail="") == []
    # The reading tucked into the detail ("2 to 2.5 years") is judged the same way.
    detail = _declared("063034849654", 2, said, "inference", numbers)
    assert any("(its detail)" in p and "SAM 3.1" in p for p in detail), detail
    # The rate both models agree on needs no such words.
    agreed = (
        "Figure 4.2 on the same page plots the oil rate plateau slightly below 9,000 Sm3/d at "
        "about 8,400 Sm3/d."
    )
    assert _declared("063034849654", 2, agreed, "inference", numbers[:2], detail="") == []


@pytest.mark.parametrize(
    ("run_id", "index", "claim"),
    [
        # "near"/"just under" round like "about" (0.6939 as 70%)
        (
            "69519dd0a995",
            1,
            "F-12 needs attention most: it makes about 1,600 Sm3/d of oil at a water cut near 70%, "
            "with its downhole gauge dead since 2010-10-12.",
        ),
        (
            "69519dd0a995",
            1,
            "F-12 needs attention most: it makes 1 551 Sm3/d oil at a water cut just under 70%, "
            "with its downhole gauge dead since 2010-10-12.",
        ),
        # a well the cited result names beside the one it is about
        (
            "69519dd0a995",
            2,
            "F-12 was shut in for 12 days from 2010-08-16, at the same time as F-14, and came "
            "back about 10% higher at 2,000 Sm3/d.",
        ),
        # a range sharing its percent sign
        (
            "dca494b07eb4",
            5,
            "On the 2010 log's low-rate pass, a plug anywhere in the 3,180–3,220 m blank pipe "
            "would shut off about 31–33% of the water and about 1% of the oil.",
        ),
        # the pair rule (Oct 29 night): a change whose ends a calculation lists among its
        # inputs, and a day of a date the edit declared beside the same number written "31%"
        (
            "485c17df2acb",
            2,
            "After the 2014 job oil went up about 4.8-fold, from 177 to 857 Sm3/d, and water "
            "cut fell from 97% to 34%.",
        ),
        (
            "3d27b4a64831",
            1,
            "F-12 was shut in 32 days from 2014-11-30 to 2014-12-31 and returned with water cut "
            "down from about 97% to about 31% and oil up about 5 times.",
        ),
    ],
)
def test_panel9_true_rephrasings_pass(run_id: str, index: int, claim: str) -> None:
    """The ninth panel's true edits the checker sent back (judges9/ai_engineer/probes.json
    A07, T05, T07, T14; D06 and the eighth panel's S05 after the pair rule), through the
    reader's-edit check as the judges ran them."""
    run = _run(run_id)
    log = EvidenceLog.restore(run["evidence"].values())
    confidence = run["result"]["findings"][index - 1]["confidence"]
    out = check_edits(run["question"], run["result"], log, [(index, claim, confidence)], None)
    assert out["accepted"], out["problems"]
