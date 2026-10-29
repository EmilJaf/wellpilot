"""The third judge panel's probes (Oct 28 night): edits of featured answers that the
checker accepted though their meaning was wrong, now each caught, by "Try to fool it" on the
live site; and the true phrasings next to them that must still pass.

Run on the frozen time-machine run (8aef8d548380, as of 2010-12-31) and the retired
decline-vs-plan run (e51eada95269), through the same check the page uses."""

import json
from pathlib import Path

import pytest

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog

FIXTURES = Path(__file__).parent / "fixtures"


def load(run_id: str) -> tuple[dict, EvidenceLog]:
    for folder in ("runs", "retired_runs"):
        if (path := FIXTURES / folder / f"{run_id}.json").exists():
            run = json.loads(path.read_text())
            return run, EvidenceLog.restore(run["evidence"].values())
    raise FileNotFoundError(run_id)


TM = "8aef8d548380"  # finding 1: F-12 on alarm; 2: its 55% shift; 3: F-14; 5: the forecast
PLAN = "e51eada95269"  # finding 1: breakthrough vs the plan; 3: the 2010-07-18 change
F1 = (
    "F-12 needs attention most: alarm level at 1 551 Sm3/d oil with 69% water cut, downhole "
    "gauge dead since 2010-10-12 for 81 days, 123 258 Sm3 oil without pressure data."
)
F2 = (
    "F-12’s oil rate fell 55% around 2010-05-02, averaging 1 888 Sm3/d after it, a gradual "
    "shift with water "
    "cut rising from 3% to 57% and liquid holding while choke narrowed."
)
F5 = (
    "Forecast for F-12 for next 12 months (2011-01-01 to 2011-12-31) is 359 007 Sm3 P50 with "
    "calibrated band 324 192 to 469 712 Sm3."
)
BREAKTHROUGH = (
    "The plan expected water breakthrough after about 2 years, and F-12 first exceeded 5% "
    "water cut on 2009-10-31, about 21 months after first oil."
)

CAUGHT = [
    # what it is: oil, water, liquid; a rate, a volume, barrels, days
    (
        TM,
        1,
        F1.replace("Sm3/d oil", "Sm3/d liquid"),
        "fact",
        "named liquid, but in E1 it is the oil",
    ),
    (TM, 1, F1.replace("Sm3/d oil", "Sm3/d of water"), "fact", "named water"),
    (TM, 1, F1.replace("123 258 Sm3 oil", "123 258 Sm3 of water"), "fact", "in E3 it is the oil"),
    (TM, 1, F1.replace("123 258 Sm3", "123 258 barrels"), "fact", "written in barrels"),
    (TM, 1, F1.replace("81 days", "81 months"), "fact", "written in months"),
    (TM, 5, F5.replace("Sm3 P50", "Sm3 per month P50"), "inference", "a rate per month or year"),
    (TM, 5, F5.replace("359 007 Sm3 P50", "374 234 Sm3 P50"), "inference", "named P50"),
    (TM, 5, F5.replace("359 007 Sm3 P50", "302 158 Sm3 P50"), "inference", "exponential model"),
    # the misreading the time machine refutes
    (
        TM,
        1,
        F1.replace(
            "downhole gauge dead since 2010-10-12", "reservoir pressure collapsed on 2010-10-12"
        ),
        "fact",
        "a dead gauge, not a pressure drop",
    ),
    (TM, 1, F1.replace("since 2010-10-12", "since 2010-10-11"), "fact", "runs from 2010-10-12"),
    # direction, the date a change belongs to, which way a change ran
    (TM, 2, F2.replace("fell 55%", "was 55% higher"), "inference", "written as a rise"),
    (TM, 2, F2.replace("fell 55%", "went up 55%"), "inference", "written as a rise"),
    (
        TM,
        2,
        F2.replace("around 2010-05-02", "after 2010-06-04"),
        "inference",
        "belongs to 2010-05-02",
    ),
    (TM, 2, F2.replace("around 2010-05-02", "in June 2010"), "inference", "belongs to 2010-05-02"),
    (
        TM,
        2,
        F2.replace("rising from 3% to 57%", "falling from 57% to 3%"),
        "inference",
        "wrong way",
    ),
    (TM, 2, F2.replace("a gradual shift", "an abrupt step down"), "inference", "gradual shift"),
    # causes
    (
        TM,
        2,
        "Water breakthrough cut F-12’s oil rate 55% around 2010-05-02 to 1 888 Sm3/d, with water "
        "cut rising from 3% to 57% and liquid holding while choke narrowed.",
        "fact",
        "states a cause",
    ),
    # durations the plan did not give
    (PLAN, 1, BREAKTHROUGH.replace("about 2 years", "about 6 years"), "inference", "6 years"),
    (
        PLAN,
        3,
        None,  # filled below: the change across given the between-periods number
        "inference",
        "called the change across",
    ),
]


def swapped_across(run: dict) -> str:
    claim = run["result"]["findings"][2]["claim"]
    assert "72%" in claim and "22%" in claim, claim
    return claim.replace("72%", "XX").replace("22%", "72%").replace("XX", "22%")


@pytest.mark.parametrize("run_id, index, claim, confidence, why", CAUGHT)
def test_a_wrong_meaning_is_caught(run_id, index, claim, confidence, why) -> None:
    run, log = load(run_id)
    claim = claim or swapped_across(run)
    result = check_edits(run["question"], run["result"], log, [(index, claim, confidence)])
    assert not result["accepted"], claim
    assert any(why in p for p in result["problems"]), result["problems"]


SUMMARY = (
    "F-12 needs attention most: it is on alarm with a dead downhole gauge and water taking "
    "capacity. F-14 is on caution with rising water cut. Field totals have not exceeded the "
    "platform’s design limits for liquid, oil or water cut. {last}"
)


@pytest.mark.parametrize(
    "last, why",
    [
        # a hedge must govern what it hedges
        (
            "F-12 will produce 359 thousand Sm3 oil in the next 12 months, "
            "as engineers may expect.",
            "has warnings",
        ),
        ("F-12 needs attention because its gauge died, which could be costly.", "cause as certain"),
        ("Water breakthrough is why its oil fell.", "cause as certain"),
        (
            "Its downhole pressure collapsed to zero in October 2010, a reservoir problem, "
            "not a sensor fault.",
            "a dead gauge",
        ),
    ],
)
def test_a_wrong_summary_is_caught(last: str, why: str) -> None:
    run, log = load(TM)
    result = check_edits(run["question"], run["result"], log, [], SUMMARY.format(last=last))
    assert not result["accepted"]
    assert any(why in p for p in result["problems"]), result["problems"]


PASS = [
    (TM, 1, F1, "fact"),
    (TM, 2, F2, "inference"),
    (TM, 5, F5, "inference"),
    (TM, 5, F5.replace("469 712 Sm3", "about 470,000 Sm3"), "inference"),
    (TM, 1, F1.replace("downhole gauge dead", "BHP gauge read 0"), "fact"),
    (PLAN, 1, BREAKTHROUGH, "inference"),
]


@pytest.mark.parametrize("run_id, index, claim, confidence", PASS)
def test_the_true_phrasing_still_passes(run_id, index, claim, confidence) -> None:
    run, log = load(run_id)
    result = check_edits(run["question"], run["result"], log, [(index, claim, confidence)])
    assert result["accepted"], result["problems"]


def test_a_summary_that_denies_the_collapse_passes() -> None:
    run, log = load(TM)
    last = (
        "Its apparent pressure collapse in October 2010 is a gauge failure, not the reservoir; "
        "the forecast for F-12 suggests about 359 thousand Sm3 oil in the next 12 months."
    )
    result = check_edits(run["question"], run["result"], log, [], SUMMARY.format(last=last))
    assert result["accepted"], result["problems"]


# The outside review (Oct 28): edits of the time machine's answer (0d95b4d9cf46, as of
# 2010-12-31) that "Try to fool the checker" accepted. Each now comes back.
TM2 = "0d95b4d9cf46"  # 1: F-12's gauge; 2: its water cut and oil; 4: F-14; 5: limits; 6: forecast


def edited(index: int, old: str, new: str) -> tuple[str, int, str | None, str]:
    run, _ = load(TM2)
    claim = run["result"]["findings"][index - 1]["claim"]
    assert old in claim, (index, old)
    return TM2, index, claim.replace(old, new), run["result"]["findings"][index - 1]["confidence"]


REVIEW = [
    (*edited(1, "since 2010-10-12", "since 2010-10-11"), "runs from 2010-10-12"),
    (*edited(2, "has risen to", "has fallen to"), "reached by a fall"),
    (*edited(6, "324,000 to 470,000", "470,000 to 324,000"), "high to low"),
    (*edited(4, "its oil rate is 34% lower", "F-12's is 34% lower"), "only for F-14"),
    (
        *edited(
            5,
            "with maximum water cut 67% below the 80% limit, but exceeded water injection "
            "capacity of 16,000 Sm3/d in 3 months up to 17,206 Sm3/d",
            "and never exceeded water injection capacity of 16,000 Sm3/d",
        ),
        "water injection capacity",
    ),
    (*edited(5, "67% below the 80% limit", "67% under the 80% limit"), "difference from the limit"),
    (*edited(2, "about 69%", "about 0.69%"), "a share of 0.0069"),
    (*edited(2, "its oil rate is 55% lower than before", "its oil rate is 2,010 Sm3/d"), "2,010"),
    (*edited(2, "about 69%", "about two thirds"), "number in words"),
    (*edited(2, "its oil rate is 55% lower than before", "its oil is half what it was"), "words"),
]


@pytest.mark.parametrize("run_id, index, claim, confidence, why", REVIEW)
def test_the_outside_reviews_edits_are_caught(run_id, index, claim, confidence, why) -> None:
    run, log = load(run_id)
    result = check_edits(run["question"], run["result"], log, [(index, claim, confidence)])
    assert not result["accepted"], claim
    assert any(why in p for p in result["problems"]), result["problems"]


def test_the_outside_reviews_summary_edits_are_caught() -> None:
    run, log = load(TM2)
    summary = run["result"]["summary"]
    for edit, why in [
        (summary.replace("324,000 to 470,000", "470,000 to 324,000"), "high to low"),
        (
            summary.replace(
                "but exceeded water injection capacity in 3 months",
                "and never exceeded water injection capacity",
            ),
            "water injection capacity",
        ),
    ]:
        result = check_edits(run["question"], run["result"], log, [], edit)
        assert not result["accepted"]
        assert any(why in p for p in result["problems"]), result["problems"]


def declared(index: int, claim: str, numbers: list[tuple[str, float, str]]) -> list[str]:
    """The checker's problems for finding `index` rewritten with these declarations, as a
    careful reader (or the model) would declare them."""
    from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission

    run, log = load(TM2)
    findings = [Finding.model_validate(f) for f in run["result"]["findings"]]
    findings[index - 1] = findings[index - 1].model_copy(
        update={
            "claim": claim,
            "numbers": [DeclaredNumber(text=t, value=v, evidence_id=e) for t, v, e in numbers],
        }
    )
    summary_numbers = [DeclaredNumber.model_validate(n) for n in run["result"]["summary_numbers"]]
    submission = Submission(
        summary=run["result"]["summary"], findings=findings, summary_numbers=summary_numbers
    )
    problems = check_submission(submission, log, run["question"])
    return [p for p in problems if p.startswith(f"Finding {index}")]


@pytest.mark.parametrize(
    "index, claim, numbers, why",
    [
        # E16's 2010-11 value given to 2010-12, the date first
        (2, "In 2010-12 F-12's oil was 44,468 Sm3.", [("44,468", 44468, "E16")], "2010-12 row"),
        (2, "F-12's water cut has risen to about 0.69%.", [("0.69%", 0.69, "E2")], "0.0069"),
        (
            2,
            "F-12's oil rate is 1,551 thousand Sm3/d.",
            [("1,551 thousand", 1551, "E2")],
            "drop the scale word",
        ),
        (2, "F-12's oil rate is now 2,010 Sm3/d.", [("2,010", 2010, "E7")], "years"),
        (
            1,
            "F-12's downhole gauge last read 253.9 psi on 2010-10-11.",
            [("253.9", 253.9, "E5")],
            "written in psi",
        ),
        (3, "A bridge plug was set at 3,210 ft on 2010-09-23.", [("3,210", 3210, "E9")], "feet"),
    ],
)
def test_a_declared_wrong_scale_unit_or_row_is_caught(index, claim, numbers, why) -> None:
    problems = declared(index, claim, numbers)
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize(
    "index, claim, numbers",
    [
        (2, "In 2010-12 F-12's oil was 45,218 Sm3.", [("45,218", 45218, "E16")]),
        (2, "F-12's water cut has risen to about 69%.", [("69%", 0.6939, "E2")]),
        (2, "F-12's oil rate is 1,551 Sm3/d.", [("1,551", 1551, "E2")]),
        (2, "F-12's water cut rose in the first half of 2010.", []),
        (1, "F-12's downhole gauge last read 253.9 bar on 2010-10-11.", [("253.9", 253.9, "E5")]),
        (3, "A bridge plug was set at 3,210 m on 2010-09-23.", [("3,210", 3210, "E9")]),
        (
            5,
            "The field's maximum water cut was 67%, under the 80% limit.",
            [("67%", 0.6715, "E3"), ("80%", 0.8, "E3")],
        ),
        (
            2,
            "F-12's oil rate was 1,551 Sm3/d, down 16.5% over the last 90 days.",
            [("1,551", 1551, "E2"), ("16.5%", -0.165, "E2")],
        ),
    ],
)
def test_the_true_declared_phrasing_passes(index, claim, numbers) -> None:
    assert declared(index, claim, numbers) == []
