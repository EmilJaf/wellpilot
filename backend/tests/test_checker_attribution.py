"""Numbers in the evidence but not the ones the text says: the second panel of judges found
these passing as "verified" (Oct 27 evening). Each rule judges only what the evidence states
outright, so the correct versions next to each wrong one must still pass.

Evidence: run efcae2e67ec3's first answer (as of 2010-12-31). E1 field_screen (every well,
no well argument), E4 detect_events for F-12, E5 for F-14, E10 forecast_production F-12.
"""

import json

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.evidence import EvidenceLog

AS_OF_2010 = "efcae2e67ec3-1"


def corpus_log(name: str) -> EvidenceLog:
    return EvidenceLog.restore(json.loads((CORPUS_DIR / f"{name}.json").read_text())["evidence"])


def d(text: str, value: float, eid: str) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


def check(claim: str, ids: list[str], numbers: list) -> list[str]:
    log = corpus_log(AS_OF_2010)
    finding = Finding(claim=claim, evidence_ids=ids, confidence="fact", numbers=numbers)
    return check_submission(Submission(summary="Done.", findings=[finding]), log)


def test_a_number_given_to_another_well_is_caught() -> None:
    wrong = check("F-14 produced 1 551 Sm3/d of oil.", ["E1"], [d("1 551", 1551, "E1")])
    right = check("F-12 produced 1 551 Sm3/d of oil.", ["E1"], [d("1 551", 1551, "E1")])
    both = check(
        "F-12 and F-14 produced 1 551 and 2 175 Sm3/d.",
        ["E1"],
        [d("1 551", 1551, "E1"), d("2 175", 2175, "E1")],
    )

    assert wrong and "only for F-12" in wrong[0]
    assert right == [] and both == []


def test_a_date_given_to_another_well_is_caught_through_field_wide_evidence() -> None:
    wrong = check("F-14's downhole gauge has read 0 since 2010-10-12.", ["E4", "E1"], [])
    right = check("F-12's downhole gauge has read 0 since 2010-10-12.", ["E4", "E1"], [])

    assert wrong and any("2010-10-12 (evidence has it only for F-12)" in p for p in wrong)
    assert right == []


def test_a_number_named_as_another_quantity_is_caught() -> None:
    wrong = check("F-12's water cut on 2010-12-31 was 0.98.", ["E1"], [d("0.98", 0.9846, "E1")])
    right = check("F-12's uptime on 2010-12-31 was 0.98.", ["E1"], [d("0.98", 0.9846, "E1")])
    after = check("F-12 ran at 0.69 water cut.", ["E1"], [d("0.69", 0.6939, "E1")])
    # Only filler words bind a name to a number: "to", "limit", "producing" end its reach.
    unbound = check(
        "F-12's water cut stayed below its limit, with 0.98 uptime.",
        ["E1"],
        [d("0.98", 0.9846, "E1")],
    )

    assert wrong and "named water cut, but in E1 it is the uptime" in wrong[0]
    assert right == [] and after == [] and unbound == []


def test_a_fall_written_as_a_rise_is_caught() -> None:
    rose = check("F-12's oil rate rose 55% around 2010-05-02.", ["E4"], [d("55%", 0.55, "E4")])
    fell = check("F-12's oil rate fell 55% around 2010-05-02.", ["E4"], [d("55%", 0.55, "E4")])

    assert rose and "written as a rise" in rose[0]
    assert fell == []


def test_from_and_to_the_wrong_way_round_is_caught() -> None:
    numbers = [d("1 888", 1888, "E4"), d("4 207", 4207, "E4")]
    wrong = check("F-12's weekly oil rate went from 1 888 to 4 207 Sm3/d.", ["E4"], numbers)
    right = check("F-12's weekly oil rate went from 4 207 to 1 888 Sm3/d.", ["E4"], numbers[::-1])

    assert wrong and "runs the wrong way" in wrong[0]
    assert right == []


def test_a_number_tied_to_another_events_date_is_caught() -> None:
    wrong = check("F-12's oil rate stepped down 55% on 2010-06-04.", ["E4"], [d("55%", 0.55, "E4")])
    right = check("F-12's oil rate stepped down 55% on 2010-05-02.", ["E4"], [d("55%", 0.55, "E4")])
    crossing = check(
        "F-12's water cut first exceeded 0.5 on 2010-06-04.", ["E4"], [d("0.5", 0.5, "E4")]
    )

    assert wrong and "belongs to 2010-05-02" in wrong[0]
    assert right == [] and crossing == []


def test_an_undeclared_number_the_evidence_holds_comes_back_as_a_declaration() -> None:
    """Run efcae2e67ec3 deleted a true "21 months after first oil" when told it was
    undeclared; the rejection now says how to declare it."""
    problems = check(
        "F-14's water cut first exceeded 0.5 on 2010-04-05, 21 months after first oil.",
        ["E5"],
        [d("0.5", 0.5, "E5")],
    )

    assert problems and 'declare them as {"text": "21", "value": ' in problems[0]
    assert '"evidence_id": "E5"}' in problems[0]


def test_a_findings_detail_is_checked_like_its_claim() -> None:
    claim = "F-12 needs attention first: its downhole gauge is dead."
    right = Finding(
        claim=claim,
        detail="The gauge has read 0 on 81 days since 2010-10-12.",
        evidence_ids=["E4"],
        confidence="fact",
        numbers=[d("81", 81, "E4")],
    )
    wrong = right.model_copy(update={"detail": "The gauge has read 0 on 91 days since 2010-10-12."})
    log = corpus_log(AS_OF_2010)

    assert check_submission(Submission(summary="Done.", findings=[right]), log) == []
    problems = check_submission(
        Submission(
            summary="Done.", findings=[wrong.model_copy(update={"numbers": [d("91", 91, "E4")]})]
        ),
        log,
    )
    assert problems and "'91' = 91 is not in E4" in problems[0]


def test_several_calculations_in_one_call_are_checked_one_by_one() -> None:
    log = corpus_log(AS_OF_2010)
    batch = log.add(
        "calculate",
        {"expressions": ["4207 - 1888", "1551 / 2175"]},
        {
            "results": [
                {"expression": "4207 - 1888", "value": 2319.0},
                {"expression": "1551 / 2175", "value": 0.7131},
            ]
        },
        lambda: None,
    ).id
    numbers = [d("2 319", 2319, batch), d("0.71", 0.7131, batch)]
    finding = Finding(
        claim="F-12's weekly rate fell by 2 319 Sm3/d; it makes 0.71 of F-14's rate.",
        evidence_ids=[batch],
        confidence="fact",
        numbers=numbers,
    )

    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []
