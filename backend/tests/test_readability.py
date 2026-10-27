"""Readability: claims and a summary a reader takes in at a glance. Asked once per run by
the loop (test_agent.py), never a reason to lose verification."""

from wellpilot.agent.checker import (
    DeclaredNumber,
    Finding,
    Submission,
    check_submission,
    readability,
)
from wellpilot.agent.evidence import EvidenceLog


def d(text: str, value: float) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id="E1")


def problems(claim: str, numbers: list[DeclaredNumber], summary="Done.", summary_numbers=()):
    finding = Finding(claim=claim, evidence_ids=["E1"], confidence="inference", numbers=numbers)
    return readability(
        Submission(summary=summary, findings=[finding], summary_numbers=list(summary_numbers))
    )


def test_a_long_number_is_asked_rounded_with_a_suggestion() -> None:
    """Run 8aef8d548380's finding 5: '359 007 Sm3 P50 with calibrated band 324 192 to
    469 712 Sm3'."""
    found = problems(
        "F-12 makes 359 007 Sm3 P50 in 2011, band 324 192 to 469 712 Sm3.",
        [d("359 007", 359007), d("324 192", 324192), d("469 712", 469712)],
    )

    assert len(found) == 2
    assert "'359 007' (about 359,000)" in found[0] and "(about 470,000)" in found[0]
    assert "P50" in found[1] and "central estimate" in found[1]


def test_numbers_a_reader_takes_in_pass() -> None:
    assert (
        problems(
            "F-12 made 1 551 Sm3/d at a 72.3% choke, 359k Sm3 in 2011, 1.13 million in all.",
            [d("1 551", 1551), d("72.3%", 72.3), d("359k", 359007), d("1.13 million", 1133053)],
        )
        + problems(
            "The plan's 9.000 Sm3/d, 13 000 Sm3/d of liquid.",
            [d("9.000", 9000), d("13 000", 13000)],
        )
        == []
    )


def test_small_decimals_count_too_and_quotes_stay_as_written() -> None:
    found = problems(
        'Water cut 0.6939; the PUD says "Væskekapasitet 13000 og 11 438 som topp".',
        [d("0.6939", 0.6939), d("11 438", 11438)],
    )

    assert len(found) == 1 and "'0.6939' (about 0.694)" in found[0] and "11 438" not in found[0]


def test_the_summary_is_asked_too() -> None:
    found = problems(
        "F-12 leads.",
        [],
        summary="F-12 makes about 359 007 Sm3 next year.",
        summary_numbers=[d("359 007", 359007)],
    )

    assert found == [
        "Summary: round '359 007' (about 359,000) in the summary to three significant digits; "
        "a reader takes in three. Declare the rounded text with the same value and evidence id."
    ]


def test_the_suggested_rounding_passes_the_checker() -> None:
    log = EvidenceLog()
    log.add(
        "forecast_production", {"well": "F-12"}, {"p50": 359007.0, "p10": 469712.0}, lambda: None
    )
    finding = Finding(
        claim="F-12 makes about 359,000 Sm3 next year, up to 470,000.",
        evidence_ids=["E1"],
        confidence="inference",
        numbers=[d("359,000", 359007), d("470,000", 469712)],
    )
    submission = Submission(summary="Done.", findings=[finding], summary_numbers=[])

    assert check_submission(submission, log) == []
    assert readability(submission) == []


def test_a_claim_with_many_numbers_is_asked_to_move_some_to_its_detail() -> None:
    """Eval run e8f9b5d63312 wrote nine numbers in one claim; the reader got the point from
    two. Dates and quotes do not count."""
    dense = (
        "At the time of the logs a plug at 3150 m would have shut off ~97% to 100% of water and "
        "~73% to 81% of oil, while a plug at 3210 m would have shut off only ~33% to 38% of "
        "water and ~1% of oil."
    )
    plain = "The 2014 plug at 3150 m sat below ~97% of the 2010-09-22 water, the 2010 one ~33%."

    found = problems(dense, []) + problems(plain, [])

    assert len(found) == 1 and found[0].startswith("Finding 1: the claim holds 9 numbers")
