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

    assert any("runs the wrong way" in p for p in wrong)
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


def test_a_findings_detail_numbers_are_looked_up_in_its_evidence() -> None:
    """The claim declares its numbers; the detail holds values copied from the tools, which
    need no declaring but must be in the evidence the finding cites."""
    right = Finding(
        claim="F-12 needs attention first: its downhole gauge is dead.",
        detail="Gauge failure: 81 zero days from 2010-10-12, last good reading 253.9 bar.",
        evidence_ids=["E4"],
        confidence="fact",
        numbers=[],
    )
    wrong = right.model_copy(update={"detail": "Gauge failure: 91 zero days from 2010-10-12."})
    misdated = right.model_copy(update={"detail": "Gauge failure: 81 zero days from 2010-10-21."})
    log = corpus_log(AS_OF_2010)

    def problems(finding: Finding) -> list[str]:
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert problems(right) == []
    assert "in its detail, 91 not found in E4" in problems(wrong)[0]
    assert "2010-10-21 not found" in problems(misdated)[0]


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


def test_a_level_reached_during_a_month_may_stand_in_the_next_row() -> None:
    """Run cf597aa90554 (synthetic field): "the choke was cut to 40% in 2018-11". November's
    monthly row averages the days before and after the cut (48%); December holds 40."""
    log = EvidenceLog()
    table = {
        "columns": ["month", "choke_pct", "oil_rate_per_producing_day"],
        "rows": [["2018-10", 70.0, 785.0], ["2018-11", 48.2, 512.0], ["2018-12", 40.0, 445.0]],
    }
    eid = log.add("production_monthly", {"well": "P-2"}, table, lambda: None).id

    def check(claim: str, text: str, value: float) -> list[str]:
        finding = Finding(
            claim=claim, evidence_ids=[eid], confidence="fact", numbers=[d(text, value, eid)]
        )
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert check("P-2's choke was cut to 40% in 2018-11.", "40%", 40) == []
    assert check("P-2's oil rate fell to 512 Sm3/d in 2018-11.", "512", 512) == []
    assert check("P-2's choke was at 40% in 2018-11.", "40%", 40)  # a level, not a change


def test_a_month_name_before_a_value_is_not_a_date() -> None:
    """Run 4461982df3dc's detail "Monthly p50 Jan 1363 Dec 840.53" was read as January of
    the year 1363 ("date(s) 1363-01 not found") three submissions running; the run ended
    unverified. A month in words with a real year is still a date."""
    log = EvidenceLog()
    monthly = {"columns": ["month", "p50_rate"], "rows": [["2011-01", 1363.0], ["2011-12", 840.53]]}
    eid = log.add("forecast_production", {"well": "F-12"}, monthly, lambda: None).id

    def check(claim: str) -> list[str]:
        finding = Finding(
            claim=claim,
            detail="Monthly p50 Jan 1363 Dec 840.53.",
            evidence_ids=[eid],
            confidence="inference",
            numbers=[d("1,363", 1363, eid), d("841", 840.53, eid)],
        )
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert check("F-12's rate is forecast to fall from 1,363 to 841 Sm3/d.") == []
    wrong = check("F-12's rate is forecast to fall from 1,363 to 841 Sm3/d by Dec 2013.")
    assert any("2013-12" in p for p in wrong)


def test_a_number_belongs_to_the_well_named_before_it_not_after() -> None:
    """Run 336805e71b52 (synthetic field): the summary's "water cut stayed near 0.9029, and
    field replacement fell ... after injector I-2 stopped" gave P-1's water cut to I-2, the
    only well its sentence names. P-1, the subject of the sentence before, owns it; the run
    was sent back three times and ended unverified."""
    log = EvidenceLog()
    flood = {
        "by_year": {"columns": ["year", "water_cut"], "rows": [[2019, 0.9012], [2020, 0.9029]]}
    }
    wf = log.add("waterflood", {"well": "P-1"}, flood, lambda: None).id
    stopped = {"wells": [{"well": "I-2", "issues": [{"kind": "stopped", "since": "2019-08-31"}]}]}
    screen = log.add("field_screen", {}, stopped, lambda: None).id
    finding = Finding(
        claim="P-1's water cut stayed near 0.90.",
        evidence_ids=[wf],
        confidence="fact",
        numbers=[d("0.90", 0.9029, wf)],
    )

    def check(summary: str) -> list[str]:
        submission = Submission(
            summary=summary, findings=[finding], summary_numbers=[d("0.9029", 0.9029, wf)]
        )
        return check_submission(submission, log)

    assert screen
    assert (
        check(
            "P-1's oil decline sped up in late 2019. Its water cut stayed near 0.9029, and "
            "field replacement fell after injector I-2 stopped on 2019-08-31."
        )
        == []
    )
    wrong = check("Water cut stayed near 0.9029 at I-2 before it stopped on 2019-08-31.")
    assert any("only for P-1" in p for p in wrong)


def test_a_well_the_result_names_may_be_named() -> None:
    """Run e51eada95269: F-12's shut-ins were "field-wide with F-14", cited from
    detect_events for F-12, whose shut-in record lists F-14. A well the evidence never
    names is still sent back."""
    log = EvidenceLog()
    shut = [
        {
            "kind": "shut_in",
            "well": "F-12",
            "start": "2010-08-16",
            "details": {"days": 12, "field_wide": True, "field_wide_with": ["F-14"]},
        }
    ]
    eid = log.add("detect_events", {"well": "F-12"}, shut, lambda: None).id

    def check(detail: str) -> list[str]:
        finding = Finding(
            claim="F-12 was shut in for 12 days from 2010-08-16.",
            detail=detail,
            evidence_ids=[eid],
            confidence="fact",
            numbers=[d("12 days", 12, eid)],
        )
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert check("Field-wide, with F-14.") == []
    assert any("names F-11" in p for p in check("Field-wide, with F-11."))


def test_the_declaration_hint_names_the_value_nearest_the_number_written() -> None:
    """Run e8c3374117a8 wrote "3 150 m" undeclared; its log held 3147 (a zone's bottom) and
    3150 (the plug's depth). Both round to "3 150"; the hint names 3150."""
    log = EvidenceLog()
    eid = log.add(
        "production_log",
        {"well": "F-12"},
        {"zones_m": [[3131, 3147]], "below": [{"depth_m": 3150, "water_share": 0.971}]},
        lambda: None,
    ).id
    finding = Finding(
        claim="F-12's log had 97% of the water entering below 3 150 m.",
        evidence_ids=[eid],
        confidence="fact",
        numbers=[DeclaredNumber(text="97%", value=0.971, evidence_id=eid)],
    )

    (problem,) = check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert '{"text": "3 150", "value": 3150, "evidence_id": "E1"}' in problem


def test_the_hint_for_a_percent_names_the_share_not_a_rate_near_it() -> None:
    log = EvidenceLog()
    eid = log.add(
        "production_monthly",
        {"well": "F-12"},
        {"choke_pct": 72.3, "water_cut": 0.7213},
        lambda: None,
    ).id
    finding = Finding(
        claim="F-12's water cut was 72%.", evidence_ids=[eid], confidence="fact", numbers=[]
    )

    (problem,) = check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert '"value": 0.7213' in problem
