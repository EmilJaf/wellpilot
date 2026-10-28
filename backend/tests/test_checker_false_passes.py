"""False passes: wrong or unsupported answers the checker must reject.

The other half of test_checker_false_rejections.py: every rule that lets more answers
through gets a row here showing what it must still stop.
"""

import json

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.evidence import EvidenceLog


def _tool() -> None:
    """Stands in for a tool function."""


def corpus_log(name: str) -> EvidenceLog:
    case = json.loads((CORPUS_DIR / f"{name}.json").read_text())
    return EvidenceLog.restore(case["evidence"])


def add(log: EvidenceLog, tool: str, arguments: dict, result) -> str:
    return log.add(tool, arguments, result, _tool).id


def d(text: str, value: float, eid: str) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


def check(claim: str, ids: list[str], numbers: list, log: EvidenceLog, question="") -> list[str]:
    finding = Finding(claim=claim, evidence_ids=ids, confidence="fact", numbers=numbers)
    return check_submission(Submission(summary="Done.", findings=[finding]), log, question)


F12 = "6fe2c784147c-1"  # E5 detect_events for F-12, E9 a clean fit


def test_a_duration_that_is_not_a_window_still_needs_evidence() -> None:
    log = corpus_log(F12)

    assert check("F-12 was shut in for 30 days of maintenance.", ["E5"], [], log)
    assert check("A 45-day shut-in followed.", ["E5"], [], log)


def test_a_small_integer_declared_with_another_value_is_caught() -> None:
    log = corpus_log(F12)

    problems = check("F-12 is one of 3 producers.", ["E5"], [d("3", 4, "E5")], log)

    assert problems and "does not show its declared value" in problems[0]


def test_only_whole_numbers_from_the_question_are_free() -> None:
    log = corpus_log(F12)
    question = "Did F-12 lose 72.5% of its rate after 2010?"

    assert check("F-12 lost 72.5% of its rate.", ["E5"], [], log, question)


def test_a_year_needs_evidence_of_that_year() -> None:
    log = corpus_log(F12)

    assert check("The gauge failed in 2003.", ["E5"], [], log)


def test_a_literal_echoed_from_the_select_list_is_still_not_evidence() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "run_sql",
        {"sql": "SELECT DATE '2013-05-17' AS d, 4321 AS n FROM production_daily LIMIT 1"},
        {"columns": ["d", "n"], "rows": [["2013-05-17", 4321]]},
    )

    problems = check("On 2013-05-17 the rate was 4 321 Sm3/d.", [eid], [d("4 321", 4321, eid)], log)

    assert any("2013-05-17" in p for p in problems)
    assert any("4 321" in p for p in problems)


def test_a_date_written_in_words_is_checked_as_a_date() -> None:
    log = corpus_log(F12)  # E5 holds 2010-07-18 and 2010-10-12, nothing in March 2010

    assert check("The oil rate collapsed in March 2010.", ["E5"], [], log)
    assert check("The gauge failed on 5 Jun 2010.", ["E5"], [], log)
    assert check("The gauge failed on October 13, 2010.", ["E5"], [], log)
    assert check("The plan dates from March 2005.", ["E5"], [], log)  # the PUD is Feb 2005


def test_a_scaled_unit_must_still_match() -> None:
    log = EvidenceLog()
    eid = add(
        log, "list_wells", {}, {"columns": ["well", "cum_oil_sm3"], "rows": [["F-12", 4579610]]}
    )

    assert check("F-12 produced 5.58 MSm3.", [eid], [d("5.58 MSm3", 5580000, eid)], log)
    assert check("F-12 produced 4.58 kSm3.", [eid], [d("4.58 kSm3", 4580, eid)], log)


def test_calculate_constants_are_conversions_not_any_number() -> None:
    log = corpus_log(F12)
    for expression in ("1133053 / 1234567", "2010 - 2003"):
        calc = add(log, "calculate", {"expression": expression}, {"value": 0.918})
        problems = check("The ratio is 0.918.", [calc], [d("0.918", 0.918, calc)], log)
        assert any("calculates from" in p for p in problems), expression


def test_absurd_numbers_are_rejected_not_crashed_on() -> None:
    log = corpus_log(F12)
    for text in ("1e400", "9" * 400):
        assert check(f"The rate was {text} Sm3/d.", ["E5"], [], log)
        assert check(f"The rate was {text} Sm3/d.", ["E5"], [d(text, 1146, "E5")], log)
    assert check("The rate was 1e400 Sm3/d.", ["E5"], None, log)  # legacy path


def test_round_numbers_are_good_to_two_significant_digits() -> None:
    log = EvidenceLog()
    eid = add(log, "detect_events", {"well": "F-14"}, {"before_mean": 3593.1, "rate": 1480.0})

    assert check("The rate was about 4000 Sm3/d.", [eid], [d("4000", 3593.1, eid)], log)
    assert check("The rate was 1,000 Sm3/d.", [eid], [d("1,000", 1000, eid)], log)
    ok = EvidenceLog()
    oid = add(ok, "detect_events", {"well": "F-14"}, {"before_mean": 4049.93})
    assert check("The rate was about 4000 Sm3/d.", [oid], [d("4000", 4049.93, oid)], ok) == []


def test_a_full_declared_value_pins_the_scale() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "production_monthly",
        {"well": "F-12"},
        {"water_cut": 0.31, "choke_pct": 72.3, "after_mean": 1145.72, "b": 1.13},
    )

    assert check("Water cut reached 72%.", [eid], [d("72%", 0.72, eid)], log)
    assert check("Cumulative oil 1146k Sm3.", [eid], [d("1146k", 1146000, eid)], log)
    assert check("Cumulative oil 1.13 million Sm3.", [eid], [d("1.13 million", 1130000, eid)], log)
    # Still fine: the short value leaves the scale open, the full one matches its scale.
    assert check("Water cut reached 31%.", [eid], [d("31%", 0.31, eid)], log) == []
    assert check("Choke at 72%.", [eid], [d("72%", 72, eid)], log) == []


def test_a_translation_not_in_the_evidence_is_the_findings_own_words() -> None:
    log = corpus_log(F12)
    eid = next(
        e.id for e in log if "Maksimum vannkutt 0,80" in json.dumps(e.result, ensure_ascii=False)
    )
    invented = (
        'The PUD sets "Maksimum vannkutt 0,80" [F-12 exceeded it because the injectors drove '
        "water into the producers]."
    )

    problems = check(invented, [eid], [], log)

    assert problems and "states a cause" in problems[0]


def test_a_translation_inside_the_quote_marks_is_still_the_findings_own_words() -> None:
    log = corpus_log(F12)
    eid = next(
        e.id for e in log if "Maksimum vannkutt 0,80" in json.dumps(e.result, ensure_ascii=False)
    )
    invented = (
        'The PUD sets "Maksimum vannkutt 0,80 [F-12 exceeded it because the injectors drove '
        'water into the producers]".'
    )

    problems = check(invented, [eid], [], log)

    assert problems and "states a cause" in problems[0]


def summary_problems(summary: str) -> list[str]:
    log = corpus_log(F12)
    finding = Finding(claim="The rate fell.", evidence_ids=["E5"], confidence="fact", numbers=[])
    return check_submission(Submission(summary=summary, findings=[finding]), log)


def test_a_hedge_counts_only_in_its_own_sentence() -> None:
    assert summary_problems("In May 2010 the rate collapsed because water broke through.")
    assert summary_problems("The fit is probably exponential. The fall was caused by water.")
    assert summary_problems("The fall was most likely caused by water breakthrough.") == []


def test_common_causal_phrasings_are_causes() -> None:
    for phrase in (
        "as a result of water breakthrough",
        "attributable to water breakthrough",
        "triggered by water breakthrough",
        "owing to water breakthrough",
        "which explains the faster decline",
        "a consequence of water breakthrough",
        "and water breakthrough was the main driver",
        "which reflects water breakthrough",
        "a direct effect of water breakthrough",
    ):
        assert summary_problems(f"The rate fell, {phrase}."), phrase
    # Petroleum vocabulary and evidence references are not causes.
    assert summary_problems("F-12 produces under water drive; the fit results in E9 show it.") == []


def test_a_declared_number_does_not_cover_a_longer_one_that_contains_it() -> None:
    """Each case would pass if the declared text were matched as a plain substring."""
    log = EvidenceLog()
    eid = add(log, "tool", {}, {"values": [2, 50, 1, 12]})
    for claim, declared in [
        ("Rate fell 2.5x in 2 years.", "2"),
        ("Water cut 0.50 after 50 days.", "50"),
        ("It made 12 500 Sm3 in 1 year.", "1"),
        ("Rate 1.2e+03 after 1 day.", "1"),
        ("It made 12,000 Sm3 in 12 days.", "12"),
    ]:
        value = float(declared)
        problems = check(claim, [eid], [d(declared, value, eid)], log)
        assert any("undeclared number" in p for p in problems), claim


def test_an_undeclared_number_is_named_as_written() -> None:
    """Run fcfb4188de3b: "0,80" left undeclared was reported as "80, 0.80", and the model,
    which had declared "0.80", could not tell what was meant."""
    log = corpus_log(F12)

    problems = check(
        "The rise to 0.80 passed the plan's Maksimum vannkutt 0,85 limit.",
        ["E5"],
        [d("0.80", 0.8, "E5")],
        log,
    )
    # The same value written again in another form was declared once; that is enough.
    same_value = check(
        "The rise to 0.80 reached the plan's Maksimum vannkutt 0,80 limit.",
        ["E5"],
        [d("0.80", 0.8, "E5")],
        log,
    )

    assert problems and "undeclared number(s) 0,85." in problems[0]
    assert same_value == []


DEMO = "fcfb4188de3b-2"  # E2 detect_events for F-12; E11 calculates 7.89 from the 0.019 fit


def summary_check(summary: str, numbers: list, log: EvidenceLog) -> list[str]:
    finding = Finding(claim="F-12 declined.", evidence_ids=["E2"], confidence="fact", numbers=[])
    submission = Submission(summary=summary, findings=[finding], summary_numbers=numbers)
    return check_submission(submission, log)


def test_a_summary_number_from_a_poor_fit_must_be_stated_as_uncertain() -> None:
    log = corpus_log(DEMO)
    di = [d("0.0566", 0.05655, "E8")]  # the plateau fit's own decline rate

    problems = summary_check("F-12 declined at 0.0566 per year before 2010.", di, log)
    hedged = summary_check("The fit suggests F-12 declined at 0.0566 per year.", di, log)

    assert problems and "E8, which has warnings" in problems[0]
    assert hedged == []
    # The hedge must be in the number's own sentence.
    assert summary_check("The fit is likely poor. It declined at 0.0566 per year.", di, log)
    # A clause after a semicolon is the same sentence (run 83f441296bb7's first answer).
    assert not summary_check(
        "It declined at 0.0566 per year; the fit suggests so, with low confidence.", di, log
    )


def test_a_comparison_built_on_a_poor_fit_stays_out_of_the_summary() -> None:
    """The demo's "7.89 times faster" divides the post-breakthrough decline rate by the
    plateau fit's (r2 0.019): a plateau has no decline rate, so the ratio is noise however
    it is hedged (the reservoir engineer judge's point). A finding may keep it as an
    inference."""
    log = corpus_log(DEMO)
    ratio = [d("7.89", 7.89, "E11")]

    for summary in (
        "F-12 then declined about 7.89 times faster.",
        "The fit suggests F-12 declined about 7.89 times faster.",
    ):
        problems = summary_check(summary, ratio, log)
        assert problems and "Leave it out of the summary" in problems[0]
    finding = Finding(
        claim="The fit suggests the decline sped up about 7.89 times.",
        evidence_ids=["E11", "E8", "E9"],
        confidence="inference",
        numbers=ratio,
    )
    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []


def test_a_finding_about_a_well_needs_evidence_about_that_well() -> None:
    log = corpus_log(DEMO)
    numbers = [d("72%", -0.7171, "E2")]

    problems = check("F-14's oil rate fell 72% around 2010-07-18.", ["E2"], numbers, log)
    own_well = check("F-12's oil rate fell 72% around 2010-07-18.", ["E2"], numbers, log)
    field_wide = check("F-14 and F-12 were both listed.", ["E1"], [], log)

    assert problems and any("names F-14" in p for p in problems)
    assert any("only for F-12" in p for p in problems)  # the number is F-12's, too
    assert own_well == []
    assert field_wide == []  # list_wells is about every well


def test_a_declared_value_written_again_must_read_the_same() -> None:
    """ "94 %" declared covers "94%" written again, but "0.100" is not 100 (no decimal comma
    makes it Norwegian), and 95% is not 94 %."""
    log = corpus_log(F12)
    drop = [d("72 %", 72, "E5")]

    assert check("The rate fell 72 % [72% in the summary].", ["E5"], drop, log) == []
    later = check("The rate fell 72 %, later 75%.", ["E5"], drop, log)
    norwegian = check("It made 0.072 Sm3, then 72 % less.", ["E5"], drop, log)

    rounded = check(
        "Each made 1.77% of the oil, less than 2%.", ["E5"], [d("1.77%", 0.0177, "E5")], log
    )

    plateau = [d("9 000", 9000, "E5")]
    translated = check("A plateau of 9 000 Sm3/d [9.000 Sm3/d in the plan].", ["E5"], plateau, log)

    assert any("undeclared number(s) 75%" in p for p in later)
    assert not any("undeclared" in p for p in translated)  # the plan's 9.000 is 9 000
    assert any("undeclared number(s) 2%" in p for p in rounded)  # a bound, not 1.77% rounded
    assert any("undeclared number(s) 0.072" in p for p in norwegian)


def test_document_references_are_names_not_numbers() -> None:
    log = corpus_log(F12)
    claim = (
        "The plan's Figure 4.2 and Tabell 4.5 (section 4.5.1, § 4.5.2, pages 23-25) show "
        "the profile."
    )

    assert check(claim, ["E5"], [], log) == []
    assert check("The plan shows 4.2 Sm3/d.", ["E5"], [], log)  # a bare value still counts


MONTHLY = {
    "columns": ["month", "water_cut", "oil_sm3"],
    "rows": [["2009-06", 0.0197, 120000], ["2012-06", 0.85, 30000]],
}


def test_a_table_value_tied_to_a_date_must_be_in_that_dates_row() -> None:
    """The AI judge's probe: 0.85 as F-12's water cut in 2009-06 (it was 0.0197) used to
    pass because 0.85 is somewhere in the monthly table."""
    log = EvidenceLog()
    eid = add(log, "production_monthly", {"well": "F-12"}, MONTHLY)
    wrong = [d("0.85", 0.85, eid)]

    problems = check("F-12's water cut was 0.85 in 2009-06.", [eid], wrong, log)
    right = check("F-12's water cut was 0.85 in 2012-06.", [eid], wrong, log)

    assert problems and "not in its 2009-06 row" in problems[0]
    assert right == []


def test_a_row_tie_is_judged_only_when_it_is_unambiguous() -> None:
    log = EvidenceLog()
    eid = add(log, "production_monthly", {"well": "F-12"}, MONTHLY)
    cut = [d("0.85", 0.85, eid)]

    # A comma ends the number's clause; the date belongs to the next one.
    assert check("Water cut reached 0.85, in 2009-06 it had been low.", [eid], cut, log) == []
    # A range of dates is no one row; a day is not a month's row; a missing month is not judged.
    assert check("Water cut was 0.85 in 2009-06 to 2012-06.", [eid], cut, log) == []
    day = check("Water cut was 0.85 on 2009-06-15.", [eid], cut, log)
    assert not any("row" in p for p in day)  # (the day itself is not in E1: another rule)
    missing = check("Water cut was 0.85 in 2011-03.", [eid], cut, log)
    assert not any("row" in p for p in missing)


def test_water_drove_is_a_cause_water_drive_is_not() -> None:
    """Run 7b95c36af427's fact "Water drove the fall: ..." passed: the exception for the
    recovery mechanism ("water drive") covered the verb too."""
    from wellpilot.agent.checker import CAUSAL

    assert CAUSAL.search("Water drove the fall: water cut was 51%.")
    assert CAUSAL.search("Water driving the decline.")
    assert not CAUSAL.search("The reservoir is under water drive.")
    assert not CAUSAL.search("A gas drive mechanism.")
