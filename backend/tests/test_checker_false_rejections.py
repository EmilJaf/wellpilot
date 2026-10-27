"""False rejections: correct answers, written the way Muse Spark Open writes them, that
the checker must accept.

Seeded from the Oct 27 audit's probes. Evidence is restored from the real corpus cases, so
shapes are exactly what the tools return. Tests named *_control check that the set-up is
sound. A case marked pending is a known false rejection that a later fix will close;
strict xfail makes the suite fail once it passes, so the mark gets removed.

Every checker change adds a row here (an answer it must accept) and one to
test_checker_false_passes.py (an answer it must reject).
"""

import json

import pytest

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.corpus import CORPUS_DIR as CORPUS
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.ingest.report_facts import squash


def pending(reason: str):
    return pytest.mark.xfail(strict=True, reason=reason)


def corpus_log(name: str) -> EvidenceLog:
    case = json.loads((CORPUS / f"{name}.json").read_text())
    return EvidenceLog.restore(case["evidence"])


def _tool() -> None:
    """Stands in for a tool function."""


def add(log: EvidenceLog, tool: str, arguments: dict, result, warnings=()) -> str:
    item = log.add(tool, arguments, result, _tool)
    if warnings:
        object.__setattr__(item, "warnings", tuple(warnings))
    return item.id


def d(text: str, value: float, eid: str) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


def finding(claim, ids, numbers, confidence="fact") -> Finding:
    return Finding(claim=claim, evidence_ids=ids, confidence=confidence, numbers=numbers)


def check(*findings, summary="Done.", summary_numbers=None, log, question="") -> list[str]:
    return check_submission(
        Submission(summary=summary, findings=list(findings), summary_numbers=summary_numbers or []),
        log,
        question,
    )


def evidence_with_quote(log: EvidenceLog, quote: str) -> str:
    """Id of the first evidence whose text holds the quote (squashed)."""
    target = squash(quote)
    for item in log:
        if target in squash(json.dumps(item.result, ensure_ascii=False)):
            return item.id
    raise AssertionError(f"no evidence holds {quote!r}")


# F-12 run: E5 detect_events (2010-10-12 gauge failure, 253.85, 4050 -> 1146, 2095 days...),
# E4/E7/E8 search_reports (PUD facts), E9 clean fit (di 0.4461, r2 0.8732), E11 poor fit,
# E12-E16, E18 calculate, E17 run_sql.
F12 = "6fe2c784147c-1"
# F-14 forecast run: E2 detect_events, E6 fit, E7 forecast (P50 276078...), E9 monthly.
F14 = "4882e62d527c-1"
# Gauge-failure run with run_sql E4/E5/E6.
GAUGE = "95600d0cf26d-1"


# A. dates written in words


@pending("block 2: dates written with month names")
def test_A1_day_month_year_in_words_is_a_date_not_a_number() -> None:
    """'12 October 2010' is how an engineer writes a date. Only ISO dates are recognised,
    so 12 and 11 become 'undeclared numbers' although the prompt says dates need no
    declaration."""
    log = corpus_log(F12)
    f = finding(
        "F-12's downhole gauge failed on 12 October 2010; the last good BHP was 253.85 bar "
        "on 11 October 2010.",
        ["E5"],
        [d("253.85", 253.85, "E5")],
    )
    assert check(f, log=log) == []


@pending("block 2: dates written with month names")
def test_A1b_declaring_the_day_number_does_not_help() -> None:
    """If the model obeys the rejection and declares '11', the day is not a number in the
    evidence (dates are excluded from the number pool), so it is rejected again: a loop."""
    log = corpus_log(F12)
    f = finding(
        "The last good BHP was 253.85 bar on 11 October 2010.",
        ["E5"],
        [d("253.85", 253.85, "E5"), d("11", 11, "E5")],
    )
    assert check(f, log=log) == []


@pending("block 2: dates written with month names")
def test_A2_month_day_comma_year() -> None:
    log = corpus_log(F12)
    f = finding(
        "The oil-rate regime change came around July 18, 2010: weekly mean 4 050 to 1 146 Sm3/d.",
        ["E5"],
        [d("4 050", 4050, "E5"), d("1 146", 1146, "E5")],
    )
    assert check(f, log=log) == []


def test_A3_abbreviated_month() -> None:
    log = corpus_log(F12)
    f = finding(
        "Water cut first exceeded 0.5 on 4 Jun 2010, 27.69 months after first oil.",
        ["E5"],
        [d("0.5", 0.5, "E5"), d("27.69", 27.69, "E5")],
    )
    assert check(f, log=log) == []


def test_A4_month_year_only_control() -> None:
    """'October 2010' has only a year: this already passes."""
    log = corpus_log(F12)
    f = finding("The gauge failed in October 2010.", ["E5"], [])
    assert check(f, log=log) == []


# B. years


def test_B1_year_of_the_plan_named_in_the_system_prompt() -> None:
    """The system prompt itself says 'the February 2005 development plan'. A finding that
    mentions the 2005 plan while citing only data evidence gets 'undeclared number 2005'."""
    log = corpus_log(F12)
    f = finding(
        "After 2010-07-18 F-12 declined at 0.4461 per year, far steeper than the plateau the "
        "2005 plan assumed.",
        ["E9"],
        [d("0.4461", 0.4461, "E9")],
        confidence="inference",
    )
    assert check(f, log=log) == []


def test_B2_declared_year_is_rejected_although_years_need_no_declaration() -> None:
    """Models over-declare (the corpus has '2010-10-12' declared as a number). A bare year
    declared with its evidence fails: years are not in the number pool."""
    log = corpus_log(F12)
    f = finding(
        "In 2010 the oil rate fell from 4 050 to 1 146 Sm3/d.",
        ["E5"],
        [d("4 050", 4050, "E5"), d("1 146", 1146, "E5"), d("2010", 2010, "E5")],
    )
    assert check(f, log=log) == []


def test_B3_year_range_control() -> None:
    log = corpus_log(F12)
    f = finding("F-12 produced from 2008 to 2016.", ["E5"], [])
    assert check(f, log=log) == []


def test_B4_years_returned_as_integers_by_sql_are_not_years() -> None:
    """GROUP BY year(date) returns years as integers. They are not ISO dates, so the claim's
    years are 'undeclared numbers'; declaring them is what the prompt says not to do."""
    log = EvidenceLog()
    eid = add(
        log,
        "run_sql",
        {
            "sql": "SELECT year(date) AS yr, sum(oil_sm3) AS oil FROM production_daily "
            "WHERE well = 'F-12' GROUP BY yr ORDER BY yr"
        },
        {
            "columns": ["yr", "oil"],
            "rows": [[2010, 804545], [2011, 548116]],
            "row_count": 2,
            "truncated": False,
        },
    )
    f = finding(
        "F-12 produced 804 545 Sm3 in 2010 and 548 116 Sm3 in 2011.",
        [eid],
        [d("804 545", 804545, eid), d("548 116", 548116, eid)],
    )
    assert check(f, log=log) == []


# C. durations from the question / method


def test_C1_backtest_question_wording_365_days() -> None:
    """The backtest question says 'the 365 days after <cutoff>'. The model echoes it. 365
    is not in the forecast result (horizon_days is an argument), so it can neither be
    left undeclared nor declared."""
    log = corpus_log(F14)
    f = finding(
        "Forecast for the 365 days after 2013-01-01: P50 cumulative oil 276 078 Sm3.",
        ["E7"],
        [d("276 078", 276078, "E7")],
    )
    assert check(f, log=log) == []


def test_C1b_declaring_365_against_the_forecast_fails_too() -> None:
    log = corpus_log(F14)
    f = finding(
        "Forecast for the 365 days after 2013-01-01: P50 cumulative oil 276 078 Sm3.",
        ["E7"],
        [d("276 078", 276078, "E7"), d("365", 365, "E7")],
    )
    assert check(f, log=log) == []


def test_C2_twelve_months_without_hyphen() -> None:
    log = corpus_log(F14)
    s = "Over the next 12 months F-14 is forecast to produce 276 078 Sm3 (P50)."
    assert (
        check(
            finding("P50 276 078 Sm3.", ["E7"], [d("276 078", 276078, "E7")]),
            summary=s,
            summary_numbers=[d("276 078", 276078, "E7")],
            log=log,
        )
        == []
    )


def test_C3_thirty_day_median_without_hyphen() -> None:
    log = corpus_log(F12)
    f = finding(
        "The 30 day median water cut first exceeded 0.05 on 2009-10-31.",
        ["E5"],
        [d("0.05", 0.05, "E5")],
    )
    assert check(f, log=log) == []


def test_C4_hyphenated_windows_control() -> None:
    log = corpus_log(F14)
    f = finding("The 12-month forecast P50 is 276 078 Sm3.", ["E7"], [d("276 078", 276078, "E7")])
    assert check(f, log=log) == []


# D. units and scales


@pending("block 2: MSm3 / kSm3 / percent as units")
def test_D1_MSm3_with_the_full_value_declared() -> None:
    """'4.58 MSm3' is the standard way to write million Sm3 in this field (the PUD uses
    MSm3 throughout). 'million' and 'k' are scales; 'M' is not."""
    log = EvidenceLog()
    eid = add(
        log, "list_wells", {}, {"columns": ["well", "cum_oil_sm3"], "rows": [["F-12", 4579610]]}
    )
    f = finding("F-12 produced 4.58 MSm3 of oil in total.", [eid], [d("4.58 MSm3", 4579610, eid)])
    assert check(f, log=log) == []


@pending("block 2: MSm3 / kSm3 / percent as units")
def test_D1b_MSm3_with_the_short_value_declared() -> None:
    log = EvidenceLog()
    eid = add(
        log, "list_wells", {}, {"columns": ["well", "cum_oil_sm3"], "rows": [["F-12", 4579610]]}
    )
    f = finding("F-12 produced 4.58 MSm3 of oil in total.", [eid], [d("4.58", 4.58, eid)])
    assert check(f, log=log) == []


def test_D2_million_control() -> None:
    log = EvidenceLog()
    eid = add(
        log, "list_wells", {}, {"columns": ["well", "cum_oil_sm3"], "rows": [["F-12", 4579610]]}
    )
    f = finding("F-12 produced 4.58 million Sm3 of oil.", [eid], [d("4.58 million", 4579610, eid)])
    assert check(f, log=log) == []


@pending("block 2: MSm3 / kSm3 / percent as units")
def test_D3_percent_as_a_word() -> None:
    log = corpus_log(F14)
    f = finding(
        "Water cut was 75.6 percent in December 2012.",
        ["E9"],
        [d("75.6 percent", 0.7564, "E9")],
    )
    assert check(f, log=log) == []


@pending("block 2: MSm3 / kSm3 / percent as units")
def test_D4_kSm3_glued_unit() -> None:
    log = corpus_log(F14)
    f = finding("P50 cumulative oil 276 kSm3.", ["E7"], [d("276 kSm3", 276078, "E7")])
    assert check(f, log=log) == []


def test_D5_mill_Sm3_from_the_PUD_control() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, "1.2 mill Sm3/d")
    f = finding(
        "The plan expected gas plateau at about 1.2 mill Sm3/d.", [eid], [d("1.2", 1.2, eid)]
    )
    assert check(f, log=log) == []


# E. calculate constants


def _with_calc(expression: str, value: float, base: str = F12) -> tuple[EvidenceLog, str]:
    log = corpus_log(base)
    eid = add(
        log, "calculate", {"expression": expression}, {"expression": expression, "value": value}
    )
    return log, eid


def test_E1_dividing_by_1e6_is_a_unit_conversion() -> None:
    """1133053 Sm3 (E5) -> 1.13 MSm3. 1000 is a constant, 1e6 is not."""
    log, eid = _with_calc("1133053 / 1e6", 1.133)
    f = finding("That is 1.13 MSm3 of oil.", [eid], [d("1.13", 1.133, eid)])
    assert check(f, log=log) == []


@pending("block 2: unit-conversion constants in calculate")
def test_E2_dividing_by_1000000() -> None:
    log, eid = _with_calc("1133053 / 1000000", 1.133)
    f = finding("That is 1.13 MSm3 of oil.", [eid], [d("1.13", 1.133, eid)])
    assert check(f, log=log) == []


@pending("block 2: unit-conversion constants in calculate")
def test_E3_barrels_conversion() -> None:
    """Sm3 -> bbl is the one conversion an international audience asks for."""
    log, eid = _with_calc("1133053 * 6.2898", 7126682)
    f = finding("About 7.13 million barrels.", [eid], [d("7.13 million", 7126682, eid)])
    assert check(f, log=log) == []


@pending("block 2: year arithmetic in calculate")
def test_E4_year_arithmetic() -> None:
    """2010 - 2008 = 2 years: both years are in E5's dates but dates are not numbers."""
    log, eid = _with_calc("2010 - 2008", 2)
    f = finding("Breakthrough came 2 years after first oil.", [eid, "E5"], [d("2", 2, eid)])
    assert check(f, log=log) == []


def test_E5_constants_control() -> None:
    log, eid = _with_calc("2904 / 4050 * 100", 71.7)
    f = finding("A drop of 71.7%.", [eid], [d("71.7%", 71.7, eid)])
    assert check(f, log=log) == []


# F. quotes and translations

NOR = "Produksjonen avtar grunnet økende vannrate."


@pending("block 6: quote forms")
def test_F1_translation_in_parentheses_is_the_reports_claim_too() -> None:
    """Brackets are handled; parentheses, the other natural way to add a translation, are
    not. 'due to' then counts as the finding's own cause."""
    log = corpus_log(F12)
    eid = evidence_with_quote(log, NOR)
    f = finding(
        f"The PUD states “{NOR}” (Production declines due to increasing water rate.)",
        [eid],
        [],
    )
    assert check(f, log=log) == []


def test_F1b_bracketed_translation_control() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, NOR)
    f = finding(
        f"The PUD states “{NOR}” [Production declines due to increasing water rate.]", [eid], []
    )
    assert check(f, log=log) == []


@pending("block 6: quote forms")
def test_F2_two_scare_quoted_words_read_as_one_quote() -> None:
    """Scare quotes on two words 15+ characters apart are read as one long quotation."""
    log = corpus_log(F12)
    f = finding(
        "The 'plateau' period ended and the 'decline' period began around 2010-07-18.",
        ["E5"],
        [],
    )
    assert check(f, log=log) == []


@pending("block 6: quote forms")
def test_F3_quote_with_an_ellipsis() -> None:
    """Eliding the middle of a quotation with an ellipsis is standard practice."""
    log = corpus_log(F12)
    eid = evidence_with_quote(log, "Platåraten på 9.000 Sm3 /d vil vare om lag 19 måneder")
    f = finding(
        "The PUD expected “Platåraten på 9.000 Sm3 /d vil vare … 19 måneder.”",
        [eid],
        [d("9.000", 9000, eid), d("19", 19, eid)],
    )
    assert check(f, log=log) == []


@pending("block 6: quote forms")
def test_F4_quote_preceded_by_a_dash_without_space() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, NOR)
    f = finding(
        f"The PUD foresaw it —“{NOR}” [Production declines due to increasing water rate.]",
        [eid],
        [],
    )
    assert check(f, log=log) == []


@pending("block 6: quote forms")
def test_F5_quote_followed_by_a_question_mark() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, "Maksimum vannkutt 0,80")
    f = finding(
        "Was the plan's limit of “Maksimum vannkutt 0,80”? Yes, F-12 passed it on 2012-02-14.",
        [eid, "E5"],
        [],
    )
    assert check(f, log=log) == []


def test_F6_slash_translation_control() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, NOR)
    f = finding(
        f"The PUD states '{NOR}' / 'Production declines due to increasing water rate.'",
        [eid],
        [],
    )
    assert check(f, log=log) == []


# G. causal regex over petroleum vocabulary


@pending("block 6: causal and hedge vocabulary")
def test_G1_water_drive_is_a_recovery_mechanism_term() -> None:
    log = corpus_log(F12)
    f = finding(
        "F-12 produces under water drive from the injectors: water cut shifted to 0.8141 "
        "around 2010-10-24.",
        ["E5"],
        [d("0.8141", 0.8141, "E5")],
    )
    assert check(f, log=log) == []


@pending("block 6: causal and hedge vocabulary")
def test_G2_results_in_followed_by_an_evidence_id() -> None:
    """'The fit results in E9 show...' is a reference to evidence, not a cause."""
    log = corpus_log(F12)
    f = finding(
        "The fit results in E9 show Di 0.4461 per year with r2_log 0.8732.",
        ["E9"],
        [d("0.4461", 0.4461, "E9"), d("0.8732", 0.8732, "E9")],
    )
    assert check(f, log=log) == []


@pending("block 6: causal and hedge vocabulary")
def test_G3_summary_hedged_with_could_or_presumably() -> None:
    log = corpus_log(F12)
    f = finding("Oil fell 72%.", ["E5"], [d("72%", 0.7171, "E5")])
    for hedge in ("could be", "is presumably", "seems to be", "is perhaps", "is plausibly"):
        problems = check(
            f, summary=f"The faster decline {hedge} due to water breakthrough.", log=log
        )
        assert problems == [], (hedge, problems)


# H. declared small integers and zero


def test_H1_declared_count_of_wells() -> None:
    """The prompt says declare every number. A declared '4' (4 wells) is checked against
    the evidence, where the count is the number of rows, not a value."""
    log = corpus_log(F12)
    f = finding("Two of the 4 wells in the data are producers.", ["E1"], [d("4", 4, "E1")])
    assert check(f, log=log) == []


def test_H2_declared_duration_in_years() -> None:
    log = corpus_log(F12)
    f = finding(
        "The post-2010 fit covers almost 6 years at Di 0.4461 per year.",
        ["E9"],
        [d("6", 6, "E9"), d("0.4461", 0.4461, "E9")],
        confidence="inference",
    )
    assert check(f, log=log) == []


def test_H3_undeclared_small_integer_control() -> None:
    log = corpus_log(F12)
    f = finding("Two of the 4 wells in the data are producers.", ["E1"], [])
    assert check(f, log=log) == []


# I. SQL literal that the data also returns


def test_I1_single_day_lookup_returns_the_typed_date() -> None:
    """The most natural SQL for 'what happened on day X' is WHERE date = 'X'. The returned
    date equals the literal, so it is dropped as 'typed by the model' and the claim's date
    is 'not found' in the only evidence that holds it."""
    log = EvidenceLog()
    eid = add(
        log,
        "run_sql",
        {
            "sql": "SELECT date, oil_sm3, whp_bar, bhp_bar FROM production_daily "
            "WHERE well = 'F-12' AND date = '2010-10-12'"
        },
        {
            "columns": ["date", "oil_sm3", "whp_bar", "bhp_bar"],
            "rows": [["2010-10-12", 1759, 54.11, None]],
            "row_count": 1,
            "truncated": False,
        },
    )
    f = finding(
        "On 2010-10-12 F-12 made 1 759 Sm3 at WHP 54.11 bar with BHP missing.",
        [eid],
        [d("1 759", 1759, eid), d("54.11", 54.11, eid)],
    )
    assert check(f, log=log) == []


def test_I2_min_date_equal_to_the_filter_bound() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "run_sql",
        {
            "sql": "SELECT min(date) AS first_zero, count(*) AS n FROM production_issues "
            "WHERE well = 'F-12' AND issue = 'zero_gauge_reading' AND date >= '2010-10-12'"
        },
        {
            "columns": ["first_zero", "n"],
            "rows": [["2010-10-12", 4190]],
            "row_count": 1,
            "truncated": False,
        },
    )
    f = finding(
        "The zero readings start on 2010-10-12 (4 190 flagged rows).",
        [eid],
        [d("4 190", 4190, eid)],
    )
    assert check(f, log=log) == []


def test_I3_between_wider_than_the_claim_control() -> None:
    """The corpus case: BETWEEN '2010-10-10' AND '2010-10-20' leaves inner days as data."""
    log = corpus_log(GAUGE)
    f = finding(
        "On 2010-10-11 oil was 1 765 Sm3 with WHP 54.27 bar.",
        ["E6"],
        [d("1 765", 1765, "E6"), d("54.27", 54.27, "E6")],
    )
    assert check(f, log=log) == []


# J/K. number formats


def test_J1_scientific_notation_copied_from_the_event_summary() -> None:
    log = corpus_log(F12)
    f = finding(
        "oil_rate weekly mean 4.05e+03 -> 1.15e+03 around 2010-07-18.",
        ["E5"],
        [d("4.05e+03", 4050, "E5"), d("1.15e+03", 1146, "E5")],
    )
    assert check(f, log=log) == []


@pending("block 2: unicode minus")
def test_K1_unicode_minus_in_text_ascii_in_declaration() -> None:
    log = corpus_log(F12)
    f = finding("Oil rate change −72% around 2010-07-18.", ["E5"], [d("-72%", -0.7171, "E5")])
    assert check(f, log=log) == []


def test_K2_range_declared_as_one_text_is_sent_back_to_split() -> None:
    """By design: a range is two numbers. Declared as one text it is sent back with a
    message that says how to fix it (K2b is the accepted form)."""
    log = corpus_log(F12)
    eid = evidence_with_quote(log, "1,2-1,3 MSm3/d")
    f = finding("The plan expected gas plateau of 1,2-1,3 MSm3/d.", [eid], [d("1,2-1,3", 1.2, eid)])
    problems = check(f, log=log)
    assert problems and "declare each number separately" in problems[0]


def test_K2b_range_declared_as_two_numbers_control() -> None:
    log = corpus_log(F12)
    eid = evidence_with_quote(log, "1,2-1,3 MSm3/d")
    f = finding(
        "The plan expected gas plateau of 1,2-1,3 MSm3/d.",
        [eid],
        [d("1,2", 1.2, eid), d("1,3", 1.3, eid)],
    )
    assert check(f, log=log) == []


def test_K3_assorted_glued_formats_control() -> None:
    log = corpus_log(F12)
    f = finding(
        "Di=0.4461/yr (44.6 %/yr), r2_log=0.8732; rate 4 050→1 146Sm3/d (−72 %), a 7.89-fold "
        "and 17.73x change; 0.8141 (81.4%).",
        ["E5", "E9", "E14", "E15"],
        [
            d("0.4461", 0.4461, "E9"),
            d("44.6 %", 0.4461, "E9"),
            d("0.8732", 0.8732, "E9"),
            d("4 050", 4050, "E5"),
            d("1 146", 1146, "E5"),
            d("72 %", 0.7171, "E5"),
            d("7.89-fold", 7.89, "E15"),
            d("17.73x", 17.73, "E14"),
            d("0.8141", 0.8141, "E5"),
            d("81.4%", 0.8141, "E5"),
        ],
        confidence="inference",
    )
    assert check(f, log=log) == []


def test_K4_truncated_instead_of_rounded_is_rejected() -> None:
    """By design: 0.4461 written as 0.44 is truncated, not rounded (0.45)."""
    log = corpus_log(F12)
    f = finding("Di about 0.44 per year.", ["E9"], [d("0.44", 0.4461, "E9")])
    assert check(f, log=log) != []


# L. legacy (numbers=None) path


def test_L1_legacy_norwegian_decimal_percent() -> None:
    log = corpus_log(F12)
    f = Finding(
        claim="The PUD states “Maks vanninnhold 80 %” and a water cut limit of 0,80.",
        evidence_ids=["E7"],
        confidence="fact",
        numbers=None,
    )
    assert check(f, log=log) == []


def test_L2_legacy_control() -> None:
    log = corpus_log(F12)
    f = Finding(
        claim="Oil fell from 4 050 to 1 146 Sm3/d (-72%) around 2010-07-18.",
        evidence_ids=["E5"],
        confidence="fact",
        numbers=None,
    )
    assert check(f, log=log) == []


# M. fact on a poor fit, reported as poor


@pending("block 6: a fact that reports a poor fit as poor")
def test_M1_reporting_that_a_fit_is_poor_is_a_fact() -> None:
    """Two corpus runs lost a round on this. Rejected by design; listed for the record."""
    log = corpus_log(F12)
    f = finding(
        "The plateau window 2008-02-13 to 2010-07-18 gives a poor fit (r2_log 0.019), so no "
        "decline rate is taken from it.",
        ["E11"],
        [d("0.019", 0.01942, "E11")],
    )
    assert check(f, log=log) == []


# N. sanity: wrong answers are still rejected


def test_N1_wrong_number_control() -> None:
    log = corpus_log(F12)
    # 1 190 is at least 30 away from every number in E5 (E5 holds 1146 and 1158).
    f = finding("Oil fell to 1 190 Sm3/d.", ["E5"], [d("1 190", 1190, "E5")])
    assert check(f, log=log) != []


def test_N2_invented_quote_control() -> None:
    log = corpus_log(F12)
    f = finding("The PUD says “water will never reach the producers at all”.", ["E4"], [])
    assert check(f, log=log) != []
