import copy

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog, code_ref


def _tool() -> None:
    """Stands in for a tool function in code_ref."""


def log_with(*results: dict) -> EvidenceLog:
    log = EvidenceLog()
    for result in results:
        log.add("tool", {"well": "F-12"}, result, _tool)
    return log


EVENTS = {
    "summary": "oil_rate shifts around 2010-07-18: weekly mean 4.05e+03 -> 1146 (-72%).",
    "details": {"before_mean": 4049.93, "after_mean": 1145.72, "relative_change": -0.7171},
}
FIT = {"best": {"di_per_year": 0.7532, "r2_log": 0.91}, "warnings": []}
POOR_FIT = {"best": {"di_per_year": 0.0575, "r2_log": 0.0165}, "warnings": ["poor fit"]}
CUMULATIVE = {"columns": ["well", "cum_oil_sm3"], "rows": [["F-12", 4579610]]}


def submit(
    *findings: Finding, summary: str = "F-12 declined after water breakthrough."
) -> Submission:
    return Submission(summary=summary, findings=list(findings))


def test_numbers_and_dates_from_cited_evidence_pass() -> None:
    log = log_with(EVENTS, FIT, CUMULATIVE)
    finding = Finding(
        claim="F-12 oil rate fell 72% around 2010-07-18, from 4049.93 to 1145.72 Sm3/d; "
        "afterwards Di was 0.7532/yr. Cumulative oil 4,579,610 Sm3, about 4.58 million.",
        evidence_ids=["E1", "E2", "E3"],
        confidence="fact",
    )

    assert check_submission(submit(finding), log) == []


def test_number_missing_from_the_cited_evidence_is_reported() -> None:
    log = log_with(EVENTS, FIT)
    finding = Finding(claim="Di grew 13x to 0.7532/yr.", evidence_ids=["E2"], confidence="fact")

    problems = check_submission(submit(finding), log)

    assert len(problems) == 1
    assert "13x not found in E2" in problems[0]
    assert "calculate" in problems[0]


def test_numbers_must_come_from_the_cited_items_not_any_item() -> None:
    log = log_with(EVENTS, FIT)
    finding = Finding(claim="Di was 0.7532/yr.", evidence_ids=["E1"], confidence="fact")

    assert "0.7532 not found in E1" in check_submission(submit(finding), log)[0]


def test_wrong_dates_and_years_are_reported() -> None:
    log = log_with(EVENTS)
    finding = Finding(
        claim="The shift happened on 2010-07-25, in 2011.", evidence_ids=["E1"], confidence="fact"
    )

    problems = check_submission(submit(finding), log)

    assert "2011" in problems[0]
    assert "2010-07-25" in problems[1]


def test_month_matches_a_day_in_that_month() -> None:
    log = log_with(EVENTS)
    finding = Finding(claim="The shift came in 2010-07.", evidence_ids=["E1"], confidence="fact")

    assert check_submission(submit(finding), log) == []


def test_unknown_or_missing_evidence_ids() -> None:
    log = log_with(EVENTS)

    unknown = check_submission(
        submit(Finding(claim="x", evidence_ids=["E9"], confidence="fact")), log
    )
    none = check_submission(submit(Finding(claim="x", evidence_ids=[], confidence="fact")), log)

    assert "unknown evidence E9" in unknown[0]
    assert "cites no evidence" in none[0]


def test_fact_on_evidence_with_warnings_must_be_an_inference() -> None:
    log = log_with(POOR_FIT)
    as_fact = Finding(claim="Plateau Di was 0.0575/yr.", evidence_ids=["E1"], confidence="fact")
    as_inference = as_fact.model_copy(update={"confidence": "inference"})

    assert "marked fact" in check_submission(submit(as_fact), log)[0]
    assert check_submission(submit(as_inference), log) == []


def test_small_integers_and_identifiers_are_not_checked() -> None:
    log = log_with(FIT)
    finding = Finding(
        claim="Across 3 steps the P90 case and r2_log for F-15 D and F-1 C in Sm3/d agree.",
        evidence_ids=["E1"],
        confidence="inference",
    )

    assert check_submission(submit(finding), log) == []


def test_summary_numbers_must_exist_in_some_evidence() -> None:
    log = log_with(EVENTS, FIT)
    finding = Finding(claim="Di was 0.7532/yr.", evidence_ids=["E2"], confidence="fact")

    problems = check_submission(submit(finding, summary="Oil fell 72%, then 55%."), log)

    assert problems == ["Summary: 55% not found in any evidence."]


def test_code_ref_hashes_the_source() -> None:
    ref = code_ref(_tool)

    assert ref.startswith("tests.test_checker._tool@") or ref.startswith("test_checker._tool@")
    assert len(ref.rsplit("@", 1)[1]) == 12


def test_scientific_notation_is_one_number() -> None:
    log = log_with({"best": {"di_per_year": 9.798e-17, "r2_log": -2.22e-16}, "warnings": []})
    finding = Finding(
        claim="Plateau Di was 9.798e-17/yr with r2_log -2.22e-16.",
        evidence_ids=["E1"],
        confidence="fact",
    )

    assert check_submission(submit(finding), log) == []


def test_year_like_numbers_pass_when_they_are_evidence_values() -> None:
    log = log_with({"details": {"zero_days": 2095, "rate_after": 2000.0}, "start": "2010-10-12"})
    finding = Finding(
        claim="The gauge read 0 on 2095 days from 2010; the rate after restart was 2000.",
        evidence_ids=["E1"],
        confidence="fact",
    )

    assert check_submission(submit(finding), log) == []


def test_causal_claim_marked_fact_is_rejected() -> None:
    log = log_with(EVENTS)
    as_fact = Finding(
        claim="Water breakthrough drove the 72% drop.", evidence_ids=["E1"], confidence="fact"
    )
    as_inference = as_fact.model_copy(update={"confidence": "inference"})

    problems = check_submission(submit(as_fact), log)

    assert len(problems) == 1
    assert "states a cause ('drove')" in problems[0]
    assert check_submission(submit(as_inference), log) == []


REPORT = {
    "facts": {
        "columns": ["page", "statement_en", "quote"],
        "rows": [
            [
                23,
                "Injection water breaks through in the wells after about 2 years.",
                "injeksjonsvannet som bryter gjennom i brønnene etter omlag 2 år.",
            ]
        ],
    }
}


def test_quoted_text_must_come_from_the_cited_evidence() -> None:
    log = log_with(REPORT, EVENTS)
    quoted = Finding(
        claim="The plan expected 'injeksjonsvannet som bryter gjennom i brønnene etter omlag "
        '2 år\' / "Injection water breaks through in the wells after about 2 years."',
        evidence_ids=["E1"],
        confidence="fact",
    )
    invented = quoted.model_copy(
        update={"claim": "The plan said 'water will never reach the producing wells'."}
    )
    wrong_source = quoted.model_copy(update={"evidence_ids": ["E2"]})

    assert check_submission(submit(quoted), log) == []
    assert "quoted text not found in E1" in check_submission(submit(invented), log)[0]
    assert "quoted text not found in E2" in check_submission(submit(wrong_source), log)[0]


def test_apostrophes_and_short_quotes_are_not_citations() -> None:
    log = log_with(REPORT)
    finding = Finding(
        claim="The oil's 'plateau' and the operator's plan agree on 2 years.",
        evidence_ids=["E1"],
        confidence="inference",
    )

    assert check_submission(submit(finding), log) == []


def test_summary_causes_need_a_hedge() -> None:
    log = log_with(EVENTS)
    finding = Finding(claim="Oil fell 72%.", evidence_ids=["E1"], confidence="fact")

    certain = check_submission(
        submit(finding, summary="Oil fell because water broke through."), log
    )
    hedged = check_submission(
        submit(finding, summary="Oil fell, most likely because water broke through."), log
    )

    assert "Summary states a cause as certain ('because')" in certain[0]
    assert hedged == []


PUD_FACTS = {
    "facts": {
        "columns": ["statement_en", "quote"],
        "rows": [
            [
                "The plateau rate of 9.000 Sm3/d will last about 19 months.",
                "Platåraten på 9.000 Sm3 /d vil vare om lag 19 måneder.",
            ],
            ["Maximum water cut is 0,80.", "Maksimum vannkutt 0,80"],
            [
                "Production declines due to increasing water rate.",
                "Produksjonen avtar grunnet økende vannrate.",
            ],
        ],
    }
}


def test_norwegian_number_formats_in_evidence() -> None:
    log = log_with(PUD_FACTS)
    finding = Finding(
        claim="The plan's plateau was 9,000 Sm3/d for 19 months and the water cut limit 0.80; "
        "the quote reads 'Maksimum vannkutt 0,80'.",
        evidence_ids=["E1"],
        confidence="fact",
    )

    assert check_submission(submit(finding), log) == []


def test_a_cause_inside_a_quote_is_the_reports_claim() -> None:
    log = log_with(PUD_FACTS)
    reported = Finding(
        claim="The plan states 'Production declines due to increasing water rate.'",
        evidence_ids=["E1"],
        confidence="fact",
    )
    asserted = Finding(
        claim="F-12 declined, driving the drop below plan.", evidence_ids=["E1"], confidence="fact"
    )

    assert check_submission(submit(reported), log) == []
    assert "states a cause ('driving')" in check_submission(submit(asserted), log)[0]


def test_a_bracketed_translation_of_a_quote_is_the_reports_claim_too() -> None:
    log = log_with(PUD_FACTS)
    translated = Finding(
        claim="The plan states “Produksjonen avtar grunnet økende vannrate.” [Production "
        "declines due to increasing water rate.]",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[],
    )
    own_brackets = Finding(
        claim="F-12 fell [due to water].", evidence_ids=["E1"], confidence="fact", numbers=[]
    )

    assert check_submission(submit(translated), log) == []
    assert "states a cause ('due to')" in check_submission(submit(own_brackets), log)[0]


def _add(log: EvidenceLog, tool: str, arguments: dict, result, warnings=()) -> None:
    item = log.add(tool, arguments, result, _tool)
    if warnings:
        object.__setattr__(item, "warnings", tuple(warnings))


def test_literals_typed_into_sql_are_not_evidence() -> None:
    log = EvidenceLog()
    sql = "SELECT 'Vanngjennombrudd ventes etter 7 år' AS q, DATE '2013-05-17' AS d, 4321 AS n"
    _add(
        log,
        "run_sql",
        {"sql": sql},
        {
            "columns": ["q", "d", "n"],
            "rows": [["Vanngjennombrudd ventes etter 7 år", "2013-05-17", 4321]],
        },
    )
    finding = Finding(
        claim="The plan said 'Vanngjennombrudd ventes etter 7 år' and 4321 Sm3 by 2013-05-17.",
        evidence_ids=["E1"],
        confidence="fact",
    )

    problems = " ".join(check_submission(submit(finding), log))

    assert "4321" in problems and "2013-05-17" in problems and "quoted text not found" in problems


def test_a_calculation_needs_inputs_from_evidence_and_keeps_their_warnings() -> None:
    log = EvidenceLog()
    _add(log, "fit_decline", {}, {"best": {"di_per_year": 0.0575}}, warnings=["poor fit"])
    _add(log, "calculate", {"expression": "0.4321"}, {"expression": "0.4321", "value": 0.4321})
    _add(
        log,
        "calculate",
        {"expression": "0.0575 * 100"},
        {"expression": "0.0575 * 100", "value": 5.75},
    )

    invented = Finding(claim="Decline was 0.4321/yr.", evidence_ids=["E2"], confidence="inference")
    laundered = Finding(claim="Decline was 5.75% per year.", evidence_ids=["E3"], confidence="fact")
    honest = laundered.model_copy(update={"confidence": "inference"})

    assert "no evidence contains" in check_submission(submit(invented), log)[0]
    assert (
        "rests on evidence with warnings (E3: poor fit)"
        in check_submission(submit(laundered), log)[0]
    )
    assert check_submission(submit(honest), log) == []


def test_numbers_must_round_to_the_claim_not_merely_be_close() -> None:
    log = log_with(EVENTS)  # holds 4049.93 and 1145.72

    close = Finding(claim="Rate fell to 1150 Sm3/d.", evidence_ids=["E1"], confidence="fact")
    rounded = Finding(
        claim="Rate fell to 1146 Sm3/d, about 1.1k.", evidence_ids=["E1"], confidence="fact"
    )

    assert "1150 not found" in check_submission(submit(close), log)[0]
    assert check_submission(submit(rounded), log) == []


def test_day_needs_that_day_and_wellbore_names_are_not_numbers() -> None:
    log = log_with({"rows": [["2014-02", 5205]]})

    day = Finding(claim="On 2014-02-17 the rate was 5205.", evidence_ids=["E1"], confidence="fact")
    month = Finding(
        claim="In 2014-02 well 15/9-F-12 made 5205 Sm3/d.", evidence_ids=["E1"], confidence="fact"
    )

    assert "2014-02-17" in check_submission(submit(day), log)[0]
    assert check_submission(submit(month), log) == []


def test_numbers_glued_to_units_are_checked() -> None:
    log = log_with({"value": 4321})
    finding = Finding(claim="It made 98765Sm3 of oil.", evidence_ids=["E1"], confidence="fact")

    assert "98765 not found" in check_submission(submit(finding), log)[0]


def test_summary_quotes_must_exist() -> None:
    log = log_with(PUD_FACTS)
    finding = Finding(claim="Water cut limit 0.80.", evidence_ids=["E1"], confidence="fact")

    problems = check_submission(
        submit(finding, summary="The plan said 'water would never reach the wells at all'."), log
    )

    assert problems and "water would never reach" in problems[0]


def test_space_grouped_thousands_are_one_number() -> None:
    log = log_with(EVENTS, {"oil": 1133053})
    finding = Finding(
        claim="Plateau rate around 4 050 Sm3/d; 1 133 053 Sm3 oil since.",
        evidence_ids=["E1", "E2"],
        confidence="fact",
    )

    assert check_submission(submit(finding), log) == []


def declared(text: str, value: float, evidence_id: str = "E1") -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=evidence_id)


def test_declared_numbers_in_any_format_pass() -> None:
    log = log_with(EVENTS, {"oil": 1133053})
    finding = Finding(
        claim="Rate fell from around 4 050 to 1146 Sm3/d (-72%), with 1.13 million Sm3 since 2010.",
        evidence_ids=["E1", "E2"],
        confidence="fact",
        numbers=[
            declared("4 050", 4050),
            declared("1146", 1146),
            declared("72%", 72),
            declared("1.13 million", 1.13, "E2"),
        ],
    )

    assert check_submission(submit(finding), log) == []


def test_declared_numbers_are_checked_against_their_own_evidence() -> None:
    log = log_with(EVENTS, {"oil": 1133053})
    # 1150 would be a fair rounding of 1145.72 to tens; 1160 is not.
    wrong_value = Finding(
        claim="Rate fell to 1160 Sm3/d.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1160", 1160)],
    )
    wrong_source = Finding(
        claim="Oil since: 1133053 Sm3.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1133053", 1133053, "E2")],
    )
    not_in_text = Finding(
        claim="Rate fell sharply.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1146", 1146)],
    )

    assert "'1160' = 1160 is not in E1" in check_submission(submit(wrong_value), log)[0]
    assert "comes from E2, which it does not cite" in check_submission(submit(wrong_source), log)[0]
    assert check_submission(submit(not_in_text), log) == []  # an unused declaration hides nothing


def test_every_written_number_must_be_declared() -> None:
    log = log_with(EVENTS)
    finding = Finding(
        claim="In 2010 rate fell to 1146 Sm3/d, then to 980, across 3 wells; plan said "
        "'Maksimum vannkutt 0,80 for alle brønner'.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1146", 1146)],
    )

    problems = check_submission(submit(finding), log)

    # 2010 is a year in the evidence, 3 is a small count, 0,80 sits inside a quote
    # (checked as a quote); only 980 is an undeclared number.
    assert any("undeclared number(s) 980." in p for p in problems)
    assert not any("2010" in p for p in problems)


def test_calendar_windows_need_no_declaring_but_durations_do() -> None:
    log = log_with(EVENTS)
    window = Finding(
        claim="The 30-day median fell to 1146 Sm3/d; the 12-month forecast follows it.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1146", 1146)],
    )
    duration = Finding(
        claim="After a 45-day shut-in the rate fell to 1146 Sm3/d.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[declared("1146", 1146)],
    )

    assert check_submission(submit(window), log) == []
    assert "undeclared number(s) 45." in check_submission(submit(duration), log)[0]


def test_summary_numbers_are_declared_too() -> None:
    log = log_with(EVENTS)
    finding = Finding(claim="Rate fell.", evidence_ids=["E1"], confidence="fact", numbers=[])

    good = check_submission(
        Submission(
            summary="Oil fell 72%.", findings=[finding], summary_numbers=[declared("72%", 0.72)]
        ),
        log,
    )
    bad = check_submission(
        Submission(
            summary="Oil fell 72%, then 55%.",
            findings=[finding],
            summary_numbers=[declared("72%", 0.72)],
        ),
        log,
    )

    assert good == []
    assert "Summary: undeclared number(s) 55%" in bad[0]


def test_display_spans_mark_what_the_checker_read() -> None:
    from wellpilot.agent.checker import display_spans

    claim = "F-12 fell 72% on 12 October 2010; the plan says “Maksimum vannkutt 0,80” for 2 years."
    numbers = [
        DeclaredNumber(text="72%", value=0.72, evidence_id="E1"),
        DeclaredNumber(text="12", value=12, evidence_id="E1"),  # only in the well and the date
        DeclaredNumber(text="0,80", value=0.8, evidence_id="E2"),
        DeclaredNumber(text="2", value=2, evidence_id="E2"),
    ]

    spans = display_spans(claim, numbers)

    marked = [(claim[s["start"] : s["end"]], s["kind"], s.get("evidence_id")) for s in spans]
    assert marked == [
        ("72%", "number", "E1"),
        ("“Maksimum vannkutt 0,80”", "quote", None),
        ("0,80", "number", "E2"),
        ("2", "number", "E2"),
    ]


def test_runs_saved_before_spans_get_them_on_load() -> None:
    from wellpilot.api.runs import RunRecord

    finding = {"claim": "Rate 4 050 Sm3/d.", "evidence_ids": ["E1"], "confidence": "fact"}
    finding["numbers"] = [{"text": "4 050", "value": 4050, "evidence_id": "E1"}]
    result = {"summary": "Done.", "summary_numbers": [], "findings": [finding]}
    record = RunRecord.from_file(
        {
            "id": "0123456789ab",
            "question": "Rate?",
            "as_of": None,
            "created_at": "2026-10-27T00:00:00+00:00",
            "status": "done",
            "events": [{"seq": 0, "type": "final", **copy.deepcopy(result)}],
            "result": result,
            "evidence": {},
        }
    )

    for answer in (record.result, record.events[0]):
        spans = answer["findings"][0]["spans"]
        assert [(s["start"], s["end"]) for s in spans] == [(5, 10)]
        assert answer["summary_spans"] == []
