from wellpilot.agent.checker import Finding, Submission, check_submission
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
