"""What the words around a number say it is, against the value it was taken from.

Each rule here came from the checker bench (tests/checker_bench.py): a declared number
swapped for another value of the same result, which the checker let through. Each test
pins the false version it now stops and the true version it must still accept.
"""

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog


def _tool() -> None:
    """Stands in for a tool function."""


def add(log: EvidenceLog, tool: str, arguments: dict, result) -> str:
    return log.add(tool, arguments, result, _tool).id


def d(text: str, value: float, eid: str) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


def check(
    claim: str, eid: str, numbers: list, log: EvidenceLog, confidence: str = "fact"
) -> list[str]:
    finding = Finding(claim=claim, evidence_ids=[eid], confidence=confidence, numbers=numbers)
    return check_submission(Submission(summary="Done.", findings=[finding]), log)


def test_a_rate_per_year_is_no_fit_statistic() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "fit_decline",
        {"well": "F-12"},
        {"best": {"model": "exponential", "di_per_year": 0.4461, "r2_log": 0.8732}},
    )

    assert not check("F-12 declined at about 0.45 per year.", eid, [d("0.45", 0.4461, eid)], log)
    assert check("F-12 declined at about 0.87 per year.", eid, [d("0.87", 0.8732, eid)], log)


def test_a_date_that_opens_a_phrase_dates_its_numbers() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "production_monthly",
        {"well": "F-12"},
        {
            "columns": ["month", "oil_rate_sm3_per_day", "water_cut"],
            "rows": [
                ["2014-11", 185.7, 0.9623],
                ["2015-01", 862.4, 0.3107],
                ["2015-09", 400.0, 0.7373],
            ],
        },
    )

    def claim(cut: str) -> str:
        return f"In 2014-11 F-12 made 185.7 Sm3/d at {cut} water cut, in 2015-01 862.4 Sm3/d."

    def numbers(cut: str, value: float) -> list:
        return [d("185.7", 185.7, eid), d(cut, value, eid), d("862.4", 862.4, eid)]

    assert not check(claim("0.9623"), eid, numbers("0.9623", 0.9623), log)
    assert check(claim("0.7373"), eid, numbers("0.7373", 0.7373), log)


def test_the_items_of_a_list_share_the_words_after_it() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "waterflood",
        {"well": "F-12"},
        {
            "to_limits": [
                {"water_cut": 0.8, "oil_left_sm3": 0},
                {"water_cut": 0.9, "oil_left_sm3": 0},
                {"water_cut": 0.95, "oil_left_sm3": 0},
            ],
            "monthly": {
                "columns": ["month", "water_cut"],
                "rows": [["2010-06", 0.5446], ["2011-10", 0.807]],
            },
        },
    )

    def claim(first: str) -> str:
        return f"F-12 had already passed the {first}, 90% and 95% water cut limits."

    true = [d("80%", 0.8, eid), d("90%", 0.9, eid), d("95%", 0.95, eid)]
    false = [d("54%", 0.5446, eid), d("90%", 0.9, eid), d("95%", 0.95, eid)]
    # The trend's limits are a projection: an inference (Oct 30).
    assert not check(claim("80%"), eid, true, log, "inference")
    assert check(claim("54%"), eid, false, log, "inference")


def test_a_bare_year_is_the_years_row_unless_an_extreme_is_named() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "waterflood",
        {"well": "F-12"},
        {
            "by_year": {"columns": ["year", "liquid_rate_sm3_per_day"], "rows": [[2010, 5130]]},
            "monthly": {
                "columns": ["month", "liquid_rate_sm3_per_day"],
                "rows": [["2010-01", 5817], ["2010-04", 5600]],
            },
        },
    )

    def claim(rate: str, verb: str = "held at") -> str:
        return f"Yearly water cut was high in 2010, while liquid {verb} {rate} Sm3/d in 2010."

    assert not check(claim("5 130"), eid, [d("5 130", 5130, eid)], log)
    assert check(claim("5 817"), eid, [d("5 817", 5817, eid)], log)
    assert not check(claim("5 817", "peaked at"), eid, [d("5 817", 5817, eid)], log)


def test_a_weight_on_the_string_is_no_depth() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "well_operations",
        {"well": "F-12"},
        {"summary": "Set plug with mid packer element at 3150 m. P/U weight at 2500 m 4400 lbs."},
    )

    assert not check("A plug was set at 3150 m.", eid, [d("3150", 3150, eid)], log)
    assert check("A plug was set at 4400 m.", eid, [d("4400", 4400, eid)], log)


def test_a_plugs_depth_is_not_a_gauges() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "well_operations",
        {"well": "F-12"},
        {"summary": "Installed plug with mid element at 3150 m. Hung off gauge at 2929 m."},
    )

    assert not check("A plug was set at 3150 m.", eid, [d("3150", 3150, eid)], log)
    assert check("A plug was set at 2929 m.", eid, [d("2929", 2929, eid)], log)


def test_a_range_shares_the_unit_after_its_end() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "well_operations",
        {"well": "F-12"},
        {"perforations": [{"top_m": 3126.2, "bottom_m": 3130.7}], "rate_sm3_per_day": 4500},
    )

    def claim(top: str) -> str:
        return f"F-12 was perforated from {top} to 3130.7 m."

    assert not check(
        claim("3126.2"), eid, [d("3126.2", 3126.2, eid), d("3130.7", 3130.7, eid)], log
    )
    assert check(claim("4500"), eid, [d("4500", 4500, eid), d("3130.7", 3130.7, eid)], log)


def test_a_number_beside_a_quote_is_the_quotes() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "read_report_page",
        {"doc_id": "volve_pud", "page": 23},
        {"text": "Oljekapasitet 9000 Sm3 /d. Væskekapasitet 13000 Sm3 /d."},
    )

    def claim(rate: str) -> str:
        return f'The plan gives an oil capacity of {rate} Sm3/d ("Oljekapasitet 9000 Sm3 /d").'

    assert not check(claim("9,000"), eid, [d("9,000", 9000, eid)], log)
    problems = check(claim("13,000"), eid, [d("13,000", 13000, eid)], log)
    assert problems and "beside the quote" in problems[0]


def test_a_count_of_months_fits_its_years() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "run_python",
        {"well": "F-12"},
        {"from": "2011-01-01", "to": "2013-12-31", "result": {"n": 36, "choke_mean": 87.4}},
    )

    assert not check("Over 36 months of 2011-2013 the choke opened.", eid, [d("36", 36, eid)], log)
    problems = check(
        "Over 87 months of 2011-2013 the choke opened.", eid, [d("87", 87.4, eid)], log
    )
    assert any("at most 36 months" in p for p in problems)


def test_a_period_mean_is_no_level_on_one_day() -> None:
    """The tenth panel's reservoir judge: "to 1 888 around 2010-05-02" gives the May-December
    mean as the rate that day (about 2,640)."""
    log = EvidenceLog()
    eid = add(
        log,
        "detect_events",
        {"well": "F-12"},
        [
            {
                "kind": "regime_change",
                "signal": "oil_rate",
                "start": "2010-05-02",
                "details": {"before_mean": 4207, "after_mean": 1888, "relative_change": -0.5512},
            }
        ],
    )
    numbers = [d("55%", 0.5512, eid), d("1 888", 1888, eid)]
    day = "F-12's oil rate shifted down 55% around 2010-05-02 to 1 888 Sm3/d."
    problems = check(day, eid, numbers, log)
    assert any("mean of the period after 2010-05-02" in p for p in problems)
    average = (
        "F-12's oil rate shifted down 55% around 2010-05-02 and averaged 1 888 Sm3/d after it."
    )
    assert not check(average, eid, numbers, log)


def test_a_forecast_is_an_inference() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "forecast_production",
        {"well": "F-12"},
        {
            "cumulative_oil_sm3": {"p50": 359007, "p90": 324192, "p10": 469712},
            "fit": {"r2_log": 0.9},
        },
    )
    claim = "F-12 should make about 359,000 Sm3 in the next 12 months."
    numbers = [d("359,000", 359007, eid)]
    assert any("projection" in p for p in check(claim, eid, numbers, log))
    assert not check(claim, eid, numbers, log, "inference")
    # The fit behind it describes the past: a fact may give it.
    assert not check("The fit has r2 0.9.", eid, [d("0.9", 0.9, eid)], log)
