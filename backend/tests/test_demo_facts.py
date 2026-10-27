"""The facts the demo video and README quote, checked through the agent's own tools on the
committed Volve fixture. A change in ingest or analysis that moves one of them fails here,
before it reaches a recording.
"""

import json
from pathlib import Path

import pytest

from wellpilot.agent.tools import execute
from wellpilot.data.store import ProductionStore

PROCESSED = Path(__file__).parent / "fixtures" / "processed"


@pytest.fixture(scope="module")
def store() -> ProductionStore:
    return ProductionStore(PROCESSED)


def call(store: ProductionStore, tool: str, **args) -> dict | list:
    result = execute(store, tool, json.dumps(args))
    assert "result" in result, result
    return result["result"]


def test_f12_regime_change_water_and_gauge(store) -> None:
    events = call(store, "detect_events", well="F-12")
    by = {(e["kind"], e["signal"], e["start"]): e for e in events}

    drop = by[("regime_change", "oil_rate", "2010-07-18")]
    assert round(drop["details"]["before_mean"]) == 4050
    assert round(drop["details"]["after_mean"]) == 1146
    assert "4,050 -> 1,146 (-72%)" in drop["summary"]

    crossings = {
        e["details"]["threshold"]: e["start"] for e in events if e["kind"] == "threshold_crossing"
    }
    assert crossings == {0.05: "2009-10-31", 0.5: "2010-06-04", 0.8: "2012-02-14"}

    gauge = next(e for e in events if e["kind"] == "gauge_failure")
    assert gauge["start"] == "2010-10-12"
    assert "2095 days" in gauge["summary"] and "1830 days" in gauge["summary"]
    assert "1,133,053 Sm3" in gauge["summary"]

    jan_2012 = by[("shut_in", "on_stream_hrs", "2012-01-03")]
    assert jan_2012["details"]["days"] == 11 and jan_2012["details"]["field_wide"]


def test_f12_decline_after_the_regime_change(store) -> None:
    fit = call(store, "fit_decline", well="F-12", start="2010-07-18", end="2012-12-09")["best"]

    assert fit["model"] == "exponential"
    assert fit["di_per_year"] == pytest.approx(0.4461, abs=1e-4)
    assert fit["r2_log"] == pytest.approx(0.8732, abs=1e-4)


def test_pud_plan_facts_are_exact_quotes(store) -> None:
    facts = call(store, "search_reports", query="vannkutt", limit=30)["facts"]
    quotes = {row[facts["columns"].index("quote")] for row in facts["rows"]}

    assert "Maksimum vannkutt 0,80" in quotes


def test_the_field_against_the_2005_plan(store) -> None:
    """The plan expected the field to stop producing after 2012; it produced until 2016."""
    result = call(store, "plan_vs_actual")
    rows = {r[0]: r for r in result["years"]["rows"]}

    assert rows[2012][3] == 0.4 and rows[2013][3] == 0.0  # plan P50 ends after 2012
    assert result["plan_p50_last_year"] == 2012 and result["actual_last_year"] == 2016
    assert result["totals_msm3"]["actual"] == pytest.approx(10.04, abs=0.01)
    assert result["totals_msm3"]["plan_p50"] == pytest.approx(11.34, abs=0.01)
    assert result["actual_after_plan_p50_ends_msm3"] == pytest.approx(2.48, abs=0.01)
    assert [r[0] for r in result["wells_first_oil_after_plan_p50_ends"]["rows"]] == [
        "F-11",
        "F-15 D",
        "F-1 C",
        "F-5",
    ]
    assert "2010 2,14 1,49 2,13 2,82" in result["plan_rows_quoted"]
    # The plan assumed first oil a year earlier: by year of production the field started
    # ahead of plan (1.76 against 1.53 MSm3) and fell behind from its second year.
    aligned = {r[0]: r for r in result["by_production_year"]["rows"]}
    assert aligned[1][1:] == [2007, 1.53, 2008, pytest.approx(1.76, abs=0.01)]
    assert aligned[2][2] == 3.1 and aligned[2][4] == pytest.approx(2.68, abs=0.01)


def test_the_field_against_the_plans_design_limits(store) -> None:
    """The plan's 0,80 water cut is a field limit: the field reached it in June 2012, the
    year the plan's forecast ends; liquid never reached the 13000 Sm3/d capacity."""
    result = call(store, "field_vs_design")
    limits = result["design_limits"]
    against = result["field_against_limits"]

    assert limits["water_cut"] == {
        "limit": "maximum water cut",
        "value": 0.8,
        "quote": "Maksimum vannkutt 0,80",
    }
    assert limits["liquid_sm3_per_day"]["value"] == 13000
    assert against["water_cut"]["first_month_at_or_above"] == "2012-06"
    assert against["liquid_sm3_per_day"]["months_at_or_above"] == 0
    assert against["liquid_sm3_per_day"]["max_month"] == "2014-06"
    assert against["oil_sm3_per_day"]["max"] == pytest.approx(8924, abs=1)
    # The 0.80 share at 13000 Sm3/d liquid is 10400 Sm3/d of water: the field ran at the
    # design water cut from 2012 but never over its water handling (worst day 10240).
    water = limits["water_sm3_per_day"]
    assert water["value"] == 10400 and water["quote"] is None
    assert against["water_sm3_per_day"]["months_at_or_above"] == 0
    assert against["water_sm3_per_day"]["max_day"] == pytest.approx(10240, abs=1)


def test_pud_quotes_are_located_on_their_page(store) -> None:
    """The UI marks a quote on the page image Muse Spark read it from."""
    import json as _json

    rows = store.query(
        "SELECT quote_boxes FROM report_facts WHERE doc_id = 'volve_pud' AND page = 23 "
        "AND quote = 'Maksimum vannkutt 0,80'"
    ).rows
    boxes = _json.loads(rows.iloc[0, 0])

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 0 < x < 1 and 0 < y < 1 and 0 < w < 0.5 and 0 < h < 0.05


def test_table_rows_parse_with_cells_written_apart_by_bars() -> None:
    """Muse Spark Open writes table rows as "2010 | 2,14 | 1,49 | 2,13 | 2,82"."""
    from wellpilot.agent.tools import DESIGN_LIMITS, PLAN_ROW, _cells

    assert PLAN_ROW.match(_cells("2010 | 2,14 | 1,49 | 2,13 | 2,82"))[1] == "2010"
    water_cut = next(p for key, p, _ in DESIGN_LIMITS if key == "water_cut")
    assert water_cut.match(_cells("Maksimum vannkutt | 0,80"))[1] == "0,80"


def test_the_field_screen_flags_f12_at_the_end_of_2010() -> None:
    """As of 2010-12-31 the screen puts F-12 first: its downhole gauge has been dead since
    2010-10-12 while it kept producing, and its rate stepped down in May 2010."""
    from datetime import date

    from wellpilot.analysis.screen import screen

    result = screen(ProductionStore(PROCESSED, as_of=date(2010, 12, 31)))
    first = result["wells"][0]
    kinds = {i["kind"]: i for i in first["issues"]}

    assert result["as_of"] == "2010-12-31"
    assert first["well"] == "F-12" and first["level"] == "alarm"
    assert kinds["dead_gauge"]["since"] == "2010-10-12"
    assert "123,258 Sm3" in kinds["dead_gauge"]["text"]
    assert kinds["step_down"]["since"] == "2010-05-02"


def test_the_field_screen_ends_on_the_last_day_of_oil(store) -> None:
    result = call(store, "field_screen")

    assert result["as_of"] == "2016-09-17"  # not December, when only F-4 injected
    assert {w["well"] for w in result["wells"]} >= {"F-12", "F-14", "F-11", "F-4"}


def test_waterflood_diagnostics_for_f12_at_the_end_of_2012() -> None:
    """With data to 2012-12-31: F-12's WOR trend leaves about 165k Sm3 before water cut 0.90
    and 560k before 0.95 (it went on to make about 444k); liquid held near 5,000 Sm3/d while
    the choke opened to 100%; the field injected more water than it produced liquid."""
    import json as _json
    from datetime import date

    from wellpilot.agent.tools import execute

    store = ProductionStore(PROCESSED, as_of=date(2012, 12, 31))
    result = execute(store, "waterflood", _json.dumps({"well": "F-12"}))["result"]
    fit = result["wor_fit"]
    limits = {round(lim["water_cut"], 2): lim for lim in fit["to_limits"]}
    years = {r[0]: r for r in result["by_year"]["rows"]}
    cols = result["by_year"]["columns"]
    voidage = result["field_voidage"]

    assert fit["r2"] == pytest.approx(0.92, abs=0.01) and result["no_trend"] is None
    assert limits[0.8]["already_passed"]
    assert limits[0.9]["remaining_oil_sm3"] == pytest.approx(164_702, rel=0.01)
    assert limits[0.95]["remaining_oil_sm3"] == pytest.approx(560_299, rel=0.01)
    assert years[2012][cols.index("choke_pct")] == 100
    assert 4_900 < years[2011][cols.index("liquid_rate_sm3_per_day")] < 5_300
    # At reservoir conditions (Bo 1.385 from the plan) injection kept pace with voidage,
    # within 10% every year; surface volumes alone would say it exceeded it by 10-50%.
    ratio = voidage["columns"].index("replacement_ratio")
    surface = voidage["columns"].index("surface_ratio")
    assert all(0.9 < r[ratio] < 1.15 for r in voidage["rows"])
    assert all(r[surface] > 1.05 for r in voidage["rows"])


def test_a_poor_wor_trend_is_left_out_and_its_tables_carry_no_warning() -> None:
    """Over F-12's whole life its last 36 months of WOR (2013-05 to 2016-08) do not rise: no
    projection, and liquid and choke by year stay plain data (run 7e187f965639's first
    answer had to hedge "5 130 Sm3/d" because a fit warning covered the whole result)."""
    import json as _json

    from wellpilot.agent.tools import execute

    data = execute(ProductionStore(PROCESSED), "waterflood", _json.dumps({"well": "F-12"}))[
        "result"
    ]
    assert data["wor_fit"] is None and "WOR is not rising" in data["no_trend"]
    assert "warnings" not in data and data["by_year"]["rows"]
