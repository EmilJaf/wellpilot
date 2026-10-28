"""Hindsight on the real Volve fixture: a forecast made with data up to 2010-12-31, set
against the oil F-12 actually produced in 2011 and the classic fit's forecast."""

import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from wellpilot.agent.tools import execute
from wellpilot.analysis.decline import fit_decline, forecast
from wellpilot.data.store import ProductionStore
from wellpilot.hindsight import hindsight

PROCESSED = Path(__file__).parent / "fixtures" / "processed"
CUT = "2010-12-31"


@pytest.fixture(scope="module")
def stores() -> tuple[ProductionStore, ProductionStore]:
    return ProductionStore(PROCESSED), ProductionStore(PROCESSED, as_of=date.fromisoformat(CUT))


def run_with_forecast(past: ProductionStore, **args) -> dict:
    """A saved run as the API keeps it, with one forecast_production evidence item."""
    result = execute(past, "forecast_production", json.dumps(args))["result"]
    return {
        "question": "Which wells need attention?",
        "as_of": CUT,
        "verified": True,
        "result": {"summary": "F-12 first.", "findings": []},
        "evidence": {
            "E1": {"id": "E1", "tool": "forecast_production", "arguments": args, "result": result}
        },
    }


def test_a_forecast_from_the_end_of_2010_against_2011(stores) -> None:
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-07-18")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = hindsight(run, full, past)

    (f,) = h["forecasts"]
    daily = full.well_daily("F-12")
    year = daily["date"].between(pd.Timestamp("2011-01-01"), pd.Timestamp("2011-12-31"))
    assert f["actual_sm3"] == round(daily.loc[year, "oil_sm3"].sum())
    assert 330_000 < f["actual_sm3"] < 350_000
    agent, classic = f["methods"]
    reported = run["evidence"]["E1"]["result"]["cumulative_oil_sm3"]
    assert agent["p50_sm3"] == round(reported["p50"]) and agent["window_start"] == "2010-07-18"
    # The classic fit over the whole history, plateau included, about doubles 2011.
    assert classic["error"] > 0.8 and abs(agent["error"]) < 0.15
    # The fair baselines: fits on the last 3, 6, 12 and 24 months, made the same day, none
    # of them as far off as the whole history.
    fixed = {w["months"]: w["error"] for w in f["fixed_windows"]}
    assert set(fixed) == {3, 6, 12, 24}
    assert all(abs(e) < abs(classic["error"]) for e in fixed.values())
    # The data after the cut is there now: weekly rates run to the end of the forecast.
    assert f["forecast_from"] == "2011-01-01" and f["horizon_end"] == "2011-12-31"
    assert f["rate"]["date"][-1] > "2011-12-01" and f["data_ends"] is None


def test_the_drawn_agent_curve_is_the_fit_the_agent_reported(stores) -> None:
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-07-18")
    fit = fit_decline(past.well_daily("F-12", "2010-07-18")).best
    p50 = forecast(fit, 365, start=pd.Timestamp("2011-01-01")).cumulative["p50"]
    assert p50 == pytest.approx(run["evidence"]["E1"]["result"]["cumulative_oil_sm3"]["p50"])


def test_every_well_gets_its_next_year(stores) -> None:
    full, past = stores
    run = {**run_with_forecast(past, well="F-12"), "evidence": {}}
    wells = {w["well"]: w for w in hindsight(run, full, past)["wells"]}

    assert set(wells) == {"F-12", "F-14"}  # the only producers by the end of 2010
    f12 = wells["F-12"]
    assert f12["gauge_dead_at_cut"] and f12["gauge_back_on"] is None  # never came back
    assert f12["oil_year_after_sm3"] < 0.5 * f12["oil_year_before_sm3"]
    assert f12["water_cut_year_later"] > f12["water_cut_at_cut"]


def test_a_run_that_saw_all_the_data_has_no_hindsight(stores) -> None:
    full, past = stores
    with pytest.raises(ValueError, match="saw all the data"):
        hindsight({"as_of": None, "evidence": {}}, full, past)


def test_the_forecast_the_answer_cites_comes_first(stores) -> None:
    """Run 8aef8d548380 forecast twice and answered with the first; the reveal showed both,
    the unused one (band 290k-725k) right under the answer's, as if it were a second answer."""
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-07-18")
    tried = execute(
        past, "forecast_production", json.dumps({"well": "F-12", "start": "2010-09-01"})
    )
    run["evidence"]["E2"] = {
        "id": "E2",
        "tool": "forecast_production",
        "arguments": {"well": "F-12", "start": "2010-09-01"},
        "result": tried["result"],
    }
    run["result"]["findings"] = [{"claim": "F-12 makes about 360k.", "evidence_ids": ["E2"]}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = hindsight(run, full, past)

    assert [(f["evidence_id"], f["cited"]) for f in h["forecasts"]] == [("E2", True), ("E1", False)]


def test_the_reveal_says_when_the_agents_window_starts_at_a_change_it_found(stores) -> None:
    """The fair story (the outside review): not '+6% against a whole-history fit', but the
    agent dating its own window from the shift it had detected."""
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-05-02")
    run["evidence"]["E2"] = {
        "id": "E2",
        "tool": "detect_events",
        "arguments": {"well": "F-12"},
        "result": [
            {"kind": "regime_change", "signal": "water_cut", "start": "2010-02-28"},
            {"kind": "regime_change", "signal": "oil_rate", "start": "2010-05-02", "summary": "x"},
        ],
    }
    other = run_with_forecast(past, well="F-12", start="2010-07-18")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (found,) = hindsight(run, full, past)["forecasts"]
        (fixed,) = hindsight(other, full, past)["forecasts"]

    assert found["window_from_change"] == {
        "date": "2010-05-02",
        "evidence_id": "E2",
        "kind": "change",
        "text": "x",
    }
    assert fixed["window_from_change"] is None


def test_a_window_that_starts_after_work_on_the_well_says_so(stores) -> None:
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-09-23")
    run["evidence"]["E2"] = {
        "id": "E2",
        "tool": "well_operations",
        "arguments": {"well": "F-12"},
        "result": {"jobs": [{"start": "2010-09-20", "end": "2010-09-23", "kind": "workover"}]},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (f,) = hindsight(run, full, past)["forecasts"]

    assert f["window_from_change"]["kind"] == "work"
    assert f["window_from_change"]["text"] == "workover on the well, 2010-09-20 to 2010-09-23"


def test_the_error_splits_into_rate_and_time_on_stream(stores) -> None:
    """The reservoir engineer's point: F-12's +6% for 2011 was the rate per producing day
    too low and the time on stream too high, cancelling. The reveal says so."""
    full, past = stores
    run = run_with_forecast(past, well="F-12", start="2010-05-02")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (f,) = hindsight(run, full, past)["forecasts"]

    d = f["decomposition"]
    combined = (1 + d["rate_error"]) * (1 + d["uptime_error"]) - 1
    assert abs(combined - f["methods"][0]["error"]) < 0.01
    assert d["uptime_actual"] < d["uptime_forecast"]  # 2011 had more time shut in


def test_the_work_done_after_the_date_and_its_worth() -> None:
    """Asked at the end of October 2014, F-12's next job was the plug at 3150 m and the new
    perforations of 2014-11-22..27: water cut 0.97 to about 0.34, oil 177 to about 860
    Sm3/d, and the year after it well above the decline it was on."""
    from wellpilot.hindsight import _work_next

    full = ProductionStore(Path(__file__).parent / "fixtures" / "processed")

    (job,) = [j for j in _work_next(full, pd.Timestamp("2014-10-31"), ["F-12"])]

    assert (job["start"], job["end"], job["plug_m"]) == ("2014-11-22", "2014-11-27", 3150.0)
    assert job["before"]["water_cut"] > 0.95 and job["after"]["water_cut"] < 0.4
    assert job["before"]["oil_sm3_per_day"] < 200 < 800 < job["after"]["oil_sm3_per_day"]
    assert job["extra_oil_sm3"] > 100_000 and job["price_usd_per_bbl"] == 50.0
    assert job["extra_value_usd"] == round(job["extra_oil_sm3"] * 6.2898 * 50, -5)
