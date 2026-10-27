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
