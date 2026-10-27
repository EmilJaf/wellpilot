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
