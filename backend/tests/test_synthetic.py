"""The deterministic tools on a synthetic field with planted events (wellpilot.synthetic):
what they must find, and what they must not invent."""

import json
import warnings

import pandas as pd
import pytest

from wellpilot.agent.tools import execute
from wellpilot.data.store import ProductionStore
from wellpilot.synthetic import write


@pytest.fixture(scope="module")
def field(tmp_path_factory) -> tuple[ProductionStore, dict]:
    out = tmp_path_factory.mktemp("synthetic") / "processed"
    truth = write(out)["truth"]
    return ProductionStore(out), truth


def events(store: ProductionStore, well: str) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return execute(store, "detect_events", json.dumps({"well": well}))["result"]


def test_the_ingest_catches_the_planted_sheet_errors(field) -> None:
    store, _ = field
    issues = store.query("SELECT issue, count(*) AS n FROM production_issues GROUP BY 1").rows
    found = dict(zip(issues["issue"], issues["n"], strict=True))

    assert found["missing_day"] == 6 and found["negative_volume"] == 2
    assert found["zero_whp_while_flowing"] == 1


def test_the_dead_gauge_is_found_on_its_day(field) -> None:
    store, _ = field
    gauge = [e for e in events(store, "P-1") if e["kind"] == "gauge_failure"]

    assert [e["start"] for e in gauge] == ["2018-04-10"]


def test_water_breakthrough_is_found_within_a_month(field) -> None:
    store, truth = field
    planted = next(e for e in truth["events"] if e["kind"] == "water_breakthrough")
    found = {
        e["details"]["threshold"]: e["start"]
        for e in events(store, "P-1")
        if e["kind"] == "threshold_crossing"
    }
    for cut, day in planted["water_cut_crosses"].items():
        lag = (pd.Timestamp(found[float(cut)]) - pd.Timestamp(day)).days
        assert 0 <= lag <= 31, (cut, day, found[float(cut)])


def test_the_field_wide_shut_in_and_the_stopped_injector(field) -> None:
    store, _ = field
    shut = [e for e in events(store, "P-3") if e["kind"] == "shut_in"]
    screen = execute(store, "field_screen", "{}")["result"]
    injector = next(w for w in screen["wells"] if w["well"] == "I-2")

    assert [(e["start"], e["end"]) for e in shut] == [("2019-06-03", "2019-06-16")]
    assert shut[0]["details"]["field_wide"]
    assert [i["kind"] for i in injector["issues"]] == ["stopped"]


def test_the_mechanical_step_down_is_found_in_its_week(field) -> None:
    store, _ = field
    steps = [e["start"] for e in events(store, "P-2") if e["kind"] == "regime_change"]

    assert any("2018-11-01" <= s <= "2018-11-15" for s in steps)


def test_depletion_shows_no_water_and_falling_pressure(field) -> None:
    store, _ = field
    result = execute(store, "waterflood", json.dumps({"well": "P-3"}))["result"]
    monthly = store.query(
        "SELECT min(bhp_bar) AS low, max(bhp_bar) AS high, max(water_cut) AS wc "
        "FROM production_daily WHERE well = 'P-3' AND on_stream_hrs > 12"
    ).rows.iloc[0]

    assert result["wor_fit"] is None and "too few months" in result["no_trend"]
    assert monthly["wc"] < 0.03 and monthly["high"] - monthly["low"] > 100


@pytest.mark.xfail(strict=True, reason="regime changes are found on smooth declines too")
def test_a_smooth_decline_has_no_regime_change(field) -> None:
    """P-3 and P-4 follow a smooth Arps decline: nothing happened to them. The detector
    segments a decline into steps (P-3: -50% and -46%, P-4: -31%), and the screen flags P-4
    for a step down."""
    store, _ = field
    for well in ("P-3", "P-4"):
        oil = [e for e in events(store, well) if e["kind"] == "regime_change"]
        assert oil == [], well
