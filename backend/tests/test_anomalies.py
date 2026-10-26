import numpy as np
import pandas as pd
import pytest

from wellpilot.analysis.anomalies import (
    Event,
    gauge_failures,
    mark_field_wide,
    regime_changes,
    shut_ins,
    threshold_crossing,
    weekly_signal,
)
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production import DAILY_FILE

PROCESSED = get_settings().processed_dir


def flowing_well(days: int = 730, well: str = "F-12", seed: int = 0) -> pd.DataFrame:
    """A producer flowing 24 h a day at a gently declining rate with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=days, freq="D"),
            "well": well,
            "flow_kind": "production",
            "on_stream_hrs": 24.0,
            "oil_sm3": 2000.0 * np.exp(-0.0005 * t) * np.exp(rng.normal(0, 0.02, days)),
            "water_cut": 0.05 + rng.normal(0, 0.01, days),
            "bhp_bar": 250.0 + rng.normal(0, 1, days),
            "whp_bar": 55.0 + rng.normal(0, 0.5, days),
        }
    )


def zero_issues(daily: pd.DataFrame, mask: np.ndarray, column: str = "bhp_bar") -> pd.DataFrame:
    rows = daily.loc[mask]
    return pd.DataFrame(
        {
            "date": rows["date"],
            "well": rows["well"],
            "column": column,
            "issue": "zero_gauge_reading",
            "raw_value": 0.0,
            "action": "set_missing",
        }
    )


def test_weekly_signal_rejects_unknown_signals() -> None:
    with pytest.raises(ValueError, match="unknown signal"):
        weekly_signal(flowing_well(), "oil_sm3")


def test_regime_change_finds_a_water_cut_step() -> None:
    daily = flowing_well()
    daily.loc[365:, "water_cut"] += 0.55

    events = regime_changes(daily, "water_cut")

    assert len(events) == 1
    event = events[0]
    assert abs((event.start - pd.Timestamp("2011-01-01")).days) <= 14
    assert event.details["before_mean"] == pytest.approx(0.05, abs=0.01)
    assert event.details["after_mean"] == pytest.approx(0.60, abs=0.01)


def test_regime_change_on_oil_rate_reports_rates_not_logs() -> None:
    daily = flowing_well()
    daily.loc[400:, "oil_sm3"] *= 0.5

    events = regime_changes(daily, "oil_rate")

    assert len(events) == 1
    assert events[0].details["relative_change"] == pytest.approx(-0.5, abs=0.1)
    assert events[0].details["before_mean"] > 1000


def test_steady_signal_has_no_regime_change() -> None:
    assert regime_changes(flowing_well(), "whp_bar") == []


def test_threshold_crossing_is_sustained_not_a_single_day() -> None:
    daily = flowing_well(days=1000)
    daily["water_cut"] = np.linspace(0.0, 1.0, 1000)
    daily.loc[100, "water_cut"] = 0.9  # one noisy spike must not count

    event = threshold_crossing(daily, "water_cut", 0.5)

    day = (event.start - pd.Timestamp("2010-01-01")).days
    assert 500 <= day <= 520
    assert event.details["months_after_first_oil"] == pytest.approx(day / 30.44, abs=0.1)


def test_threshold_crossing_none_and_below() -> None:
    daily = flowing_well()

    assert threshold_crossing(daily, "water_cut", 0.5) is None
    assert threshold_crossing(daily, "oil_rate", 1900, direction="below") is not None


def test_gauge_failure_spans_shut_ins_and_counts_flowing_days() -> None:
    daily = flowing_well(days=100)
    dead = np.arange(100) >= 40
    daily.loc[dead, "bhp_bar"] = np.nan
    daily.loc[60:64, ["on_stream_hrs", "oil_sm3"]] = 0.0
    issues = zero_issues(daily, dead & ~daily.index.isin(range(60, 65)))

    events = gauge_failures(daily, issues)

    assert len(events) == 1
    event = events[0]
    assert event.start == pd.Timestamp("2010-02-10")
    assert event.end == pd.Timestamp("2010-04-10")
    assert event.details["flowing_days"] == 55
    assert event.details["last_good_date"] == pd.Timestamp("2010-02-09")
    assert "sensor failure" in event.summary


def test_a_good_reading_splits_gauge_runs() -> None:
    daily = flowing_well(days=100)
    dead = (np.arange(100) >= 20) & (np.arange(100) != 50)
    daily.loc[dead, "bhp_bar"] = np.nan
    issues = zero_issues(daily, dead)

    events = gauge_failures(daily, issues)

    assert [(e.start.day, e.end.day) for e in events] == [(21, 19), (21, 10)]


def test_gauge_reading_zero_while_shut_in() -> None:
    daily = flowing_well(days=60)
    off = np.arange(60) >= 30
    daily.loc[off, ["on_stream_hrs", "oil_sm3"]] = 0.0
    daily.loc[off, "bhp_bar"] = np.nan

    events = gauge_failures(daily, zero_issues(daily, off))

    assert events[0].details["flowing_days"] == 0
    assert "shut in" in events[0].summary


def test_shut_in_tolerates_a_missing_row_and_compares_rates() -> None:
    daily = flowing_well(days=200)
    daily.loc[100:109, ["on_stream_hrs", "oil_sm3"]] = 0.0
    daily.loc[110:, "oil_sm3"] *= 1.5
    daily = daily.drop(index=105)

    events = shut_ins(daily, min_days=3)

    assert len(events) == 1
    assert events[0].details["days"] == 10
    assert events[0].details["relative_change"] == pytest.approx(0.5, abs=0.1)


def test_short_stops_are_not_shut_ins() -> None:
    daily = flowing_well(days=100)
    daily.loc[50:51, ["on_stream_hrs", "oil_sm3"]] = 0.0

    assert shut_ins(daily, min_days=3) == []


def _event(well: str, kind: str, start: str, end: str) -> Event:
    return Event(kind, well, "on_stream_hrs", pd.Timestamp(start), pd.Timestamp(end), {}, "x.")


def test_mark_field_wide() -> None:
    events = [
        _event("F-12", "shut_in", "2011-10-19", "2011-11-04"),
        _event("F-14", "shut_in", "2011-10-20", "2011-11-04"),
        _event("F-12", "shut_in", "2013-03-13", "2013-03-24"),
        _event("F-12", "regime_change", "2011-10-19", "2011-10-19"),
    ]

    marked = mark_field_wide(events)

    assert marked[0].details == {"field_wide": True, "also_shut_in": ["F-14"]}
    assert marked[1].details["also_shut_in"] == ["F-12"]
    assert marked[2].details["field_wide"] is False
    assert marked[3] is events[3]


@pytest.mark.skipif(not (PROCESSED / DAILY_FILE).exists(), reason="run the ingest first")
def test_real_volve_events() -> None:
    store = ProductionStore(PROCESSED)
    f12, f14 = store.well_daily("F-12"), store.well_daily("F-14")

    gauge = gauge_failures(f12, store.well_issues("F-12"))
    assert [(g.signal, g.start) for g in gauge] == [("bhp_bar", pd.Timestamp("2010-10-12"))]
    assert gauge[0].details["flowing_days"] > 1500

    wc80 = threshold_crossing(f12, "water_cut", 0.8)
    assert wc80.start.strftime("%Y-%m") == "2012-02"

    marked = mark_field_wide(shut_ins(f12, min_days=7) + shut_ins(f14, min_days=7))
    oct_2011 = [e for e in marked if e.start == pd.Timestamp("2011-10-19")]
    assert {e.well for e in oct_2011} == {"F-12", "F-14"}
    assert all(e.details["field_wide"] for e in oct_2011)
