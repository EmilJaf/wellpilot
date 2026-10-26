import json
import math

import numpy as np
import pandas as pd
import pytest

from wellpilot.config import get_settings
from wellpilot.ingest.production import (
    DAILY_FILE,
    DAILY_SHEET,
    ISSUES_FILE,
    MANIFEST_FILE,
    SOURCE_FILE,
    clean_daily,
    file_sha256,
    write_outputs,
)

REAL_SOURCE = get_settings().raw_dir / "production" / SOURCE_FILE


def raw_rows(*overrides: dict) -> pd.DataFrame:
    """Raw daily rows shaped like the Volve sheet: a healthy flowing F-12 day by default."""
    base = {
        "DATEPRD": pd.Timestamp("2010-10-11"),
        "NPD_WELL_BORE_NAME": "15/9-F-12",
        "NPD_WELL_BORE_CODE": 5599,
        "FLOW_KIND": "production",
        "WELL_TYPE": "OP",
        "ON_STREAM_HRS": 24.0,
        "BORE_OIL_VOL": 1800.0,
        "BORE_GAS_VOL": 270_000.0,
        "BORE_WAT_VOL": 3600.0,
        "BORE_WI_VOL": np.nan,
        "AVG_DOWNHOLE_PRESSURE": 270.0,
        "AVG_DOWNHOLE_TEMPERATURE": 107.0,
        "AVG_DP_TUBING": 215.0,
        "AVG_WHP_P": 55.0,
        "AVG_WHT_P": 80.0,
        "AVG_ANNULUS_PRESS": 20.0,
        "AVG_CHOKE_SIZE_P": 50.0,
        "DP_CHOKE_SIZE": 10.0,
        "AVG_CHOKE_UOM": "%",
    }
    rows = overrides or ({},)
    return pd.DataFrame([{**base, **row} for row in rows])


def issues_for(result, issue: str) -> pd.DataFrame:
    return result.issues[result.issues["issue"] == issue]


def test_renames_columns_and_shortens_well_names() -> None:
    result = clean_daily(raw_rows({"NPD_WELL_BORE_NAME": "15/9-F-15 D"}))

    row = result.daily.iloc[0]
    assert row["well"] == "F-15 D"
    assert row["wellbore"] == "15/9-F-15 D"
    assert row["oil_sm3"] == 1800.0
    assert "AVG_CHOKE_UOM" not in result.daily.columns
    assert result.issues.empty


def test_zero_downhole_gauge_is_missing_and_logged() -> None:
    raw = raw_rows(
        {"AVG_DOWNHOLE_PRESSURE": 0.0, "AVG_DOWNHOLE_TEMPERATURE": 0.0, "AVG_DP_TUBING": 55.0}
    )

    result = clean_daily(raw)

    row = result.daily.iloc[0]
    assert math.isnan(row["bhp_bar"])
    assert math.isnan(row["bht_c"])
    assert math.isnan(row["dp_tubing_bar"])
    assert row["whp_bar"] == 55.0
    zero = issues_for(result, "zero_gauge_reading")
    assert set(zero["field"]) == {"bhp_bar", "bht_c"}
    assert (zero["raw_value"] == 0.0).all()
    assert (zero["action"] == "set_missing").all()
    echo = issues_for(result, "dp_tubing_without_bhp")
    assert echo["raw_value"].tolist() == [55.0]


def test_zero_whp_is_missing_only_while_flowing() -> None:
    raw = raw_rows(
        {"AVG_WHP_P": 0.0},
        {"AVG_WHP_P": 0.0, "ON_STREAM_HRS": 0.0, "DATEPRD": pd.Timestamp("2010-10-12")},
    )

    result = clean_daily(raw)

    assert math.isnan(result.daily.loc[0, "whp_bar"])
    assert result.daily.loc[1, "whp_bar"] == 0.0
    assert len(issues_for(result, "zero_whp_while_flowing")) == 1


def test_negative_volume_is_missing() -> None:
    result = clean_daily(raw_rows({"BORE_WAT_VOL": -14.19}))

    row = result.daily.iloc[0]
    assert math.isnan(row["water_sm3"])
    assert math.isnan(row["water_cut"])
    assert issues_for(result, "negative_volume")["raw_value"].tolist() == [-14.19]


def test_water_cut_and_gor() -> None:
    result = clean_daily(
        raw_rows({}, {"BORE_OIL_VOL": 0.0, "BORE_WAT_VOL": 0.0, "BORE_GAS_VOL": 0.0})
    )

    assert result.daily.loc[0, "water_cut"] == pytest.approx(2 / 3)
    assert result.daily.loc[0, "gor_sm3_sm3"] == pytest.approx(150.0)
    assert math.isnan(result.daily.loc[1, "water_cut"])
    assert math.isnan(result.daily.loc[1, "gor_sm3_sm3"])


def test_flagged_rows_keep_their_values() -> None:
    raw = raw_rows(
        {"ON_STREAM_HRS": 25.0},
        {"ON_STREAM_HRS": 0.0, "DATEPRD": pd.Timestamp("2010-10-12")},
        {"BORE_OIL_VOL": 0.0, "BORE_WAT_VOL": 0.0, "DATEPRD": pd.Timestamp("2010-10-13")},
    )

    result = clean_daily(raw)

    assert result.daily["on_stream_hrs"].tolist() == [25.0, 0.0, 24.0]
    assert len(issues_for(result, "dst_25_hour_day")) == 1
    assert len(issues_for(result, "volume_without_uptime")) == 1
    assert len(issues_for(result, "uptime_without_volume")) == 1
    assert (result.issues["action"] == "kept").all()


def test_injection_rows_are_not_flagged_for_missing_volumes() -> None:
    raw = raw_rows(
        {
            "NPD_WELL_BORE_NAME": "15/9-F-4",
            "FLOW_KIND": "injection",
            "WELL_TYPE": "WI",
            "BORE_OIL_VOL": np.nan,
            "BORE_GAS_VOL": np.nan,
            "BORE_WAT_VOL": np.nan,
            "BORE_WI_VOL": 5000.0,
            "AVG_DOWNHOLE_PRESSURE": np.nan,
            "AVG_DOWNHOLE_TEMPERATURE": np.nan,
            "AVG_DP_TUBING": np.nan,
            "AVG_WHP_P": np.nan,
            "AVG_WHT_P": np.nan,
        }
    )

    result = clean_daily(raw)

    assert result.issues.empty
    assert result.daily.loc[0, "water_inj_sm3"] == 5000.0


def test_rejects_sheet_with_missing_columns() -> None:
    with pytest.raises(ValueError, match="BORE_OIL_VOL"):
        clean_daily(raw_rows().drop(columns=["BORE_OIL_VOL"]))


def test_write_outputs_round_trip(tmp_path) -> None:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"not really excel")
    result = clean_daily(raw_rows({"AVG_DOWNHOLE_PRESSURE": 0.0}))

    manifest = write_outputs(result, tmp_path / "out", source)

    daily = pd.read_parquet(tmp_path / "out" / DAILY_FILE)
    issues = pd.read_parquet(tmp_path / "out" / ISSUES_FILE)
    on_disk = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert len(daily) == 1
    assert issues["issue"].tolist() == ["zero_gauge_reading", "dp_tubing_without_bhp"]
    assert on_disk == manifest
    assert manifest["source_sha256"] == file_sha256(source)
    assert manifest["issues"] == {"dp_tubing_without_bhp": 1, "zero_gauge_reading": 1}


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="Volve xlsx not downloaded (data/raw)")
def test_real_volve_sheet() -> None:
    result = clean_daily(pd.read_excel(REAL_SOURCE, sheet_name=DAILY_SHEET))
    daily = result.daily

    assert len(daily) == 15_634
    f12 = daily[daily["well"] == "F-12"].set_index("date")
    # Total F-12 oil is unchanged by cleaning: no F-12 oil value is negative.
    assert f12["oil_sm3"].sum() == pytest.approx(4_579_610, abs=1)
    # The downhole gauge died on 2010-10-12: readings before it survive, none after it.
    assert f12.loc["2010-10-11", "bhp_bar"] > 200
    assert f12.loc["2010-10-12":, "bhp_bar"].notna().sum() == 0
    assert f12.loc["2010-10-12":, "oil_sm3"].sum() > 0
