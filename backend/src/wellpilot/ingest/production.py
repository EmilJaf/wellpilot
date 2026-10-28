"""Clean the Volve daily production sheet into Parquet.

Every value the cleaning changes or flags is written to an issue log, so the agent can
cite exactly which raw readings were treated as missing and why.

Run: uv run python -m wellpilot.ingest.production
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wellpilot.config import get_settings

SOURCE_FILE = "Volve production data.xlsx"
DAILY_SHEET = "Daily Production Data"

DAILY_FILE = "production_daily.parquet"
ISSUES_FILE = "production_issues.parquet"
MANIFEST_FILE = "production_manifest.json"

# Raw column -> clean column. Units: volumes Sm3, pressures bar, temperatures degC, choke %.
RAW_COLUMNS = {
    "DATEPRD": "date",
    "NPD_WELL_BORE_NAME": "wellbore",
    "NPD_WELL_BORE_CODE": "npd_wellbore_code",
    "FLOW_KIND": "flow_kind",
    "WELL_TYPE": "well_type",
    "ON_STREAM_HRS": "on_stream_hrs",
    "BORE_OIL_VOL": "oil_sm3",
    "BORE_GAS_VOL": "gas_sm3",
    "BORE_WAT_VOL": "water_sm3",
    "BORE_WI_VOL": "water_inj_sm3",
    "AVG_DOWNHOLE_PRESSURE": "bhp_bar",
    "AVG_DOWNHOLE_TEMPERATURE": "bht_c",
    "AVG_DP_TUBING": "dp_tubing_bar",
    "AVG_WHP_P": "whp_bar",
    "AVG_WHT_P": "wht_c",
    "AVG_ANNULUS_PRESS": "annulus_press_bar",
    "AVG_CHOKE_SIZE_P": "choke_pct",
    "DP_CHOKE_SIZE": "dp_choke_bar",
}
VOLUME_COLUMNS = ("oil_sm3", "gas_sm3", "water_sm3", "water_inj_sm3")
# Carried onto a day the sheet leaves out: who the well is and how it was flowing.
IDENTITY_COLUMNS = ("well", "wellbore", "npd_wellbore_code", "flow_kind", "well_type")
ISSUE_COLUMNS = ["date", "well", "field", "issue", "raw_value", "action"]


@dataclass(frozen=True)
class CleanResult:
    daily: pd.DataFrame
    issues: pd.DataFrame


class _IssueLog:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._records: list[pd.DataFrame] = []

    def set_missing(self, mask: pd.Series, column: str, issue: str) -> None:
        self._record(mask, column, issue, "set_missing")
        self._df.loc[mask, column] = np.nan

    def keep(self, mask: pd.Series, column: str, issue: str) -> None:
        self._record(mask, column, issue, "kept")

    def _record(self, mask: pd.Series, column: str, issue: str, action: str) -> None:
        hits = self._df.loc[mask, ["date", "well", column]]
        if hits.empty:
            return
        self._records.append(
            pd.DataFrame(
                {
                    "date": hits["date"],
                    "well": hits["well"],
                    "field": column,
                    "issue": issue,
                    "raw_value": hits[column].astype(float),
                    "action": action,
                }
            )
        )

    def frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=ISSUE_COLUMNS)
        issues = pd.concat(self._records, ignore_index=True)
        return issues.sort_values(["well", "date", "field"]).reset_index(drop=True)


def _water_spikes(df: pd.DataFrame) -> pd.Series:
    """Days whose water is over 1000 Sm3 and 50 times the well's median of the 15 days
    around it, while its oil stays within 10% of theirs: a keying error, not water
    breaking in (F-12 2008-05-24/25: 8,019.7 Sm3 between days of 8-30 Sm3, at 5,000 Sm3/d
    of oil and an unchanged choke). Breakthrough builds over weeks and moves the oil."""
    spikes = pd.Series(False, index=df.index)
    for _, g in df.sort_values("date").groupby("well"):
        water, oil = g["water_sm3"], g["oil_sm3"]
        around = water.rolling(15, center=True, min_periods=5).median()
        oil_around = oil.rolling(15, center=True, min_periods=5).median()
        hit = (
            (water > 1000)
            & (water > 50 * around.clip(lower=1))
            & ((oil - oil_around).abs() <= 0.1 * oil_around)
        )
        spikes.loc[g.index] = hit.fillna(False)
    return spikes


def clean_daily(raw: pd.DataFrame) -> CleanResult:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"daily sheet is missing columns: {sorted(missing)}")

    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["well"] = df["wellbore"].str.removeprefix("15/9-")
    flowing = df["on_stream_hrs"] > 0
    production = df["flow_kind"] == "production"
    log = _IssueLog(df)

    # A gauge cannot read 0 bar or 0 degC downhole; zeros are dead or disconnected gauges
    # (F-12's downhole gauge reads 0 from 2010-10-12 while the well keeps flowing).
    for column in ("bhp_bar", "bht_c", "wht_c"):
        log.set_missing(df[column] == 0, column, "zero_gauge_reading")
    # A shut-in well can be bled down to 0 bar at the wellhead; a flowing one cannot.
    log.set_missing((df["whp_bar"] == 0) & flowing, "whp_bar", "zero_whp_while_flowing")
    # AVG_DP_TUBING is BHP minus WHP upstream; without a BHP reading it just echoes WHP.
    log.set_missing(
        df["bhp_bar"].isna() & df["dp_tubing_bar"].notna(), "dp_tubing_bar", "dp_tubing_without_bhp"
    )
    for column in VOLUME_COLUMNS:
        log.set_missing(df[column] < 0, column, "negative_volume")
    log.set_missing(_water_spikes(df), "water_sm3", "water_spike")

    # Kept as-is but flagged, so analyses can decide whether to exclude them.
    log.keep(df["on_stream_hrs"] > 24, "on_stream_hrs", "dst_25_hour_day")
    has_volume = (df["oil_sm3"] > 0) | (df["water_sm3"] > 0)
    log.keep(production & ~flowing & has_volume, "on_stream_hrs", "volume_without_uptime")
    log.keep(production & flowing & ~has_volume, "on_stream_hrs", "uptime_without_volume")

    df, inserted = _fill_calendar(df)

    liquid = df["oil_sm3"] + df["water_sm3"]
    df["water_cut"] = (df["water_sm3"] / liquid).where(liquid > 0)
    df["gor_sm3_sm3"] = (df["gas_sm3"] / df["oil_sm3"]).where(df["oil_sm3"] > 0)

    columns = ["date", "well", *[c for c in RAW_COLUMNS.values() if c != "date"]]
    columns += ["water_cut", "gor_sm3_sm3"]
    daily = df[columns].sort_values(["well", "date"]).reset_index(drop=True)
    issues = pd.concat([log.frame(), inserted], ignore_index=True)
    issues["date"] = pd.to_datetime(issues["date"])  # an empty log has no dtype
    issues = issues.sort_values(["well", "date", "field"]).reset_index(drop=True)
    return CleanResult(daily=daily, issues=issues)


def _fill_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Insert the days the sheet leaves out of each well's span, as days with no hours on
    stream and no volumes.

    The Monthly sheet's hours and volumes equal the sums of the daily rows, so a missing
    day produced nothing: it is a shut-in day, not an unknown one. Left out, it inflates
    uptime (F-12, January 2012: 0.88 instead of 0.54) and hides shut-ins (F-12 and F-14
    share 85 missing days, 2012-01-03..13 among them). Each inserted day is logged.
    """
    filled, inserted = [], []
    for _, rows in df.groupby("well", sort=True):
        rows = rows.set_index("date").sort_index()
        span = pd.date_range(rows.index.min(), rows.index.max(), freq="D", name="date")
        missing = span.difference(rows.index)
        if len(missing):
            rows = rows.reindex(span)
            new = rows.index.isin(missing)
            rows[list(IDENTITY_COLUMNS)] = rows[list(IDENTITY_COLUMNS)].ffill()
            rows.loc[new, ["on_stream_hrs", *VOLUME_COLUMNS]] = 0.0
            inserted.append(
                pd.DataFrame(
                    {
                        "date": missing,
                        "well": rows.loc[new, "well"].to_numpy(),
                        "field": "on_stream_hrs",
                        "issue": "missing_day",
                        "raw_value": np.nan,
                        "action": "inserted",
                    }
                )
            )
        filled.append(rows.reset_index())
    issues = (
        pd.concat(inserted, ignore_index=True) if inserted else pd.DataFrame(columns=ISSUE_COLUMNS)
    )
    return pd.concat(filled, ignore_index=True), issues


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(result: CleanResult, out_dir: Path, source: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(out_dir / DAILY_FILE, index=False)
    result.issues.to_parquet(out_dir / ISSUES_FILE, index=False)
    manifest = {
        "source_file": source.name,
        "source_sha256": file_sha256(source),
        "sheet": DAILY_SHEET,
        "rows": len(result.daily),
        "wells": sorted(result.daily["well"].unique().tolist()),
        "date_range": [
            result.daily["date"].min().date().isoformat(),
            result.daily["date"].max().date().isoformat(),
        ],
        "issues": {k: int(v) for k, v in result.issues.groupby("issue").size().items()},
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "licence": "Equinor Open Data Licence (see README)",
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Clean the Volve daily production sheet.")
    parser.add_argument(
        "--source", type=Path, default=settings.raw_dir / "production" / SOURCE_FILE
    )
    parser.add_argument("--out", type=Path, default=settings.processed_dir)
    args = parser.parse_args(argv)

    raw = pd.read_excel(args.source, sheet_name=DAILY_SHEET)
    manifest = write_outputs(clean_daily(raw), args.out, args.source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
