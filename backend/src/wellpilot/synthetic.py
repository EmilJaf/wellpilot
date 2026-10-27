"""A synthetic oil field with planted events, to see how the agent does on data that is not
Volve and whose answers are known.

Six wells over 2015-2020, written as a raw daily sheet in Volve's columns and cleaned by the
same ingest (zero gauge readings, missing days and negative volumes go through the real
rules). What was planted is written to truth.json beside the data:

- P-1, waterflooded: water breaks through in the second half of 2016 and takes the well's
  liquid capacity (liquid held, choke opened, oil falls); its downhole gauge dies on
  2018-04-10 and reads 0 from then on; after injector I-2 stops on 2019-09-01 its liquid,
  and so its oil, falls faster.
- P-2: a mechanical step down on 2018-11-05: the choke is cut from 70% to 40%, oil and
  liquid fall 40% and the downhole pressure rises (less drawdown), water cut is unchanged.
  An operational change, not the reservoir.
- P-3: depletion in a block no injector reaches: water cut stays near 1%, downhole pressure
  falls from 280 to about 150 bar and the gas-oil ratio rises once it drops below the bubble
  point (180 bar).
- P-4: a late well, first oil 2019-04-01.
- I-1, I-2: water injectors from 2015-06-01; I-2 stops on 2019-09-01.
- The whole field is shut in from 2019-06-03 to 2019-06-16.
- Data quirks: six days missing from P-2's sheet, two negative water volumes on P-1, a
  wellhead pressure of 0 while P-4 flows.

No reports: the report and plan tools have nothing to read, and field.json tells the agent.

Run: uv run python -m wellpilot.synthetic [--out data/synthetic/processed]
Then point any command at it with DATA_DIR=data/synthetic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wellpilot.config import REPO_ROOT
from wellpilot.ingest.production import (
    DAILY_FILE,
    ISSUES_FILE,
    MANIFEST_FILE,
    RAW_COLUMNS,
    clean_daily,
)

START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")
FIELD_SHUT_IN = (pd.Timestamp("2019-06-03"), pd.Timestamp("2019-06-16"))
DEAD_GAUGE = pd.Timestamp("2018-04-10")
STEP_DOWN = pd.Timestamp("2018-11-05")
INJECTOR_STOPS = pd.Timestamp("2019-09-01")
BUBBLE_POINT_BAR = 180.0
FIELD = {
    "name": "test oil field (synthetic production data, no reports)",
    "drive": "The field is waterflooded by two injectors, I-1 and I-2; whether a producer is "
    "supported by them shows in its data.",
}


def _arps(t: np.ndarray, qi: float, di_per_year: float, b: float) -> np.ndarray:
    d = di_per_year / 365.25
    return qi * np.exp(-d * t) if b == 0 else qi / (1 + b * d * t) ** (1 / b)


def _logistic(days: pd.DatetimeIndex, middle: pd.Timestamp, width_days: float, top: float):
    t = (days - middle).days.to_numpy().astype(float)
    return top / (1 + np.exp(-t / width_days))


def _hours(days: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Hours on stream: mostly full days, some partial ones, a few short shut-ins, and the
    field-wide shut-in."""
    hours = np.full(len(days), 24.0)
    partial = rng.random(len(days)) < 0.05
    hours[partial] = rng.uniform(12, 22, partial.sum())
    for start in rng.choice(len(days), size=max(1, len(days) // 120), replace=False):
        hours[start : start + rng.integers(1, 4)] = 0.0
    shut = (days >= FIELD_SHUT_IN[0]) & (days <= FIELD_SHUT_IN[1])
    hours[shut] = 0.0
    return hours


def _frame(well: str, code: int, days, kind: str, **columns) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "DATEPRD": days,
            "NPD_WELL_BORE_NAME": well,
            "NPD_WELL_BORE_CODE": code,
            "FLOW_KIND": kind,
            "WELL_TYPE": "WI" if kind == "injection" else "OP",
        }
    )
    for raw_name, clean in RAW_COLUMNS.items():
        if clean in columns:
            raw[raw_name] = columns[clean]
        elif raw_name not in raw:
            raw[raw_name] = np.nan
    return raw


def _producer(
    well: str,
    code: int,
    first_oil: pd.Timestamp,
    rng: np.random.Generator,
    liquid: np.ndarray,
    water_cut: np.ndarray,
    choke: np.ndarray,
    bhp: np.ndarray,
    gor: np.ndarray,
) -> pd.DataFrame:
    days = pd.date_range(first_oil, END, freq="D")
    hours = _hours(days, rng)
    noise = rng.lognormal(0, 0.05, len(days))
    produced = liquid * noise * hours / 24
    oil = produced * (1 - water_cut)
    water = produced * water_cut
    flowing = hours > 0
    whp = np.where(flowing, 25 + 0.12 * bhp + rng.normal(0, 1.5, len(days)), 85.0)
    return _frame(
        well,
        code,
        days,
        "production",
        on_stream_hrs=hours,
        oil_sm3=oil.round(2),
        water_sm3=water.round(2),
        gas_sm3=(oil * gor).round(0),
        water_inj_sm3=np.nan,
        bhp_bar=(bhp + rng.normal(0, 1.0, len(days))).round(2),
        bht_c=np.full(len(days), 96.0),
        dp_tubing_bar=(bhp - whp).round(2),
        whp_bar=whp.round(2),
        wht_c=np.where(flowing, 62.0, 30.0),
        annulus_press_bar=np.full(len(days), 20.0),
        choke_pct=np.where(flowing, choke, 0.0).round(2),
        dp_choke_bar=np.where(flowing, whp - 15, 0.0).round(2),
    )


def generate(seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """The raw daily sheet (Volve's columns) and the truth of what was planted."""
    rng = np.random.default_rng(seed)
    sheets, truth = [], {"events": []}

    # P-1: waterflooded; breakthrough, liquid held by opening the choke, dead gauge, and a
    # faster liquid decline once I-2 stops.
    first = pd.Timestamp("2015-01-15")
    days = pd.date_range(first, END, freq="D")
    t = (days - first).days.to_numpy().astype(float)
    liquid = np.full(len(days), 3300.0) * np.exp(-0.03 / 365.25 * t)
    after = (days - INJECTOR_STOPS).days.to_numpy().astype(float)
    liquid = np.where(after > 0, liquid * np.exp(-0.5 / 365.25 * after), liquid)
    middle = pd.Timestamp("2017-07-01")
    water_cut = 0.003 + _logistic(days, middle, 120, 0.90)
    choke = 40 + 60 * np.clip(water_cut / 0.8, 0, 1)
    bhp = 245 - 0.004 * t
    bhp = np.where(after > 0, bhp - 0.08 * after, bhp)
    p1 = _producer(
        "P-1", 9001, first, rng, liquid, water_cut, choke, bhp, np.full(len(days), 150.0)
    )
    p1.loc[p1["DATEPRD"] >= DEAD_GAUGE, "AVG_DOWNHOLE_PRESSURE"] = 0.0
    for day in ("2016-02-03", "2017-09-21"):  # sheet errors the ingest must catch
        p1.loc[p1["DATEPRD"] == day, "BORE_WAT_VOL"] = -5.0
    sheets.append(p1)
    crossings = {
        cut: days[np.argmax(water_cut >= cut)].date().isoformat() for cut in (0.05, 0.5, 0.8)
    }
    truth["events"] += [
        {"well": "P-1", "kind": "water_breakthrough", "water_cut_crosses": crossings},
        {"well": "P-1", "kind": "dead_gauge", "start": DEAD_GAUGE.date().isoformat()},
        {
            "well": "P-1",
            "kind": "faster_decline",
            "start": INJECTOR_STOPS.date().isoformat(),
            "cause": "injector I-2 stopped; liquid then falls about 40% a year",
        },
    ]

    # P-2: decline, a mechanical step down (choke cut), little water.
    first = pd.Timestamp("2015-03-01")
    days = pd.date_range(first, END, freq="D")
    t = (days - first).days.to_numpy().astype(float)
    step = days >= STEP_DOWN
    oil_rate = _arps(t, 2600, 0.45, 0.5)
    water_cut = 0.01 + _logistic(days, pd.Timestamp("2020-06-01"), 150, 0.4)
    liquid = oil_rate / (1 - water_cut) * np.where(step, 0.6, 1.0)
    choke = np.where(step, 40.0, 70.0)
    bhp = 230 - 0.02 * t + np.where(step, 25.0, 0.0)
    p2 = _producer(
        "P-2", 9002, first, rng, liquid, water_cut, choke, bhp, np.full(len(days), 145.0)
    )
    gone = rng.choice(len(p2), size=6, replace=False)  # days left out of the sheet
    sheets.append(p2.drop(p2.index[gone]))
    truth["events"].append(
        {
            "well": "P-2",
            "kind": "step_down",
            "start": STEP_DOWN.date().isoformat(),
            "oil_change": -0.4,
            "cause": "choke cut from 70% to 40%; downhole pressure rose 25 bar; water cut "
            "unchanged",
        }
    )

    # P-3: depletion, no water; pressure falls, gas comes out below the bubble point.
    first = pd.Timestamp("2016-02-01")
    days = pd.date_range(first, END, freq="D")
    t = (days - first).days.to_numpy().astype(float)
    oil_rate = _arps(t, 1800, 0.55, 0.3)
    bhp = 280 - 150 * (1 - np.exp(-t / 900))
    gor = 150 + np.clip(BUBBLE_POINT_BAR - bhp, 0, None) * 6
    p3 = _producer(
        "P-3",
        9003,
        first,
        rng,
        oil_rate / 0.99,
        np.full(len(days), 0.01),
        np.full(len(days), 55.0),
        bhp,
        gor,
    )
    sheets.append(p3)
    below = days[np.argmax(bhp < BUBBLE_POINT_BAR)].date().isoformat()
    truth["events"].append(
        {
            "well": "P-3",
            "kind": "depletion",
            "bhp_start_bar": 280,
            "bhp_end_bar": round(float(bhp[-1])),
            "water_cut": 0.01,
            "below_bubble_point": below,
            "cause": "no injector support: pressure depletion, not water",
        }
    )

    # P-4: a late well.
    first = pd.Timestamp("2019-04-01")
    days = pd.date_range(first, END, freq="D")
    t = (days - first).days.to_numpy().astype(float)
    p4 = _producer(
        "P-4",
        9004,
        first,
        rng,
        _arps(t, 1600, 0.5, 0.4) / 0.98,
        np.full(len(days), 0.02),
        np.full(len(days), 60.0),
        np.full(len(days), 250.0) - 0.01 * t,
        np.full(len(days), 150.0),
    )
    flowing = p4["ON_STREAM_HRS"] > 0
    p4.loc[p4.index[flowing][10], "AVG_WHP_P"] = 0.0
    sheets.append(p4)
    truth["events"].append({"well": "P-4", "kind": "first_oil", "start": first.date().isoformat()})

    # Injectors.
    for well, code, stop in (("I-1", 9101, None), ("I-2", 9102, INJECTOR_STOPS)):
        days = pd.date_range("2015-06-01", END, freq="D")
        hours = _hours(days, rng)
        if stop is not None:
            hours[days >= stop] = 0.0
        injected = rng.normal(3400, 150, len(days)) * hours / 24
        sheets.append(
            _frame(
                well,
                code,
                days,
                "injection",
                on_stream_hrs=hours,
                water_inj_sm3=injected.round(0),
                bhp_bar=np.where(hours > 0, 320.0, np.nan),  # a stopped well reads nothing
                whp_bar=np.where(hours > 0, 160.0, np.nan),
                choke_pct=np.where(hours > 0, 100.0, 0.0),
            )
        )
    truth["events"].append(
        {"well": "I-2", "kind": "injector_stopped", "start": INJECTOR_STOPS.date().isoformat()}
    )
    truth["events"].append(
        {
            "kind": "field_shut_in",
            "start": FIELD_SHUT_IN[0].date().isoformat(),
            "end": FIELD_SHUT_IN[1].date().isoformat(),
        }
    )
    truth["quirks"] = [
        "6 days missing from P-2's sheet",
        "negative water volumes on P-1 on 2016-02-03 and 2017-09-21",
        "a wellhead pressure of 0 while P-4 flows",
    ]
    return pd.concat(sheets, ignore_index=True), truth


def write(out: Path, seed: int = 7) -> dict:
    """Generate, clean with the real ingest and write the processed files, field.json and
    truth.json (beside the processed directory)."""
    raw, truth = generate(seed)
    result = clean_daily(raw)
    out.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(out / DAILY_FILE, index=False)
    result.issues.to_parquet(out / ISSUES_FILE, index=False)
    manifest = {
        "source_file": f"synthetic, seed {seed}",
        "rows": len(result.daily),
        "wells": sorted(result.daily["well"].unique().tolist()),
        "issues": {k: int(v) for k, v in result.issues.groupby("issue").size().items()},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "field.json").write_text(json.dumps(FIELD, indent=2) + "\n")
    (out.parent / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return {**manifest, "truth": truth}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic field with planted events.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synthetic" / "processed")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    print(json.dumps(write(args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
