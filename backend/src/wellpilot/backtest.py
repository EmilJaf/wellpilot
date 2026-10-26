"""Backtest: how well do forecasts made at a past date match what the wells then produced?

Rolling origin: for every producer, a cutoff every 6 months (from 18 months after first
oil), a forecast of the next 365 days of oil from data up to the cutoff only, compared
with the actual volume. Methods:

- classic: Arps (auto model) fitted on the well's whole history, the quick standard.
- recent_12m: Arps fitted on the last 12 months only.
- agent: the WellPilot agent, given data up to the cutoff, picks its own window and
  runs forecast_production; its forecast is read from its evidence. This one calls the
  LLM (spends tokens), so it runs on a subset of cases and is cached per case.

Uncertainty: the decline module's bootstrap bands are far too narrow on real wells. A
calibrated band is built from each method's own errors on earlier cases (only those
whose forecast period had ended by the cutoff), and both bands' coverage is reported.

Run: uv run python -m wellpilot.backtest          (free)
     uv run python -m wellpilot.backtest --agent  (spends tokens, cached per case)
"""

import argparse
import asyncio
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wellpilot.agent.loop import run_agent
from wellpilot.agent.tools import fit_warnings, to_jsonable
from wellpilot.analysis.decline import fit_decline, forecast
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore

HORIZON_DAYS = 365
STEP_MONTHS = 6
FIRST_CUTOFF_MONTHS = 18
PRODUCERS = ("F-12", "F-14", "F-11", "F-15 D", "F-1 C")
BAND = (0.1, 0.9)  # calibrated P90-P10: the 80% interval
MIN_PAST_ERRORS = 5  # errors needed before a calibrated band is given
RESULTS_FILE = "backtest_results.parquet"
SUMMARY_FILE = "backtest_summary.json"

AGENT_QUESTION = (
    "Forecast oil production for well {well} for the {days} days after {cutoff}. Look at "
    "the well's recent behaviour, choose the history window you trust for a decline fit, "
    "run forecast_production with horizon_days={days} on that window, and report the P50 "
    "cumulative oil in your findings."
)


@dataclass(frozen=True)
class Case:
    well: str
    cutoff: pd.Timestamp
    actual_sm3: float


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.to_period("M").to_timestamp("M")


def cases(store: ProductionStore) -> list[Case]:
    found = []
    for well in PRODUCERS:
        daily = store.well_daily(well)
        oil = daily[(daily["flow_kind"] == "production") & (daily["oil_sm3"] > 0)]
        if oil.empty:
            continue
        cutoff = month_end(oil["date"].min() + pd.DateOffset(months=FIRST_CUTOFF_MONTHS))
        while cutoff + pd.Timedelta(days=HORIZON_DAYS) <= oil["date"].max():
            future = oil[
                (oil["date"] > cutoff) & (oil["date"] <= cutoff + pd.Timedelta(days=HORIZON_DAYS))
            ]
            found.append(Case(well, cutoff, float(future["oil_sm3"].sum())))
            cutoff = month_end(cutoff + pd.DateOffset(months=STEP_MONTHS))
    return found


def _decline_row(
    case: Case, daily: pd.DataFrame, method: str, start=None, end=None, model: str = "auto"
) -> dict:
    """Fit on data up to the cutoff and forecast exactly the 365 days after it, the same
    days the actual volume covers."""
    history = daily[daily["date"] <= case.cutoff]
    fit = fit_decline(history, start=start, end=end, model=model).best
    fc = forecast(fit, HORIZON_DAYS, n_boot=100, start=case.cutoff + pd.Timedelta(days=1))
    return {
        "well": case.well,
        "cutoff": case.cutoff,
        "method": method,
        "window_start": fit.t0,
        "p90": fc.cumulative["p90"],
        "p50": fc.cumulative["p50"],
        "p10": fc.cumulative["p10"],
        "actual": case.actual_sm3,
        "warned": bool(fit_warnings(fit)),
    }


def _production(store: ProductionStore, well: str) -> pd.DataFrame:
    daily = store.well_daily(well)
    return daily[daily["flow_kind"] == "production"]


def decline_methods(store: ProductionStore, all_cases: list[Case]) -> list[dict]:
    rows = []
    for case in all_cases:
        daily = _production(store, case.well)
        rows.append(_decline_row(case, daily, "classic"))
        recent = case.cutoff - pd.DateOffset(months=12)
        rows.append(_decline_row(case, daily, "recent_12m", start=recent))
    return rows


def _agent_cache(processed_dir: Path, case: Case) -> Path:
    name = f"{case.well.replace(' ', '_')}_{case.cutoff.date().isoformat()}.json"
    return processed_dir / "backtest_agent" / name


async def agent_row(
    llm, processed_dir: Path, case: Case, store: ProductionStore | None = None
) -> dict | None:
    """The agent's forecast for a case: its run is cached, and its chosen window and model
    are refitted here on data up to the cutoff, so the forecast covers exactly the same
    365 days as the other methods. With llm=None only cached runs are used."""
    path = _agent_cache(processed_dir, case)
    if path.exists():
        record = json.loads(path.read_text())
    elif llm is None:
        return None
    else:
        as_of_store = ProductionStore(processed_dir, as_of=case.cutoff.date())
        question = AGENT_QUESTION.format(
            well=case.well, days=HORIZON_DAYS, cutoff=case.cutoff.date().isoformat()
        )
        run = await run_agent(question, llm, as_of_store)
        forecasts = [
            e
            for e in run.evidence
            if e.tool == "forecast_production"
            and e.arguments.get("well") == case.well
            and e.arguments.get("horizon_days") == HORIZON_DAYS
        ]
        record = {
            "question": question,
            "run": to_jsonable(run.to_dict()),
            "forecast": to_jsonable(forecasts[-1].result) if forecasts else None,
            "forecast_evidence": forecasts[-1].id if forecasts else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    if not record["forecast_evidence"]:
        return None
    chosen = next(e for e in record["run"]["evidence"] if e["id"] == record["forecast_evidence"])
    args = chosen["arguments"]
    store = store or ProductionStore(processed_dir)
    row = _decline_row(
        case,
        _production(store, case.well),
        "agent",
        start=args.get("start"),
        end=args.get("end"),
        model=args.get("model", "auto"),
    )
    return {**row, "verified": bool(record["run"]["verified"])}


def calibrate(results: pd.DataFrame) -> pd.DataFrame:
    """Add errors and calibrated 80% bands per method.

    A case's band uses only the same method's errors on cases whose forecast period had
    already ended at its cutoff, the errors a forecaster could have known then. Early
    cases with fewer than MIN_PAST_ERRORS such errors get no band.
    """
    results = results.copy()
    results["error"] = (results["p50"] - results["actual"]) / results["actual"]
    results["in_bootstrap_band"] = results["actual"].between(results["p90"], results["p10"])
    horizon = pd.Timedelta(days=HORIZON_DAYS)
    low, high = [], []
    for _, row in results.iterrows():
        known = results[
            (results["method"] == row["method"]) & (results["cutoff"] + horizon <= row["cutoff"])
        ]["error"]
        if len(known) < MIN_PAST_ERRORS:
            low.append(np.nan)
            high.append(np.nan)
            continue
        # error = p50 / actual - 1, so the actual lies in p50 / (1 + error quantiles).
        q_low, q_high = np.quantile(known, BAND)
        low.append(row["p50"] / (1 + q_high))
        high.append(row["p50"] / (1 + q_low))
    results["calibrated_low"] = low
    results["calibrated_high"] = high
    has_band = results["calibrated_low"].notna()
    results["in_calibrated_band"] = (
        results["actual"].between(results["calibrated_low"], results["calibrated_high"])
    ).where(has_band)
    return results


def summarize(results: pd.DataFrame) -> dict:
    summary = {}
    for method, rows in results.groupby("method"):
        abs_error = rows["error"].abs()
        summary[method] = {
            "cases": len(rows),
            "median_abs_error": float(abs_error.median()),
            "mean_abs_error": float(abs_error.mean()),
            "median_bias": float(rows["error"].median()),
            "bootstrap_band_coverage": float(rows["in_bootstrap_band"].mean()),
            "calibrated_band_cases": int(rows["in_calibrated_band"].notna().sum()),
            "calibrated_band_coverage": float(
                rows["in_calibrated_band"].dropna().astype(bool).mean()
            )
            if rows["in_calibrated_band"].notna().any()
            else None,
        }
    return summary


def same_cases(results: pd.DataFrame, method: str) -> pd.DataFrame:
    """Rows of every method restricted to the cases `method` has, for a fair comparison."""
    keys = results.loc[results["method"] == method, ["well", "cutoff"]]
    return results.merge(keys, on=["well", "cutoff"])


async def run(with_agent: bool, agent_limit: int) -> dict:
    settings = get_settings()
    store = ProductionStore(settings.processed_dir)
    all_cases = cases(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # curve_fit covariance warnings on flat windows
        rows = decline_methods(store, all_cases)
    # Cached agent runs always count; --agent adds runs for the cases still missing.
    llm = None
    if with_agent:
        from wellpilot.llm import make_llm

        llm = make_llm(settings)
    step = max(1, len(all_cases) // agent_limit)
    chosen = all_cases[::step][:agent_limit] if with_agent else []
    for case in all_cases:
        cached = _agent_cache(settings.processed_dir, case).exists()
        if not cached and case not in chosen:
            continue
        if not cached:
            print(f"agent: {case.well} from {case.cutoff.date()}")
        row = await agent_row(llm, settings.processed_dir, case, store)
        if row:
            rows.append(row)
        elif not cached:
            print("  no forecast_production evidence in this run")

    results = calibrate(pd.DataFrame(rows))
    summary = {"all_cases": summarize(results)}
    if (results["method"] == "agent").any():
        summary["agent_cases"] = summarize(same_cases(results, "agent"))
    summary["horizon_days"] = HORIZON_DAYS
    summary["cutoffs"] = (
        f"every {STEP_MONTHS} months from {FIRST_CUTOFF_MONTHS} months after first oil"
    )
    results.to_parquet(settings.processed_dir / RESULTS_FILE, index=False)
    (settings.processed_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest 12-month oil forecasts.")
    parser.add_argument("--agent", action="store_true", help="include the LLM agent (tokens)")
    parser.add_argument("--agent-limit", type=int, default=8, help="agent cases to run")
    args = parser.parse_args(argv)
    summary = asyncio.run(run(args.agent, args.agent_limit))
    for scope in ("all_cases", "agent_cases"):
        if scope not in summary:
            continue
        print(f"\n{scope.replace('_', ' ')}:")
        print(
            f"{'method':12s} {'cases':>5s} {'median|err|':>12s} {'mean|err|':>10s} "
            f"{'bias':>7s} {'bootstrap band':>15s} {'calibrated band':>16s}"
        )
        for method, s in summary[scope].items():
            print(
                f"{method:12s} {s['cases']:5d} {s['median_abs_error']:12.1%} "
                f"{s['mean_abs_error']:10.1%} {s['median_bias']:+7.1%} "
                f"{s['bootstrap_band_coverage']:15.0%} "
                f"{(s['calibrated_band_coverage'] or 0):9.0%} of {s['calibrated_band_cases']:>2d}"
            )


if __name__ == "__main__":
    main()
