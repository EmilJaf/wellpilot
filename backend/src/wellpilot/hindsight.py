"""Hindsight: a run made with data up to a past date, set against what happened next.

The agent's run never sees the data after its as-of date (the store is cut there). Once it
has answered, this reads the full data and shows, for every forecast the run made, the
oil the well actually produced over the forecast's days, and what the classic decline fit
(the whole history up to the same date) would have said. For every well, it also shows
the year after the date beside the year before: oil, water cut, and whether a dead
downhole gauge came back. And the model's reasoning, read for dates and values from the
hidden period (reasoning_check).

Only the API calls this, after a run ends; nothing here reaches the agent.
"""

import pandas as pd

from wellpilot.agent.tools import fit_warnings
from wellpilot.analysis.decline import fit_decline, forecast, nominal_coverage
from wellpilot.backtest import RECENT_MONTHS, _curve, weekly_rate
from wellpilot.data.store import ProductionStore
from wellpilot.reasoning_check import reasoning_check

YEAR = pd.Timedelta(days=365)
WATER_CUT_DAYS = 30  # water cut at a date: the median over the flowing days of the month before


def _production(store: ProductionStore, well: str, start=None, end=None) -> pd.DataFrame:
    """Production days only, as the agent's fit and forecast tools read them."""
    daily = store.well_daily(well, start, end)
    return daily[daily["flow_kind"] == "production"]


def _method(name: str, fit, p50: float, actual: float, future: pd.DatetimeIndex) -> dict:
    return {
        "method": name,
        "model": fit.model,
        "window_start": fit.t0.date().isoformat(),
        "window_end": fit.end.date().isoformat(),
        "fitted": _curve(fit, pd.date_range(fit.t0, fit.end, freq="7D")),
        "forecast": _curve(fit, future),
        "p50_sm3": round(p50),
        "error": round(p50 / actual - 1, 4) if actual > 0 else None,
    }


def _forecast_hindsight(
    full: ProductionStore, past: ProductionStore, cut: pd.Timestamp, evidence: dict, run: dict
) -> dict:
    args, result = evidence["arguments"], evidence["result"]
    well = args["well"]
    first = pd.Timestamp(result["forecast_from"])
    last = pd.Timestamp(result["forecast_to"])
    daily = _production(full, well)
    actual = float(daily.loc[daily["date"].between(first, last), "oil_sm3"].sum())
    data_end = daily["date"].max()
    future = pd.date_range(first, last, freq="7D")

    # The agent's fit, refitted on the same days with the same arguments (deterministic) to
    # draw it; its volumes are the ones the agent reported.
    window = _production(past, well, args.get("start"), args.get("end"))
    agent_fit = fit_decline(window, model=args.get("model", "auto")).best
    reported = result["cumulative_oil_sm3"]
    classic_fit = fit_decline(_production(past, well)).best
    classic = forecast(classic_fit, result["horizon_days"], n_boot=100, start=first)
    # Fits on fixed recent windows, made the same day: the fair baselines. Which one would
    # have been right is known only afterwards; the agent chose its own window.
    fixed = []
    for months in RECENT_MONTHS:
        recent = _production(past, well, cut - pd.DateOffset(months=months), cut)
        try:
            fit = fit_decline(recent).best
        except (ValueError, RuntimeError):
            continue
        p50 = float(forecast(fit, result["horizon_days"], n_boot=50, start=first).cumulative["p50"])
        fixed.append(
            {
                "months": months,
                "p50_sm3": round(p50),
                "error": round(p50 / actual - 1, 4) if actual > 0 else None,
            }
        )

    band = result.get("calibrated_band_sm3")
    window_start = str(args.get("start") or "")[:10]
    # The volume is a rate per producing day times the time on stream: which one was off?
    # F-12 in 2011: the rate 12% low and the uptime 20% high made +6%.
    horizon = daily[daily["date"].between(first, last)]
    hours = horizon["on_stream_hrs"].fillna(0).clip(upper=24)
    producing_days = float(hours.sum() / 24)
    days = (last - first).days + 1
    rate_forecast = float(reported["p50"]) / (agent_fit.uptime * days) if agent_fit.uptime else None
    rate_actual = actual / producing_days if producing_days else None
    uptime_actual = producing_days / days if days else None
    decomposition = (
        {
            "rate_forecast": round(rate_forecast),
            "rate_actual": round(rate_actual),
            "rate_error": round(rate_forecast / rate_actual - 1, 4),
            "uptime_forecast": round(agent_fit.uptime, 4),
            "uptime_actual": round(uptime_actual, 4),
            "uptime_error": round(agent_fit.uptime / uptime_actual - 1, 4),
        }
        if rate_forecast and rate_actual and uptime_actual
        else None
    )
    return {
        "evidence_id": evidence["id"],
        "well": well,
        "cutoff": cut.date().isoformat(),
        "forecast_from": first.date().isoformat(),
        "horizon_end": last.date().isoformat(),
        "data_ends": data_end.date().isoformat() if data_end < last else None,
        "actual_sm3": round(actual),
        "rate": weekly_rate(daily, cut, last),
        "methods": [
            _method("agent", agent_fit, float(reported["p50"]), actual, future),
            _method("classic", classic_fit, float(classic.cumulative["p50"]), actual, future),
        ],
        "classic_warned": bool(fit_warnings(classic_fit)),
        "fixed_windows": fixed,
        "decomposition": decomposition,
        # Why the agent's window starts where it does, when a change it found dates it.
        "window_from_change": _change_at(run, well, window_start) if window_start else None,
        "agent_range": {"p90": reported["p90"], "p10": reported["p10"]},
        "p50_by_model": result.get("p50_by_model_sm3"),  # what each Arps model would say
        "calibrated_band": (
            {
                "p90": band["p90"],
                "p10": band["p10"],
                "holds_actual": bool(band["p90"] <= actual <= band["p10"]),
                "past_forecasts": band.get("past_forecasts"),
                # Recorded runs predate the field: computed from their count.
                "coverage": band.get("coverage")
                or (nominal_coverage(n) if (n := band.get("past_forecasts")) else None),
            }
            if band
            else None
        ),
        "agent": {
            "question": run["question"],
            "verified": bool(run.get("verified")),
            "summary": (run.get("result") or {}).get("summary"),
            "inferences": [
                f["claim"]
                for f in (run.get("result") or {}).get("findings", [])
                if f.get("confidence") == "inference"
            ],
        },
    }


CHANGE_KINDS = {"regime_change", "shift_down", "step_down"}


def _change_at(run: dict, well: str, day: str) -> dict | None:
    """What the run's own evidence dates to `day`, where the agent's window starts: a change
    in the well's oil rate (a detect_events regime change, a screen's shift or step), or
    work on the well ending that day or the day before (well_operations). None if nothing."""
    after = (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat()
    for item in (run.get("evidence") or {}).values():
        if item.get("tool") == "well_operations" and item.get("arguments", {}).get("well") == well:
            for job in (item.get("result") or {}).get("jobs") or []:
                if str(job.get("end", ""))[:10] in (day, after):
                    return {
                        "date": day,
                        "evidence_id": item["id"],
                        "kind": "work",
                        "text": f"{job.get('kind', 'work')} on the well, "
                        f"{job['start']} to {job['end']}",
                    }
            continue
        if item.get("tool") == "detect_events" and item.get("arguments", {}).get("well") == well:
            records = item.get("result") or []
        elif item.get("tool") == "field_screen":
            rows = (item.get("result") or {}).get("wells") or []
            records = [i for w in rows if w.get("well") == well for i in w.get("issues") or []]
        else:
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("kind") not in CHANGE_KINDS:
                continue
            if record.get("kind") == "regime_change" and record.get("signal") != "oil_rate":
                continue
            if str(record.get("start") or record.get("since") or "")[:10] == day:
                return {
                    "date": day,
                    "evidence_id": item["id"],
                    "kind": "change",
                    "text": record.get("summary") or record.get("text"),
                }
    return None


def _water_cut_at(daily: pd.DataFrame, at: pd.Timestamp) -> float | None:
    month = daily[
        daily["date"].between(at - pd.Timedelta(days=WATER_CUT_DAYS), at)
        & (daily["on_stream_hrs"] > 0)
    ]
    cut = month["water_cut"].dropna()
    return round(float(cut.median()), 3) if len(cut) else None


def _next_year(full: ProductionStore, cut: pd.Timestamp) -> list[dict]:
    """Each well that had produced oil by the cut: the year before it and the year after."""
    rows = []
    for well in full.query("SELECT DISTINCT well FROM production_daily ORDER BY 1").rows["well"]:
        daily = _production(full, well)
        before = daily[daily["date"].between(cut - YEAR + pd.Timedelta(days=1), cut)]
        if before["oil_sm3"].sum() <= 0:
            continue
        after = daily[daily["date"].between(cut + pd.Timedelta(days=1), cut + YEAR)]
        # A gauge dead at the cut (no valid reading in the last 30 flowing days, but
        # readings before): did a valid reading come back in the year after?
        flowing = daily[(daily["date"] <= cut) & (daily["on_stream_hrs"] > 0)]
        had_gauge = flowing["bhp_bar"].notna().any()
        dead = had_gauge and flowing.tail(30)["bhp_bar"].isna().all()
        back = after.loc[after["bhp_bar"].notna(), "date"].min() if dead else None
        rows.append(
            {
                "well": well,
                "oil_year_before_sm3": round(float(before["oil_sm3"].sum())),
                "oil_year_after_sm3": round(float(after["oil_sm3"].sum())),
                "water_cut_at_cut": _water_cut_at(daily, cut),
                "water_cut_year_later": _water_cut_at(daily, cut + YEAR),
                "gauge_dead_at_cut": bool(dead),
                "gauge_back_on": back.date().isoformat() if dead and pd.notna(back) else None,
            }
        )
    return rows


def hindsight(run: dict, full: ProductionStore, past: ProductionStore) -> dict:
    """`run` is a saved run (question, as_of, result, evidence by id); `past` is the store
    the run saw (cut at its as-of date), `full` the whole data."""
    if not run.get("as_of"):
        raise ValueError("this run saw all the data: there is no later data to compare with")
    cut = pd.Timestamp(run["as_of"])
    evidence = sorted(run["evidence"].values(), key=lambda e: int(e["id"][1:]))
    cited = _cited(run.get("result") or {})
    forecasts = [
        {**_forecast_hindsight(full, past, cut, e, run), "cited": e["id"] in cited}
        for e in evidence
        if e["tool"] == "forecast_production" and isinstance(e.get("result"), dict)
    ]
    # The forecasts the answer gives first; the ones the agent tried on the way after them.
    forecasts.sort(key=lambda f: not f["cited"])
    return {
        "as_of": cut.date().isoformat(),
        "forecasts": forecasts,
        "wells": _next_year(full, cut),
        "reasoning": reasoning_check(run, full),
    }


def _cited(result: dict) -> set[str]:
    """Evidence ids the answer rests on: its findings' sources and every declared number's."""
    ids = {n["evidence_id"] for n in result.get("summary_numbers") or []}
    for finding in result.get("findings") or []:
        ids |= set(finding.get("evidence_ids") or [])
        ids |= {n["evidence_id"] for n in finding.get("numbers") or []}
    return ids
