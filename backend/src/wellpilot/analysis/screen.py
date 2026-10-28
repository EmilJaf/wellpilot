"""Field screen: every well at once, its recent state and what needs attention, ranked.

An engineer looking after a field does not start from one well; they start from the list
and ask which wells need a look. This runs the deterministic detectors over every well,
as of the end of the data (or the as-of date the store is cut at), and ranks the issues:

- alarm: a water cut at or above 0.90;
- caution: a downhole gauge still dead at the end (a monitoring gap: the well flows on, but
  decisions are made without pressure), a water cut at or above 0.80, an oil rate down 30%
  or more over the last 90 days against the 90 before, 30 or more shut-in days in the last
  180, a step down or a shift to a lower regime in oil rate in the last year;
- note: a well that has stopped producing, an injector that stopped injecting.

No model is involved; every number comes from the data and every issue says how it was
found.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wellpilot.analysis.anomalies import gauge_failures, regime_changes, threshold_crossing
from wellpilot.analysis.decline import producing_day_rate
from wellpilot.data.store import ProductionStore

WINDOW = pd.Timedelta(days=90)
STEP = 0.25  # a drop of this much across a change point, 4 weeks each side, is a step
LEVELS = {"alarm": 3, "caution": 1, "note": 0}


@dataclass(frozen=True)
class Issue:
    level: str  # alarm | caution | note
    kind: str
    since: str | None
    text: str


def _median_rate(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    window = daily[(daily["date"] > start) & (daily["date"] <= end)]
    rate = producing_day_rate(window)
    return float(rate.median()) if len(rate) >= 10 else None


def _issues(store: ProductionStore, well: str, end: pd.Timestamp) -> tuple[dict, list[Issue]]:
    daily = store.well_daily(well)
    prod = daily[daily["flow_kind"] == "production"]
    oil_days = prod[prod["oil_sm3"] > 0]
    issues: list[Issue] = []
    state: dict = {}

    injected = daily["water_inj_sm3"].fillna(0)
    if len(oil_days) == 0:  # an injector all along
        recent = daily[daily["date"] > end - WINDOW]
        state["water_inj_sm3_per_day"] = float(recent["water_inj_sm3"].fillna(0).mean())
        last_injection = daily.loc[injected > 0, "date"].max()
        if pd.notna(last_injection) and last_injection < end - pd.Timedelta(days=30):
            issues.append(
                Issue("note", "stopped", str(last_injection.date()), "no injection since then")
            )
        return state, issues

    last_oil = oil_days["date"].max()
    rate_now = _median_rate(prod, end - WINDOW, end)
    rate_before = _median_rate(prod, end - 2 * WINDOW, end - WINDOW)
    recent = prod[(prod["date"] > end - WINDOW) & (prod["on_stream_hrs"] >= 12)]
    state.update(
        {
            "oil_rate_sm3_per_day": rate_now,
            "water_cut": float(recent["water_cut"].median()) if len(recent) else None,
            "uptime": float((prod.loc[prod["date"] > end - WINDOW, "on_stream_hrs"] / 24).mean())
            if len(prod)
            else None,
            "oil_rate_change_90d": rate_now / rate_before - 1 if rate_now and rate_before else None,
        }
    )

    if last_oil < end - pd.Timedelta(days=30):
        issues.append(Issue("note", "stopped", str(last_oil.date()), "no oil since then"))
        return state, issues

    for failure in gauge_failures(daily, store.well_issues(well), columns=("bhp_bar",)):
        if failure.end is not None and failure.end >= end - pd.Timedelta(days=7):
            d = failure.details
            issues.append(
                Issue(
                    "caution",
                    "dead_gauge",
                    str(failure.start.date()),
                    f"downhole pressure gauge reads 0 ({d['zero_days']} days, "
                    f"{d['oil_sm3_during']:,.0f} Sm3 oil produced without pressure data): a "
                    "monitoring gap, the well flows on",
                )
            )
    for threshold, level in ((0.9, "alarm"), (0.8, "caution")):
        crossing = threshold_crossing(daily, "water_cut", threshold)
        if crossing and (state["water_cut"] or 0) >= threshold:
            issues.append(
                Issue(
                    level,
                    "water_cut",
                    str(crossing.start.date()),
                    f"water cut above {threshold:.2f} (now {state['water_cut']:.2f})",
                )
            )
            break
    change = state["oil_rate_change_90d"]
    if change is not None and change <= -0.3:
        issues.append(
            Issue(
                "caution",
                "decline",
                str((end - WINDOW).date()),
                f"oil rate per producing day down {-change:.0%} over the last 90 days "
                f"({rate_before:,.0f} -> {rate_now:,.0f} Sm3/d)",
            )
        )
    shut = prod[(prod["date"] > end - 2 * WINDOW) & (prod["on_stream_hrs"].fillna(0) == 0)]
    if len(shut) >= 30:
        issues.append(
            Issue("caution", "shut_in", None, f"{len(shut)} days shut in over the last 180")
        )
    for change in regime_changes(daily, "oil_rate"):
        d = change.details
        relative, across = d.get("relative_change"), d.get("change_across")
        choke = (
            f"; the choke was {'opened' if d['choke_after_pct'] > d['choke_before_pct'] else 'cut'}"
            f" at the same time ({d['choke_before_pct']:.0f} -> {d['choke_after_pct']:.0f}%)"
            if d.get("choke_moved")
            else ""
        )
        if change.start <= end - pd.Timedelta(days=365) or relative is None:
            continue
        if across is not None and across <= -STEP:
            # Abrupt: an operation or a restart, worth a look at what was done that week.
            issues.append(
                Issue(
                    "caution",
                    "step_down",
                    str(change.start.date()),
                    f"oil rate dropped {-across:.0%} across the change (4 weeks each side, "
                    f"{d['across_before']:,.0f} -> {d['across_after']:,.0f} Sm3/d){choke}",
                )
            )
        elif relative < -0.3:
            # Gradual: the rate moved to a lower regime (water taking over, depletion).
            issues.append(
                Issue(
                    "caution",
                    "shift_down",
                    str(change.start.date()),
                    f"oil rate {-relative:.0%} lower in the period after than before "
                    f"(weekly mean {d['before_mean']:,.0f} -> {d['after_mean']:,.0f} Sm3/d), "
                    f"a gradual change: {-(across or 0):.0%} across it{choke}",
                )
            )
    return state, issues


def screen(store: ProductionStore) -> dict:
    # The last day the field produced oil: after shutdown every well would read "stopped".
    last_oil = "SELECT max(date) AS d FROM production_daily WHERE oil_sm3 > 0"
    end = pd.Timestamp(store.query(last_oil).rows.loc[0, "d"])
    wells = store.query(
        "SELECT well, sum(oil_sm3) AS cum_oil_sm3, sum(water_inj_sm3) AS cum_injected_sm3, "
        "min(date) FILTER (WHERE oil_sm3 > 0) AS first_oil FROM production_daily "
        "GROUP BY well ORDER BY well"
    ).rows
    field_oil = float(wells["cum_oil_sm3"].sum())
    rows = []
    for w in wells.itertuples():
        state, issues = _issues(store, w.well, end)
        cum_oil = float(w.cum_oil_sm3) if pd.notna(w.cum_oil_sm3) else 0.0
        injected = float(w.cum_injected_sm3) if pd.notna(w.cum_injected_sm3) else 0.0
        rows.append(
            {
                "well": w.well,
                "role": "producer" if cum_oil > 0 else "injector",
                "first_oil": str(pd.Timestamp(w.first_oil).date())
                if pd.notna(w.first_oil)
                else None,
                "cum_oil_sm3": cum_oil,
                "share_of_field_oil": cum_oil / field_oil if field_oil else 0.0,
                "cum_water_injected_sm3": injected,
                **state,
                # The worst issue's level in words: a number here ("attention 3") ended up
                # in answers as if it meant something to a reader.
                "level": max((i.level for i in issues), key=LEVELS.get, default=None),
                "issues": [i.__dict__ for i in sorted(issues, key=lambda i: -LEVELS[i.level])],
            }
        )
    rows.sort(key=lambda r: (-sum(LEVELS[i["level"]] for i in r["issues"]), -r["cum_oil_sm3"]))
    for r in rows:  # NaN from empty windows reads as missing
        for k, v in list(r.items()):
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return {
        "as_of": str(end.date()),
        "window_days": WINDOW.days,
        "field_oil_sm3": field_oil,
        "wells": rows,
        "levels": "alarm: act now; caution: watch; note: for the record",
    }
