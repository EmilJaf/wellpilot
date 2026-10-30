"""Event and anomaly detection on one well's daily production.

Every detector returns Events with the numbers behind them and a one-line, deterministic
summary, so the agent quotes measured facts instead of paraphrasing a chart.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.optimize import curve_fit

from wellpilot.analysis.decline import producing_day_rate

EventKind = Literal["regime_change", "threshold_crossing", "gauge_failure", "shut_in"]
SIGNALS = (
    "oil_rate",
    "water_cut",
    "gor_sm3_sm3",
    "bhp_bar",
    "whp_bar",
    "choke_pct",
    "dp_choke_bar",
)
MIN_FLOWING_HOURS = 12.0


@dataclass(frozen=True)
class Event:
    kind: EventKind
    well: str
    signal: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    details: dict
    summary: str


def _production(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[daily["flow_kind"] == "production"].sort_values("date")


def _well(daily: pd.DataFrame) -> str:
    return str(daily["well"].iloc[0]) if len(daily) else ""


def weekly_signal(daily: pd.DataFrame, signal: str) -> pd.Series:
    """Weekly mean of a signal over days on stream at least 12 h (oil_rate: producing-day)."""
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; use one of {SIGNALS}")
    prod = _production(daily)
    if signal == "oil_rate":
        series = producing_day_rate(prod, "oil_sm3", MIN_FLOWING_HOURS)
    else:
        flowing = prod[prod["on_stream_hrs"] >= MIN_FLOWING_HOURS]
        series = flowing.set_index("date")[signal]
    return series.resample("W").mean().dropna()


SHARES = {"water_cut"}  # signals that are fractions of a whole
# An oil-rate change point must lift the log rate this far off the smooth Arps curve of the
# periods around it (about 10%): on a synthetic field, smooth declines gave at most 0.04,
# a choke cut 0.48; every Volve change point gives 0.11 or more.
SMOOTH_SHIFT = 0.1
SMOOTH_WEEKS = 8
ACROSS_WEEKS = 4  # the change across a change point: medians of 4 weeks each side
# A choke moved this many points (or by half, at small openings) across an oil change
# point, the same way as the rate, is the operator's hand in it: F-12 on 2010-05-02 went
# 83 -> 58% while its downhole pressure rose (choked back); depletion or water taking over
# would leave the choke where it was.
CHOKE_POINTS = 10.0
CHOKE_RATIO = 1.5


def _choke_across(daily: pd.DataFrame, at: pd.Timestamp) -> tuple[float, float] | None:
    """Median weekly choke opening (%) in the ACROSS_WEEKS weeks either side of a date."""
    if "choke_pct" not in daily or daily["choke_pct"].notna().sum() == 0:
        return None
    choke = weekly_signal(daily, "choke_pct")
    span = pd.Timedelta(weeks=ACROSS_WEEKS)
    before = choke[(choke.index < at) & (choke.index >= at - span)]
    after = choke[(choke.index >= at) & (choke.index < at + span)]
    if before.empty or after.empty:
        return None
    return float(before.median()), float(after.median())


def _hyperbolic_log(t: np.ndarray, log_qi: float, di: float, b: float) -> np.ndarray:
    return log_qi - np.log1p(b * di * t) / b


def _beyond_smooth(x: np.ndarray, lo: int, at: int, hi: int) -> float:
    """How far the log rate jumps off one smooth Arps (hyperbolic) curve fitted on the two
    periods around a change point: the mean residual of the 8 weeks after it minus the 8
    weeks before. Near 0, the change point only cuts a smooth decline in two."""
    span = x[lo:hi]
    t = np.arange(len(span), dtype=float)
    try:
        params, _ = curve_fit(
            _hyperbolic_log,
            t,
            span,
            p0=[span[0], 0.01, 0.5],
            bounds=([-np.inf, 0.0, 1e-3], [np.inf, 1.0, 1.0]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return float("inf")  # no smooth curve fits: keep the change
    residual = span - _hyperbolic_log(t, *params)
    k = at - lo
    return float(
        residual[k : k + SMOOTH_WEEKS].mean() - residual[max(0, k - SMOOTH_WEEKS) : k].mean()
    )


def regime_changes(
    daily: pd.DataFrame, signal: str, penalty: float = 15.0, min_weeks: int = 8
) -> list[Event]:
    """Change points (PELT, L2 cost) on the standardized weekly signal.

    Oil rate is segmented on its log, so a 30% drop counts the same at any rate level. A
    mean-shift segmentation also cuts a smooth decline into steps, so an oil-rate change
    point is kept only when the rate leaves the smooth Arps curve of the periods around it
    (by at least SMOOTH_SHIFT in log rate). Each change gives the mean of the periods on
    either side and the change across it, 4 weeks each side: a large change across is a
    step (an operation, a restart); a small one under a large change of periods is gradual
    (water taking over, depletion). Lower penalty finds more, smaller changes.
    """
    weekly = weekly_signal(daily, signal)
    values = weekly.to_numpy()
    if len(values) < 2 * min_weeks:
        return []
    x = np.log(values) if signal == "oil_rate" else values
    std = x.std()
    if std == 0:
        return []
    z = (x - x.mean()) / std
    breaks = rpt.Pelt(model="l2", min_size=min_weeks, jump=1).fit(z).predict(pen=penalty)[:-1]
    if signal == "oil_rate":
        cuts = [0, *breaks, len(values)]
        breaks = [
            b
            for i, b in enumerate(breaks)
            if abs(_beyond_smooth(x, cuts[i], b, cuts[i + 2])) >= SMOOTH_SHIFT
        ]

    events, bounds = [], [0, *breaks, len(values)]
    for i, b in enumerate(breaks):
        before = float(values[bounds[i] : b].mean())
        after = float(values[b : bounds[i + 2]].mean())
        start = weekly.index[b]
        across_before = float(np.median(values[max(0, b - ACROSS_WEEKS) : b]))
        across_after = float(np.median(values[b : b + ACROSS_WEEKS]))
        # A share (water cut) changes by points: 0.034 -> 0.57 is +0.54, not "+1581%".
        if signal in SHARES:
            step = across_after - across_before
            change = {
                "change": after - before,
                "across_before": across_before,
                "across_after": across_after,
                "change_across": step,
            }
            shift = f"{after - before:+.2f}"
            across = (
                f"; across the change itself, {ACROSS_WEEKS} weeks each side, "
                f"{across_before:.2f} -> {across_after:.2f} ({step:+.2f})"
            )
        else:
            relative = (after - before) / abs(before) if before else float("nan")
            step = across_after / across_before - 1 if across_before else float("nan")
            change = {
                "relative_change": relative,
                "across_before": across_before,
                "across_after": across_after,
                "change_across": step,
            }
            shift = f"{relative:+.0%}"
            across = (
                f"; across the change itself, {ACROSS_WEEKS} weeks each side, "
                f"{_fmt(across_before)} -> {_fmt(across_after)} ({step:+.0%})"
            )
            choke = _choke_across(daily, start) if signal == "oil_rate" else None
            if choke is not None:
                moved = choke[1] - choke[0]
                ratio = max(choke) / min(choke) if min(choke) > 0 else 1.0
                operator = (abs(moved) >= CHOKE_POINTS or ratio >= CHOKE_RATIO) and (moved > 0) == (
                    step > 0
                )
                change |= {
                    "choke_before_pct": choke[0],
                    "choke_after_pct": choke[1],
                    "choke_moved": operator,
                }
                across += (
                    f"; the choke was {'opened' if moved > 0 else 'cut'} at the same time "
                    f"({choke[0]:.0f} -> {choke[1]:.0f}%), so part of the change is the "
                    "operator's, not the reservoir's"
                    if operator
                    else f"; the choke stayed about the same ({choke[0]:.0f} -> {choke[1]:.0f}%)"
                )
        events.append(
            Event(
                kind="regime_change",
                well=_well(daily),
                signal=signal,
                start=start,
                end=None,
                details={
                    "before_mean": before,
                    "after_mean": after,
                    **change,
                    "before_from": weekly.index[bounds[i]],
                    # Weeks are labelled by their Sunday: the last one may end past the data
                    # (past an as-of date), so the period ends on the data's last day.
                    "after_to": min(weekly.index[bounds[i + 2] - 1], daily["date"].max()),
                },
                summary=(
                    f"{signal} shifts around {start.date()}: weekly mean {_fmt(before)} -> "
                    f"{_fmt(after)} ({shift}) from the period before to the period after"
                    f"{across}."
                ),
            )
        )
    return events


def threshold_crossing(
    daily: pd.DataFrame,
    signal: str,
    threshold: float,
    window_days: int = 30,
    direction: Literal["above", "below"] = "above",
) -> Event | None:
    """First day the rolling median over flowing days crosses the threshold, i.e. a
    sustained crossing rather than a single noisy day."""
    prod = _production(daily)
    flowing = prod[prod["on_stream_hrs"] >= MIN_FLOWING_HOURS].set_index("date")
    if signal == "oil_rate":
        series = producing_day_rate(prod, "oil_sm3", MIN_FLOWING_HOURS)
    else:
        series = flowing[signal].dropna()
    median = series.rolling(f"{window_days}D", min_periods=window_days // 2).median()
    hits = median[median >= threshold] if direction == "above" else median[median <= threshold]
    if hits.empty:
        return None

    day = hits.index[0]
    first_oil = prod.loc[prod["oil_sm3"] > 0, "date"].min()
    months = (day - first_oil).days / 30.44 if pd.notna(first_oil) else float("nan")
    return Event(
        kind="threshold_crossing",
        well=_well(daily),
        signal=signal,
        start=day,
        end=None,
        details={
            "threshold": threshold,
            "direction": direction,
            "window_days": window_days,
            "median_at_crossing": float(hits.iloc[0]),
            "first_oil": first_oil,
            "months_after_first_oil": months,
        },
        summary=(
            f"{window_days}-day median {signal} first goes {direction} {threshold:g} on "
            f"{day.date()}, {months:.0f} months after first oil."
        ),
    )


def _fmt(x: float) -> str:
    """A number as an engineer writes it: 4,050 and 1,146, 0.513, never 4.05e+03."""
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _runs(dates: pd.Series, max_gap_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group sorted dates into runs whose consecutive gaps are at most max_gap_days."""
    if dates.empty:
        return []
    dates = dates.sort_values().reset_index(drop=True)
    new_run = dates.diff().dt.days.fillna(max_gap_days + 1) > max_gap_days
    groups = new_run.cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in dates.groupby(groups)]


def _dead_runs(
    zero_dates: pd.Series, good_dates: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group zero-reading dates into runs that no valid reading interrupts.

    Days without any reading (e.g. shut-ins) do not end a run; a real reading does.
    """
    zero = np.sort(zero_dates.to_numpy())
    if len(zero) == 0:
        return []
    good = np.sort(good_dates.to_numpy())
    # Number of valid readings strictly between each pair of consecutive zero days.
    between = np.searchsorted(good, zero[1:], side="left") - np.searchsorted(
        good, zero[:-1], side="right"
    )
    starts = np.concatenate([[0], np.flatnonzero(between > 0) + 1])
    ends = np.concatenate([starts[1:] - 1, [len(zero) - 1]])
    return [
        (pd.Timestamp(zero[s]), pd.Timestamp(zero[e])) for s, e in zip(starts, ends, strict=True)
    ]


def gauge_failures(
    daily: pd.DataFrame,
    issues: pd.DataFrame,
    columns: tuple[str, ...] = ("bhp_bar", "whp_bar"),
    min_days: int = 7,
) -> list[Event]:
    """Runs of zero gauge readings from the ingest issue log, checked against production.

    A gauge stuck at 0 while oil keeps flowing is a dead sensor, not a pressure collapse.
    """
    well = _well(daily)
    prod = _production(daily).set_index("date")
    zero = issues[
        (issues["well"] == well)
        & issues["field"].isin(columns)
        & issues["issue"].isin(["zero_gauge_reading", "zero_whp_while_flowing"])
    ]
    events = []
    for column, rows in zero.groupby("field"):
        good_dates = prod.index[prod[column].notna()].to_series()
        for start, end in _dead_runs(rows["date"], good_dates):
            days = rows[(rows["date"] >= start) & (rows["date"] <= end)]
            if len(days) < min_days:
                continue
            span = prod.loc[start:end]
            flowing_days = int((span["on_stream_hrs"] >= MIN_FLOWING_HOURS).sum())
            oil = float(span["oil_sm3"].sum())
            good = prod.loc[: start - pd.Timedelta(days=1), column].dropna()
            last_good = (good.index[-1], float(good.iloc[-1])) if len(good) else (None, None)
            if flowing_days:
                verdict = (
                    f"while the well kept flowing on {flowing_days} days ({oil:,.0f} Sm3 oil): "
                    "a sensor failure, not a pressure collapse"
                )
            else:
                verdict = "while the well was shut in; the gauge may just be switched off"
            last = (
                f" Last good reading: {last_good[1]:.1f} on {last_good[0].date()}."
                if last_good[0] is not None
                else ""
            )
            events.append(
                Event(
                    kind="gauge_failure",
                    well=well,
                    signal=str(column),
                    start=start,
                    end=end,
                    details={
                        "zero_days": len(days),
                        "flowing_days": flowing_days,
                        "oil_sm3_during": oil,
                        "last_good_date": last_good[0],
                        "last_good_value": last_good[1],
                    },
                    summary=(
                        f"{column} reads 0 on {len(days)} days from {start.date()} to "
                        f"{end.date()} {verdict}.{last}"
                    ),
                )
            )
    return events


def shut_ins(daily: pd.DataFrame, min_days: int = 3, compare_days: int = 30) -> list[Event]:
    """Runs of zero on-stream hours after first oil, with the rate before and after."""
    prod = _production(daily)
    first_oil = prod.loc[prod["oil_sm3"] > 0, "date"].min()
    if pd.isna(first_oil):
        return []
    prod = prod[prod["date"] >= first_oil]
    rate = producing_day_rate(prod, "oil_sm3", MIN_FLOWING_HOURS)
    off = prod.loc[prod["on_stream_hrs"].fillna(0) == 0, "date"]
    window = pd.Timedelta(days=compare_days)

    events = []
    # Days the sheet left out are inserted at ingest, so any day with hours ends a run.
    for start, end in _runs(off, max_gap_days=1):
        n_days = (end - start).days + 1
        if n_days < min_days:
            continue
        before = rate[start - window : start - pd.Timedelta(days=1)].median()
        after = rate[end + pd.Timedelta(days=1) : end + window].median()
        change = (after - before) / before if before and pd.notna(after) else float("nan")
        restart = (
            f" Oil rate {before:,.0f} -> {after:,.0f} Sm3/d ({change:+.0%}) after restart."
            if pd.notna(change)
            else ""
        )
        events.append(
            Event(
                kind="shut_in",
                well=_well(daily),
                signal="on_stream_hrs",
                start=start,
                end=end,
                details={
                    "days": n_days,
                    "rate_before": float(before) if pd.notna(before) else None,
                    "rate_after": float(after) if pd.notna(after) else None,
                    "relative_change": change,
                },
                summary=f"Shut in for {n_days} days from {start.date()}.{restart}",
            )
        )
    return events


def mark_field_wide(
    events: list[Event],
    tolerance_days: int = 2,
    flowing: Callable[[Event], set[str]] | None = None,
) -> list[Event]:
    """Mark shut-ins that overlap other wells' shut-ins. One is field-wide only when every
    producer that flowed in the 30 days before it (`flowing`) was shut in too: F-12's
    December 2014 shut-in overlapped F-1 C's and F-15 D's while F-11 and F-14 ran all month
    (audit, Oct 31; it used to count any one overlap as the platform)."""
    shut = [e for e in events if e.kind == "shut_in"]
    tol = pd.Timedelta(days=tolerance_days)
    marked = []
    for e in events:
        if e.kind != "shut_in":
            marked.append(e)
            continue
        others = sorted(
            {
                o.well
                for o in shut
                if o.well != e.well and o.start <= e.end + tol and e.start <= o.end + tol
            }
        )
        running = sorted((flowing(e) if flowing else set()) - {e.well} - set(others))
        details = {
            **e.details,
            "field_wide": bool(others) and not running,
            "also_shut_in": others,
            "others_flowing": running,
        }
        summary = e.summary
        if others:
            summary += f" Same time as {', '.join(others)}"
            summary += f"; {', '.join(running)} kept flowing." if running else " (field-wide)."
        marked.append(replace(e, details=details, summary=summary))
    return marked
