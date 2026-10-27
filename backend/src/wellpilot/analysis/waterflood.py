"""Waterflood diagnostics: what an oil-rate decline curve alone misses on a waterflooded well.

Volve is a waterflood: injectors push water towards the producers, and once it breaks
through, a well's oil rate falls because more of the liquid it makes is water, not
because the reservoir runs out of pressure. Three classic views show that:

- WOR against cumulative oil. Late in a waterflooded well's life log10(WOR) rises about
  linearly with cumulative oil (Np). Fitted on the recent months and extended to an
  economic water cut (0.80, 0.90, 0.95; WOR 4, 9, 19), it estimates the oil left before
  the well is uneconomic: remaining oil = Np at the limit minus Np now.
- Liquid rate and choke by year. Liquid held while the choke opens and oil falls means
  water is taking the well's capacity (fractional flow); liquid falling too would point
  at pressure support.
- Injection against production for the field (voidage replacement, approximated with
  surface volumes: water injected over oil plus water produced; no formation volume
  factors or gas, so a ratio, not a reservoir balance).

Every number comes from the production data; the fit carries warnings when it is poor.
"""

import numpy as np
import pandas as pd

from wellpilot.data.store import ProductionStore

WATER_CUT_LIMITS = (0.80, 0.90, 0.95)  # 0.80 is the platform's design limit (PUD p.23)
MIN_FIT_MONTHS = 12
FIT_MONTHS = 36  # the most recent months after breakthrough the trend is fitted on
MIN_WOR = 1.0  # water cut 0.5: the straight-line part of the WOR plot starts about here


def _monthly(daily: pd.DataFrame) -> pd.DataFrame:
    prod = daily[daily["flow_kind"] == "production"].copy()
    prod["month"] = prod["date"].dt.to_period("M").dt.to_timestamp()
    on = prod[prod["on_stream_hrs"] >= 12]
    monthly = prod.groupby("month").agg(
        oil_sm3=("oil_sm3", "sum"),
        water_sm3=("water_sm3", "sum"),
        hours=("on_stream_hrs", "sum"),
    )
    monthly["choke_pct"] = on.groupby("month")["choke_pct"].median()
    monthly = monthly[monthly["oil_sm3"] > 0]
    monthly["cum_oil_sm3"] = monthly["oil_sm3"].cumsum()
    monthly["wor"] = monthly["water_sm3"] / monthly["oil_sm3"]
    monthly["water_cut"] = monthly["water_sm3"] / (monthly["oil_sm3"] + monthly["water_sm3"])
    producing_days = monthly["hours"] / 24
    monthly["liquid_rate_sm3_per_day"] = (
        monthly["oil_sm3"] + monthly["water_sm3"]
    ) / producing_days
    return monthly.drop(columns="hours").reset_index()


def _wor_fit(monthly: pd.DataFrame, start=None, end=None) -> tuple[dict | None, list[str]]:
    window = monthly[monthly["wor"] >= MIN_WOR]
    if start is not None:
        window = window[window["month"] >= pd.Timestamp(start)]
    if end is not None:
        window = window[window["month"] <= pd.Timestamp(end)]
    window = window.tail(FIT_MONTHS)
    if len(window) < MIN_FIT_MONTHS:
        return None, [
            f"too few months with WOR >= {MIN_WOR:g} (water cut 0.5) to fit a trend: "
            f"{len(window)}, need {MIN_FIT_MONTHS}"
        ]
    x = window["cum_oil_sm3"].to_numpy()
    y = np.log10(window["wor"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    r2 = 1 - ((y - fitted) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    np_now = float(monthly["cum_oil_sm3"].iloc[-1])
    wor_now = float(monthly["wor"].iloc[-1])
    warnings = []
    if r2 < 0.5:
        warnings.append(f"poor WOR trend: r2 {r2:.2f} < 0.5; the remaining oil is not reliable")
    if slope <= 0:
        warnings.append("WOR is not rising with cumulative oil: no limit can be projected")
    limits = []
    for cut in WATER_CUT_LIMITS:
        wor_limit = cut / (1 - cut)
        if slope > 0:
            np_limit = (np.log10(wor_limit) - intercept) / slope
            remaining = max(0.0, np_limit - np_now)
        else:
            np_limit, remaining = None, None
        limits.append(
            {
                "water_cut": cut,
                "wor": round(wor_limit, 2),
                "cum_oil_at_limit_sm3": np_limit,
                "remaining_oil_sm3": remaining,
                "already_passed": bool(wor_now >= wor_limit),
            }
        )
    return {
        "window_start": str(window["month"].iloc[0].date())[:7],
        "window_end": str(window["month"].iloc[-1].date())[:7],
        "months": len(window),
        "log10_wor_per_million_sm3": slope * 1e6,
        "intercept": intercept,
        "r2": r2,
        "cum_oil_now_sm3": np_now,
        "wor_now": wor_now,
        "to_limits": limits,
    }, warnings


def _field_voidage(store: ProductionStore) -> pd.DataFrame:
    return store.query(
        """
        SELECT year(date) AS year,
               sum(coalesce(water_inj_sm3, 0)) AS injected_sm3,
               sum(coalesce(oil_sm3, 0) + coalesce(water_sm3, 0)) AS produced_liquid_sm3
        FROM production_daily GROUP BY 1 HAVING sum(coalesce(oil_sm3, 0)) > 0 ORDER BY 1
        """
    ).rows.assign(replacement_ratio=lambda t: t["injected_sm3"] / t["produced_liquid_sm3"])


def waterflood(store: ProductionStore, well: str, start=None, end=None) -> dict:
    monthly = _monthly(store.well_daily(well))
    if monthly.empty:
        raise ValueError(f"{well} produced no oil in the data")
    fit, warnings = _wor_fit(monthly, start, end)
    by_year = (
        monthly.assign(year=monthly["month"].dt.year)
        .groupby("year")
        .agg(
            liquid_rate_sm3_per_day=("liquid_rate_sm3_per_day", "median"),
            choke_pct=("choke_pct", "median"),
            water_cut=("water_cut", "median"),
            oil_sm3=("oil_sm3", "sum"),
        )
        .reset_index()
    )
    monthly = monthly.assign(month=monthly["month"].dt.strftime("%Y-%m"))
    return {
        "well": well,
        "monthly": monthly[
            ["month", "cum_oil_sm3", "wor", "water_cut", "liquid_rate_sm3_per_day", "choke_pct"]
        ],
        "wor_fit": fit,
        "by_year": by_year,
        "field_voidage": _field_voidage(store),
        "notes": (
            "WOR = water/oil by month; the trend is log10(WOR) against cumulative oil, fitted "
            f"on up to {FIT_MONTHS} recent months with WOR >= {MIN_WOR:g}. Voidage is "
            "approximated with surface volumes (no formation volume factors, no gas)."
        ),
        "warnings": warnings,
    }
