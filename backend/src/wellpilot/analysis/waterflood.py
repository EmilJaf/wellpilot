"""Waterflood diagnostics: what an oil-rate decline curve alone misses on a waterflooded well.

Volve is a waterflood: injectors push water towards the producers, and once it breaks
through, a well's oil rate falls because more of the liquid it makes is water, not
because the reservoir runs out of pressure. Three classic views show that:

- WOR against cumulative oil. Late in a waterflooded well's life log10(WOR) rises about
  linearly with cumulative oil (Np). Fitted on the recent months and extended to an
  reference water cut (0.80, 0.90, 0.95; WOR 4, 9, 19), it estimates the oil left before
  each: remaining oil = Np at the limit minus Np now. Where a well stops paying depends on
  costs and prices the data does not hold, so these are reference points, not an economic
  limit.
- Liquid rate and choke by year. Liquid held while the choke opens and oil falls means
  water is taking the well's capacity (fractional flow); liquid falling too would point
  at pressure support.
- Injection against production for the field: voidage replacement at reservoir
  conditions, water injected x Bw over oil produced x Bo plus water produced x Bw. Bo is
  the plan's most likely 1.385 m3/Sm3 (PUD p.20, Table 3.7; 1.27-1.5), Bw is taken as 1.
  The reservoir stays above the bubble point (PUD p.22), so no free gas adds voidage. A
  ratio near 1 means injection kept up with what was taken out.

Every number comes from the production data. A poor WOR trend is not returned at all, only
the reason ("no_trend"): the tables beside it are plain data, and a warning on the result
would make every one of them read as uncertain.
"""

import numpy as np
import pandas as pd

from wellpilot.data.store import ProductionStore

WATER_CUT_LIMITS = (0.80, 0.90, 0.95)  # 0.80 is the platform's design limit (PUD p.23)
BO = 1.385  # oil formation volume factor, m3/Sm3: PUD p.20, Table 3.7, most likely value
BO_RANGE = (1.27, 1.5)  # the table's low and high cases
BW = 1.0  # water formation volume factor: not in the plan; 1.0-1.03 is usual
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


def _wor_fit(monthly: pd.DataFrame, start=None, end=None) -> tuple[dict | None, str | None]:
    """The trend extended to each water-cut limit, or None and why there is none."""
    window = monthly[monthly["wor"] >= MIN_WOR]
    if start is not None:
        window = window[window["month"] >= pd.Timestamp(start)]
    if end is not None:
        window = window[window["month"] <= pd.Timestamp(end)]
    window = window.tail(FIT_MONTHS)
    if len(window) < MIN_FIT_MONTHS:
        return None, (
            f"too few months with WOR >= {MIN_WOR:g} (water cut 0.5) to fit a trend: "
            f"{len(window)}, need {MIN_FIT_MONTHS}"
        )
    x = window["cum_oil_sm3"].to_numpy()
    y = np.log10(window["wor"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    r2 = 1 - ((y - fitted) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    span = (
        f"{str(window['month'].iloc[0].date())[:7]} to {str(window['month'].iloc[-1].date())[:7]}"
    )
    if slope <= 0:
        return None, f"WOR is not rising with cumulative oil on {span}: no limit can be projected"
    if r2 < 0.5:
        return None, f"no reliable WOR trend on {span}: r2 {r2:.2f} < 0.5"
    # "Now" is the end of the window asked for, not of all the data: with end=2012-12-31 on
    # F-12 it used 2016's cumulative oil and said no oil was left to a 0.90 water cut.
    upto = monthly if end is None else monthly[monthly["month"] <= pd.Timestamp(end)]
    upto = upto if len(upto) else monthly
    np_now = float(upto["cum_oil_sm3"].iloc[-1])
    wor_now = float(upto["wor"].iloc[-1])
    limits = []
    for cut in WATER_CUT_LIMITS:
        wor_limit = cut / (1 - cut)
        np_limit = (np.log10(wor_limit) - intercept) / slope
        limits.append(
            {
                "water_cut": cut,
                "wor": round(wor_limit, 2),
                "cum_oil_at_limit_sm3": np_limit,
                "remaining_oil_sm3": max(0.0, np_limit - np_now),
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
    }, None


def _field_voidage(store: ProductionStore) -> pd.DataFrame:
    t = store.query(
        """
        SELECT year(date) AS year,
               sum(coalesce(water_inj_sm3, 0)) AS injected_sm3,
               sum(coalesce(oil_sm3, 0)) AS oil_sm3,
               sum(coalesce(water_sm3, 0)) AS water_sm3
        FROM production_daily GROUP BY 1 HAVING sum(coalesce(oil_sm3, 0)) > 0 ORDER BY 1
        """
    ).rows

    def ratio(bo: float) -> pd.Series:
        return t["injected_sm3"] * BW / (t["oil_sm3"] * bo + t["water_sm3"] * BW)

    return t.assign(
        replacement_ratio=ratio(BO),
        replacement_ratio_low=ratio(BO_RANGE[1]),  # with the plan's high Bo
        replacement_ratio_high=ratio(BO_RANGE[0]),
        surface_ratio=t["injected_sm3"] / (t["oil_sm3"] + t["water_sm3"]),
    )


def waterflood(store: ProductionStore, well: str, start=None, end=None) -> dict:
    monthly = _monthly(store.well_daily(well))
    if monthly.empty:
        raise ValueError(f"{well} produced no oil in the data")
    fit, no_trend = _wor_fit(monthly, start, end)
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
        "no_trend": no_trend,
        "by_year": by_year,
        "field_voidage": _field_voidage(store),
        "notes": (
            "WOR = water/oil by month; the trend is log10(WOR) against cumulative oil, fitted "
            f"on up to {FIT_MONTHS} recent months with WOR >= {MIN_WOR:g}. The water cuts in "
            "to_limits are reference points, not an economic limit: the data holds no costs or "
            "prices, so remaining_oil_sm3 says how much oil is left before each, not whether "
            "it pays. field_voidage: "
            f"replacement_ratio is injection over voidage at reservoir conditions (Bo {BO} "
            f"m3/Sm3 from PUD p.20, low/high with Bo {BO_RANGE[1]}/{BO_RANGE[0]}; Bw {BW:g}; "
            "above the bubble point, so no free gas); surface_ratio is the same in surface "
            "volumes, which overstates replacement."
        ),
    }
