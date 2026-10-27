"""Arps decline curve fitting and probabilistic forecasts.

Rates are per producing day: the day's volume scaled to 24 hours on stream, so shut-ins
and partial days do not look like decline. Fits are done on log rate, which treats noise
as multiplicative and keeps the early high-rate days from dominating.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

Model = Literal["exponential", "hyperbolic", "harmonic"]
MODELS: tuple[Model, ...] = ("exponential", "hyperbolic", "harmonic")
DAYS_PER_YEAR = 365.25
MIN_POINTS = 30
B_BOUNDS = (1e-3, 2.0)


def arps_rate(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Arps rate at t days after t0; di is the initial nominal decline per day."""
    t = np.asarray(t, dtype=float)
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def _exponential(t, log_qi, di):
    return log_qi - di * t


def _harmonic(t, log_qi, di):
    return log_qi - np.log1p(di * t)


def _hyperbolic(t, log_qi, di, b):
    return log_qi - np.log1p(b * di * t) / b


def producing_day_rate(
    daily: pd.DataFrame, column: str = "oil_sm3", min_hours: float = 12.0
) -> pd.Series:
    """Volume per producing day for days on stream at least min_hours with volume > 0."""
    ok = (daily["on_stream_hrs"] >= min_hours) & (daily[column] > 0)
    rate = daily.loc[ok, column] * 24.0 / daily.loc[ok, "on_stream_hrs"]
    return pd.Series(rate.to_numpy(), index=pd.DatetimeIndex(daily.loc[ok, "date"]), name=column)


@dataclass(frozen=True)
class DeclineFit:
    model: Model
    qi: float  # rate at t0, Sm3 per producing day
    di_per_year: float  # initial nominal decline
    b: float  # 0 = exponential, 1 = harmonic
    t0: pd.Timestamp  # first day of the fit window
    end: pd.Timestamp  # last day of the fit window
    n_points: int
    r2_log: float
    rmse_log: float
    aic: float
    # Mean log residual over the last third of the window: < 0 means the well is falling
    # faster than the curve at the end (e.g. water breakthrough), > 0 slower.
    residual_trend: float
    uptime: float  # mean fraction of calendar time on stream in the window
    t_days: np.ndarray = field(repr=False)
    log_rate: np.ndarray = field(repr=False)

    @property
    def di_per_day(self) -> float:
        return self.di_per_year / DAYS_PER_YEAR

    def rate(self, dates: pd.DatetimeIndex) -> np.ndarray:
        t = (pd.DatetimeIndex(dates) - self.t0).days.to_numpy()
        return arps_rate(t, self.qi, self.di_per_day, self.b)


@dataclass(frozen=True)
class DeclineResult:
    best: DeclineFit
    candidates: list[DeclineFit]


def _fit_model(model: Model, t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (qi, di per day, b) fitted on log rate."""
    slope, intercept = np.polyfit(t, y, 1)
    log_qi0, di0 = intercept, max(-slope, 1e-5)
    if model == "exponential":
        (log_qi, di), _ = curve_fit(
            _exponential, t, y, p0=[log_qi0, di0], bounds=([-np.inf, 0.0], [np.inf, 1.0])
        )
        return float(np.exp(log_qi)), float(di), 0.0
    if model == "harmonic":
        (log_qi, di), _ = curve_fit(
            _harmonic, t, y, p0=[log_qi0, di0], bounds=([-np.inf, 0.0], [np.inf, 1.0])
        )
        return float(np.exp(log_qi)), float(di), 1.0
    (log_qi, di, b), _ = curve_fit(
        _hyperbolic,
        t,
        y,
        p0=[log_qi0, di0, 0.5],
        bounds=([-np.inf, 0.0, B_BOUNDS[0]], [np.inf, 1.0, B_BOUNDS[1]]),
        maxfev=20_000,
    )
    return float(np.exp(log_qi)), float(di), float(b)


def _evaluate(
    model: Model,
    params: tuple[float, float, float],
    t: np.ndarray,
    y: np.ndarray,
    t0: pd.Timestamp,
    end: pd.Timestamp,
    uptime: float,
) -> DeclineFit:
    qi, di, b = params
    residuals = y - np.log(arps_rate(t, qi, di, b))
    n, k = len(y), (3 if model == "hyperbolic" else 2)
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    last_third = residuals[t >= t[0] + (t[-1] - t[0]) * 2 / 3]
    return DeclineFit(
        model=model,
        qi=qi,
        di_per_year=di * DAYS_PER_YEAR,
        b=b,
        t0=t0,
        end=end,
        n_points=n,
        r2_log=1.0 - rss / tss if tss > 0 else 0.0,
        rmse_log=float(np.sqrt(rss / n)),
        aic=n * np.log(max(rss, 1e-12) / n) + 2 * k,
        residual_trend=float(last_third.mean()),
        uptime=uptime,
        t_days=t,
        log_rate=y,
    )


def fit_decline(
    daily: pd.DataFrame,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    model: Model | Literal["auto"] = "auto",
    column: str = "oil_sm3",
    min_hours: float = 12.0,
) -> DeclineResult:
    """Fit Arps models to one well's daily rows within [start, end].

    With model="auto" all three models are fitted and the lowest AIC wins; every
    candidate is returned so the comparison can be shown.
    """
    window = daily
    if start is not None:
        window = window[window["date"] >= pd.Timestamp(start)]
    if end is not None:
        window = window[window["date"] <= pd.Timestamp(end)]
    rate = producing_day_rate(window, column, min_hours)
    if len(rate) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} producing days to fit, got {len(rate)}")

    t0, last = rate.index[0], rate.index[-1]
    t = (rate.index - t0).days.to_numpy().astype(float)
    y = np.log(rate.to_numpy())
    in_window = window[(window["date"] >= t0) & (window["date"] <= last)]
    uptime = float((in_window["on_stream_hrs"].fillna(0).clip(upper=24) / 24).mean())

    models = MODELS if model == "auto" else (model,)
    candidates = [_evaluate(m, _fit_model(m, t, y), t, y, t0, last, uptime) for m in models]
    best = min(candidates, key=lambda c: c.aic)
    return DeclineResult(best=best, candidates=candidates)


@dataclass(frozen=True)
class Forecast:
    dates: pd.DatetimeIndex
    # Petroleum convention: P90 is the low case (90% chance to exceed), P10 the high case.
    p90: np.ndarray
    p50: np.ndarray
    p10: np.ndarray
    # Calendar-time volume over the horizon: producing-day rate times the fit's uptime.
    cumulative: dict[str, float]
    warnings: list[str] = field(default_factory=list)  # about the band, not the fit


def _block_resample(residuals: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: stitch random runs of `block` consecutive residuals."""
    n = len(residuals)
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=-(-n // block))
    return residuals[(starts[:, None] + np.arange(block)).ravel()[:n]]


def forecast(
    fit: DeclineFit,
    horizon_days: int,
    *,
    n_boot: int = 300,
    block_days: int = 30,
    seed: int = 0,
    start: pd.Timestamp | None = None,
) -> Forecast:
    """Forecast horizon_days from `start` (default: the day after the fit window); bands
    come from a residual bootstrap.

    The bootstrap resamples the fit's log residuals in blocks of `block_days`, refits the
    same model and takes percentiles of the resulting curves. Daily residuals are strongly
    autocorrelated (choke changes, shut-in periods), and resampling single days would make
    the bands far too narrow.
    """
    first = pd.Timestamp(start) if start is not None else fit.end + pd.Timedelta(days=1)
    dates = pd.date_range(first, periods=horizon_days, freq="D")
    t_future = (dates - fit.t0).days.to_numpy().astype(float)
    p50 = arps_rate(t_future, fit.qi, fit.di_per_day, fit.b)

    fitted = np.log(arps_rate(fit.t_days, fit.qi, fit.di_per_day, fit.b))
    residuals = fit.log_rate - fitted
    rng = np.random.default_rng(seed)
    # On a short window a 30-day block is the whole series and every resample is the
    # original: the band collapses to zero width. Keep at least four blocks.
    block = min(block_days, max(2, len(residuals) // 4))
    samples = []
    for _ in range(n_boot):
        y_star = fitted + _block_resample(residuals, block, rng)
        try:
            qi, di, b = _fit_model(fit.model, fit.t_days, y_star)
        except RuntimeError:  # curve_fit did not converge on this resample
            continue
        samples.append(arps_rate(t_future, qi, di, b))
    curves = np.vstack(samples) if samples else p50[np.newaxis, :]

    cum = curves.sum(axis=1) * fit.uptime
    warnings = []
    if len(residuals) < 3 * block_days:
        warnings.append(
            f"short window ({len(residuals)} producing days): the P90-P10 band is unreliable; "
            "fit a longer window for a band."
        )
    if np.ptp(cum) == 0:
        warnings.append("the P90-P10 band has zero width: the bootstrap failed on this window.")
    return Forecast(
        dates=dates,
        p90=np.percentile(curves, 10, axis=0),
        p50=p50,
        p10=np.percentile(curves, 90, axis=0),
        cumulative={
            "p90": float(np.percentile(cum, 10)),
            "p50": float(p50.sum() * fit.uptime),
            "p10": float(np.percentile(cum, 90)),
        },
        warnings=warnings,
    )
