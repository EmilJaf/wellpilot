import numpy as np
import pandas as pd
import pytest

from wellpilot.analysis.decline import (
    DAYS_PER_YEAR,
    arps_rate,
    calibrated_band,
    fit_decline,
    forecast,
    hindcast_errors,
    producing_day_rate,
)


def synthetic_daily(
    qi: float = 2000.0,
    di_per_year: float = 0.6,
    b: float = 0.5,
    days: int = 1000,
    noise: float = 0.03,
    hours: float = 24.0,
    seed: int = 1,
) -> pd.DataFrame:
    """Daily rows for one well following an Arps curve with multiplicative noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(days, dtype=float)
    rate = arps_rate(t, qi, di_per_year / DAYS_PER_YEAR, b) * np.exp(rng.normal(0, noise, days))
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=days, freq="D"),
            "on_stream_hrs": hours,
            "oil_sm3": rate * hours / 24.0,
        }
    )


def test_arps_special_cases() -> None:
    t = np.array([0.0, 100.0, 1000.0])

    assert arps_rate(t, 1000, 0.002, 0.0) == pytest.approx(1000 * np.exp(-0.002 * t))
    assert arps_rate(t, 1000, 0.002, 1.0) == pytest.approx(1000 / (1 + 0.002 * t))
    assert arps_rate(t, 1000, 0.002, 1e-4) == pytest.approx(
        arps_rate(t, 1000, 0.002, 0.0), rel=1e-3
    )


def test_producing_day_rate_scales_partial_days_and_skips_short_ones() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=4, freq="D"),
            "on_stream_hrs": [24.0, 12.0, 6.0, 24.0],
            "oil_sm3": [1000.0, 500.0, 300.0, 0.0],
        }
    )

    rate = producing_day_rate(daily)

    assert rate.tolist() == [1000.0, 1000.0]
    assert rate.index.day.tolist() == [1, 2]


def test_fit_recovers_hyperbolic_parameters() -> None:
    result = fit_decline(synthetic_daily(qi=2000, di_per_year=0.6, b=0.5))

    best = result.best
    assert best.model == "hyperbolic"
    assert best.qi == pytest.approx(2000, rel=0.05)
    assert best.di_per_year == pytest.approx(0.6, rel=0.15)
    assert best.b == pytest.approx(0.5, abs=0.15)
    assert best.r2_log > 0.9
    assert {c.model for c in result.candidates} == {"exponential", "hyperbolic", "harmonic"}


def test_fit_recovers_exponential_decline() -> None:
    result = fit_decline(synthetic_daily(qi=1500, di_per_year=0.4, b=0.0))

    assert result.best.b < 0.15
    assert result.best.di_per_year == pytest.approx(0.4, rel=0.1)


def test_fixed_model_and_window() -> None:
    daily = synthetic_daily()

    result = fit_decline(daily, start="2010-03-01", end="2010-12-31", model="harmonic")

    assert [c.model for c in result.candidates] == ["harmonic"]
    assert result.best.b == 1.0
    assert result.best.t0 == pd.Timestamp("2010-03-01")
    assert result.best.end == pd.Timestamp("2010-12-31")
    assert result.best.n_points == 306


def test_too_few_points_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        fit_decline(synthetic_daily(days=20))


def test_residual_trend_flags_a_late_acceleration() -> None:
    clean = synthetic_daily(b=0.0, di_per_year=0.3)
    accelerated = clean.copy()
    late = np.arange(len(clean)) >= 700
    accelerated.loc[late, "oil_sm3"] *= np.exp(-0.004 * np.arange(late.sum()))

    assert abs(fit_decline(clean).best.residual_trend) < 0.02
    assert fit_decline(accelerated).best.residual_trend < -0.05


def test_forecast_bands_are_ordered_and_track_the_truth() -> None:
    daily = synthetic_daily(qi=2000, di_per_year=0.6, b=0.5, days=1000)
    fit = fit_decline(daily, end="2011-12-31").best

    fc = forecast(fit, horizon_days=365, n_boot=200)

    assert fc.dates[0] == pd.Timestamp("2012-01-01")
    assert len(fc.dates) == 365
    assert np.all(fc.p90 <= fc.p50 + 1e-9)
    assert np.all(fc.p50 <= fc.p10 + 1e-9)
    assert fc.cumulative["p90"] <= fc.cumulative["p50"] <= fc.cumulative["p10"]
    t = (fc.dates - pd.Timestamp("2010-01-01")).days.to_numpy()
    truth = arps_rate(t, 2000, 0.6 / DAYS_PER_YEAR, 0.5).sum()
    assert fc.cumulative["p50"] == pytest.approx(truth, rel=0.05)


def test_forecast_cumulative_uses_uptime() -> None:
    fit = fit_decline(synthetic_daily(hours=12.0)).best

    fc = forecast(fit, horizon_days=100, n_boot=20)

    assert fit.uptime == pytest.approx(0.5)
    assert fc.cumulative["p50"] == pytest.approx(fc.p50.sum() * 0.5)


def test_block_bootstrap_widens_bands_for_autocorrelated_noise() -> None:
    rng = np.random.default_rng(3)
    ar = np.zeros(800)
    for i in range(1, len(ar)):
        ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 0.02)
    daily = synthetic_daily(b=0.0, di_per_year=0.4, days=800, noise=0.0)
    daily["oil_sm3"] *= np.exp(ar)
    fit = fit_decline(daily, model="exponential").best

    def width(block_days: int) -> float:
        cum = forecast(fit, horizon_days=365, n_boot=200, block_days=block_days).cumulative
        return cum["p10"] - cum["p90"]

    assert width(60) > 2 * width(1)


def test_a_short_window_still_gets_a_band_and_a_warning() -> None:
    """A 30-day block on a 30-point window resampled nothing: P90 = P50 = P10."""
    fit = fit_decline(synthetic_daily(days=30, b=0.0, noise=0.05), model="exponential").best

    fc = forecast(fit, 365)

    assert fc.cumulative["p10"] > fc.cumulative["p50"] > fc.cumulative["p90"]
    assert any("short window" in w for w in fc.warnings)
    assert forecast(fit_decline(synthetic_daily()).best, 365).warnings == []


def test_hindcast_errors_use_only_what_had_happened_by_the_cutoff() -> None:
    """On a clean exponential well, forecasts made the same way from earlier dates err
    little, and none of them looks past the cutoff."""
    daily = synthetic_daily(b=0.0, days=1500, noise=0.02)
    cutoff = pd.Timestamp("2013-03-31")

    errors = hindcast_errors(daily, cutoff, window_days=365, horizon_days=365)
    later = hindcast_errors(daily, cutoff + pd.Timedelta(days=180), 365, 365)

    assert len(errors) >= 5 and max(abs(e) for e in errors) < 0.1
    assert len(later) > len(errors)  # later cutoffs have more finished forecast years
    band = calibrated_band(1000.0, errors)
    assert band["p90"] < 1000.0 * 1.1 and band["p10"] > 1000.0 * 0.9
    assert band["p90"] <= band["p10"]


def test_no_calibrated_band_from_too_few_forecasts() -> None:
    assert calibrated_band(1000.0, [0.1, -0.2]) is None
