"""Backtest on a synthetic well; the agent method uses a scripted fake LLM (no tokens)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import SUBMIT
from wellpilot.analysis.decline import DAYS_PER_YEAR, arps_rate
from wellpilot.backtest import HORIZON_DAYS, Case, agent_row, calibrate, cases, summarize
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production import DAILY_FILE, ISSUE_COLUMNS, ISSUES_FILE
from wellpilot.llm import LLM


@pytest.fixture
def processed_dir(tmp_path):
    days = 4 * 365
    t = np.arange(days, dtype=float)
    rng = np.random.default_rng(0)
    oil = arps_rate(t, 3000, 0.4 / DAYS_PER_YEAR, 0.0) * np.exp(rng.normal(0, 0.03, days))
    pd.DataFrame(
        {
            "date": pd.date_range("2008-01-01", periods=days, freq="D"),
            "well": "F-12",
            "well_type": "OP",
            "flow_kind": "production",
            "on_stream_hrs": 24.0,
            "oil_sm3": oil,
            "water_sm3": 100.0,
            "gas_sm3": 300_000.0,
            "water_inj_sm3": np.nan,
            "water_cut": 0.05,
            "bhp_bar": 250.0,
            "whp_bar": 55.0,
            "choke_pct": 50.0,
        }
    ).to_parquet(tmp_path / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        tmp_path / ISSUES_FILE, index=False
    )
    return tmp_path


def test_cases_roll_every_six_months_and_sum_actual_oil(processed_dir) -> None:
    store = ProductionStore(processed_dir)

    found = cases(store)

    # Data ends 2011-12-30, so the last cutoff with a full 365-day future is 2010-07-31.
    assert [c.cutoff.date().isoformat() for c in found] == [
        "2009-07-31",
        "2010-01-31",
        "2010-07-31",
    ]
    daily = store.well_daily("F-12")
    window = daily[(daily["date"] > "2009-07-31") & (daily["date"] <= "2010-07-31")]
    assert found[0].actual_sm3 == pytest.approx(window["oil_sm3"].sum())


def test_calibrated_band_uses_only_errors_known_at_the_cutoff() -> None:
    # Eight yearly cases; each forecast period ends a year after its cutoff.
    errors = [0.1, -0.1, 0.2, -0.2, 0.0, 0.3, 0.05, 0.6]
    results = pd.DataFrame(
        {
            "method": "classic",
            "well": "F-12",
            "cutoff": pd.date_range("2009-12-31", periods=8, freq="YE"),
            "actual": 100.0,
            "p50": [100.0 * (1 + e) for e in errors],
            "p90": [100.0 * (1 + e) * 0.99 for e in errors],
            "p10": [100.0 * (1 + e) * 1.01 for e in errors],
        }
    )

    calibrated = calibrate(results)

    assert calibrated["error"].tolist() == pytest.approx(errors)
    # A narrow bootstrap band only holds an exact forecast.
    assert calibrated["in_bootstrap_band"].sum() == 1
    # Case k knows the k earlier errors whose forecast year had ended, so the first five
    # cases get no band.
    assert calibrated["calibrated_low"].isna().tolist() == [True] * 5 + [False] * 3
    # From then on each band is built only from earlier errors: case 7 (0.05) falls
    # inside, cases 6 (0.3) and 8 (0.6) are worse than anything seen before.
    assert calibrated["in_calibrated_band"].tolist()[5:] == [False, True, False]
    summary = summarize(calibrated)["classic"]
    assert summary["cases"] == 8
    assert summary["calibrated_band_cases"] == 3
    assert summary["calibrated_band_coverage"] == pytest.approx(1 / 3)


def completion(tool_calls) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                            for i, (name, args) in enumerate(tool_calls)
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }
    )


async def test_agent_forecast_is_read_from_its_evidence_and_cached(processed_dir) -> None:
    calls = []
    script = [
        completion(
            [
                (
                    "forecast_production",
                    {"well": "F-12", "start": "2009-01-01", "horizon_days": HORIZON_DAYS},
                )
            ]
        ),
        completion(
            [
                (
                    SUBMIT,
                    {
                        "summary": "Forecast done.",
                        "findings": [
                            {
                                "claim": "Forecast made.",
                                "evidence_ids": ["E1"],
                                "confidence": "fact",
                            }
                        ],
                    },
                )
            ]
        ),
    ]

    async def create(**kwargs):
        calls.append(kwargs)
        return script[min(len(calls), len(script)) - 1]

    llm = LLM(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "fake",
        "fake",
    )
    case = Case("F-12", pd.Timestamp("2010-01-31"), 500_000.0)

    row = await agent_row(llm, processed_dir, case)
    again = await agent_row(None, processed_dir, case)  # no LLM: from the cache
    uncached = await agent_row(None, processed_dir, Case("F-12", pd.Timestamp("2010-07-31"), 1.0))

    assert row["method"] == "agent"
    assert row["verified"] is True
    # The agent's chosen window, refitted on data up to the cutoff.
    assert row["window_start"] == pd.Timestamp("2009-01-01")
    assert row["p90"] <= row["p50"] <= row["p10"]
    assert again == row
    assert uncached is None
    assert len(calls) == 2
    # The agent only ever saw data up to the cutoff.
    system = calls[0]["messages"][0]["content"]
    assert "available only up to 2010-01-31" in system
