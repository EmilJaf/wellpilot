"""The time machine's reasoning, read for the hidden period: what the data holds only after
the as-of date. On the CI fixture (real Volve data): F-12 made 339,718 Sm3 of oil in 2011."""

from pathlib import Path

from wellpilot.data.store import ProductionStore
from wellpilot.reasoning_check import reasoning_check

FULL = ProductionStore(Path(__file__).parent / "fixtures" / "processed")


def run(*reasoning: str, evidence: dict | None = None) -> dict:
    return {
        "as_of": "2010-12-31",
        "evidence": evidence or {},
        "events": [
            {"type": "step", "index": i + 1, "reasoning": text} for i, text in enumerate(reasoning)
        ],
    }


def test_a_remembered_year_and_value_are_found() -> None:
    found = reasoning_check(
        run("Plan.", "In 2011 F-12 made about 340,000 Sm3, so the decline holds."), FULL
    )

    assert found["available"] and found["steps"] == 2
    assert found["later_dates"][0]["dates"] == ["2011"] and found["later_dates"][0]["step"] == 2
    assert "F-12's oil in 2011: 339,718 Sm3" in found["hidden_values"][0]["matches"]


def test_the_forecast_year_rates_and_what_the_tools_returned_are_not() -> None:
    forecast_window = {
        "E1": {
            "id": "E1",
            "tool": "forecast_production",
            "arguments": {"well": "F-12"},
            "result": {
                "forecast_from": "2011-01-01",
                "forecast_to": "2011-12-31",
                "p50_sm3": 359007,
            },
        }
    }

    found = reasoning_check(
        run(
            "I will forecast 2011 for F-12.",
            "The step was 2082 -> 1142 Sm3/d.",
            "The window runs 2011-01-01 to 2011-12-31 and gives 359,007 Sm3.",
            evidence=forecast_window,
        ),
        FULL,
    )

    assert found["later_dates"] == [] and found["hidden_values"] == []
    assert found["forecast_mentions"] == 1


def test_no_reasoning_to_read_says_so() -> None:
    assert reasoning_check(run("", ""), FULL)["available"] is False
