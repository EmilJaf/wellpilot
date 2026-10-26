"""Agent tools and loop, driven by a scripted fake LLM: no tokens are spent."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import run_agent, system_prompt
from wellpilot.agent.tools import TOOLS, execute, to_jsonable, tool_schemas
from wellpilot.config import Settings
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production import DAILY_FILE, ISSUE_COLUMNS, ISSUES_FILE
from wellpilot.llm import LLM, make_llm


@pytest.fixture
def store(tmp_path) -> ProductionStore:
    dates = pd.date_range("2010-01-01", periods=60, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "well": "F-12",
            "well_type": "OP",
            "flow_kind": "production",
            "on_stream_hrs": 24.0,
            "oil_sm3": 2000.0 * np.exp(-0.002 * np.arange(60)),
            "water_sm3": 100.0,
            "gas_sm3": 300_000.0,
            "water_inj_sm3": np.nan,
            "water_cut": 0.05,
            "bhp_bar": 250.0,
            "whp_bar": 55.0,
            "choke_pct": 50.0,
        }
    )
    daily.to_parquet(tmp_path / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        tmp_path / ISSUES_FILE, index=False
    )
    return ProductionStore(tmp_path)


def completion(content=None, tool_calls=None, **extra) -> ChatCompletion:
    message = {"role": "assistant", "content": content, **extra}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }
    )


class FakeCompletions:
    def __init__(self, responses: list[ChatCompletion]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self.responses[min(len(self.requests), len(self.responses)) - 1]


def fake_llm(responses: list[ChatCompletion]) -> tuple[LLM, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLM(client=client, model="fake", provider="fake"), completions


def test_tool_schemas_are_plain_json_schema() -> None:
    schemas = json.dumps(tool_schemas())

    assert "anyOf" not in schemas
    assert '"title"' not in schemas
    for tool in TOOLS:
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", tool.name)


def test_execute_reports_errors_instead_of_raising(store) -> None:
    assert "unknown tool" in execute(store, "nope", "{}")["error"]
    assert "invalid arguments" in execute(store, "production_monthly", '{"wel": "F-12"}')["error"]
    assert "unknown well" in execute(store, "fit_decline", '{"well": "F-99"}')["error"]
    assert "single SELECT" in execute(store, "run_sql", '{"sql": "DROP TABLE x"}')["error"]


def test_to_jsonable() -> None:
    value = {
        "when": pd.Timestamp("2010-10-12"),
        "missing": float("nan"),
        "n": np.int64(3),
        "x": np.float64(1.234567),
        "flag": np.bool_(True),
        "nat": pd.NaT,
    }

    assert to_jsonable(value) == {
        "when": "2010-10-12",
        "missing": None,
        "n": 3,
        "x": 1.2346,
        "flag": True,
        "nat": None,
    }


async def test_agent_runs_tools_and_returns_the_answer(store) -> None:
    llm, completions = fake_llm(
        [
            completion("1. List wells.", tool_calls=[("list_wells", {})]),
            completion("F-12 produced the most oil."),
        ]
    )
    steps = []

    run = await run_agent("Which well produced most?", llm, store, on_step=steps.append)

    assert run.finished
    assert run.answer == "F-12 produced the most oil."
    assert run.total_tokens == (200, 20)
    assert steps == run.steps
    first_call = run.steps[0].tool_calls[0]
    assert first_call["name"] == "list_wells"
    assert first_call["result"]["result"][0]["well"] == "F-12"

    first, second = completions.requests
    assert "tool_choice" not in first
    assert {t["function"]["name"] for t in first["tools"]} >= {"list_wells", "run_sql"}
    assert second["messages"][2]["tool_calls"][0]["id"] == "call_0"
    tool_message = second["messages"][3]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_0"
    assert "F-12" in tool_message["content"]


async def test_agent_stops_at_the_step_limit(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[("list_wells", {})])])

    run = await run_agent("Loop forever", llm, store, max_steps=3)

    assert not run.finished
    assert run.answer is None
    assert len(completions.requests) == 3


async def test_reasoning_is_captured_when_the_provider_sends_it(store) -> None:
    llm, _ = fake_llm([completion("Done.", reasoning_content="Thinking about F-12.")])

    run = await run_agent("Hi", llm, store)

    assert run.steps[0].reasoning == "Thinking about F-12."


def test_system_prompt_mentions_the_cutoff() -> None:
    from datetime import date

    assert "2013-12-31" in system_prompt(date(2013, 12, 31))
    assert "available only up to" not in system_prompt(None)


def test_make_llm_needs_a_key() -> None:
    settings = Settings(_env_file=None, llm_provider="meta", model_api_key="")

    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        make_llm(settings)

    llm = make_llm(Settings(_env_file=None, llm_provider="meta", model_api_key="test-key"))
    assert llm.model == "muse-spark-1.3"
    assert str(llm.client.base_url).startswith("https://api.meta.ai/v1")
