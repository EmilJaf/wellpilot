"""Agent tools and loop, driven by a scripted fake LLM: no tokens are spent."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import SUBMIT, run_agent, submit_tool_schema, system_prompt
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


def submission(claim: str, evidence_ids=("E1",), confidence="fact") -> dict:
    return {
        "summary": "F-12 is the only producer here.",
        "findings": [
            {"claim": claim, "evidence_ids": list(evidence_ids), "confidence": confidence}
        ],
    }


def test_tool_schemas_are_plain_json_schema() -> None:
    schemas = json.dumps([*tool_schemas(), submit_tool_schema()])

    assert "anyOf" not in schemas
    assert "$ref" not in schemas
    assert '"title"' not in schemas
    for tool in [*TOOLS, SimpleNamespace(name=SUBMIT)]:
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", tool.name)


def test_calculate_is_safe_arithmetic(store) -> None:
    assert execute(store, "calculate", '{"expression": "0.7532 / 0.0575"}')["result"] == {
        "expression": "0.7532 / 0.0575",
        "value": 13.1,
    }
    assert "error" in execute(store, "calculate", '{"expression": "__import__(\\"os\\")"}')
    assert "division by zero" in execute(store, "calculate", '{"expression": "1/0"}')["error"]
    assert "too large" in execute(store, "calculate", '{"expression": "9**999"}')["error"]


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
        "big": np.float64(4579609.55),
        "x": np.float64(1.234567),
        "small": 0.016462,
        "noise": 9.798e-17,
        "flag": np.bool_(True),
        "nat": pd.NaT,
    }

    assert to_jsonable(value) == {
        "when": "2010-10-12",
        "missing": None,
        "n": 3,
        "big": 4579610,
        "x": 1.23,
        "small": 0.01646,
        "noise": 0.0,
        "flag": True,
        "nat": None,
    }


async def test_verified_run_cites_evidence(store) -> None:
    llm, completions = fake_llm(
        [
            completion("1. List wells.", tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("F-12 produced on 60 days."))]),
        ]
    )
    steps = []

    run = await run_agent("Which well produced most?", llm, store, on_step=steps.append)

    assert run.finished and run.verified
    assert run.summary == "F-12 is the only producer here."
    assert run.findings[0].evidence_ids == ["E1"]
    assert [e.id for e in run.evidence] == ["E1"]
    assert run.evidence[0].code.startswith("wellpilot.agent.tools.list_wells@")
    assert run.check_rounds == [[]]
    assert run.total_tokens == (200, 20)
    assert steps == run.steps

    first, second = completions.requests
    assert "tool_choice" not in first
    names = {t["function"]["name"] for t in first["tools"]}
    assert names >= {"list_wells", "run_sql", "calculate", SUBMIT}
    assert second["messages"][2]["tool_calls"][0]["id"] == "call_0"
    tool_message = second["messages"][3]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_0"
    assert json.loads(tool_message["content"])["evidence_id"] == "E1"


async def test_rejected_submission_is_fixed_and_resubmitted(store) -> None:
    llm, completions = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("F-12 produced on 61 days."))]),
            completion(tool_calls=[(SUBMIT, submission("F-12 produced on 60 days."))]),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.verified
    assert len(run.check_rounds) == 2
    assert "61 not found in E1" in run.check_rounds[0][0]
    assert run.check_rounds[1] == []
    rejection = json.loads(completions.requests[2]["messages"][-1]["content"])
    assert rejection["accepted"] is False


async def test_out_of_check_rounds_returns_an_unverified_answer(store) -> None:
    llm, completions = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("F-12 produced on 61 days."))]),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.finished and not run.verified
    assert len(run.check_rounds) == 3
    assert run.findings[0].claim == "F-12 produced on 61 days."
    assert len(completions.requests) == 4


async def test_failed_tool_calls_get_no_evidence_id(store) -> None:
    llm, _ = fake_llm(
        [
            completion(tool_calls=[("fit_decline", {"well": "F-99"})]),
            completion(tool_calls=[(SUBMIT, submission("No such well.", evidence_ids=["E1"]))]),
        ]
    )

    run = await run_agent("F-99?", llm, store, max_steps=2)

    call = run.steps[0].tool_calls[0]
    assert "evidence_id" not in call
    assert "unknown well" in call["result"]["error"]
    assert run.evidence == []
    assert "unknown evidence E1" in run.check_rounds[0][0]


async def test_plain_text_answer_is_nudged_then_returned_unverified(store) -> None:
    llm, completions = fake_llm(
        [completion("F-12 is fine.", reasoning_content="Thinking about F-12.")]
    )

    run = await run_agent("Hi", llm, store)

    assert run.finished and not run.verified
    assert run.summary == "F-12 is fine."
    assert run.steps[0].reasoning == "Thinking about F-12."
    nudge = completions.requests[1]["messages"][-1]
    assert nudge["role"] == "user" and SUBMIT in nudge["content"]


async def test_reasoning_effort_is_sent_only_when_set(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[(SUBMIT, submission("x"))])])

    await run_agent("Hi", llm, store, max_steps=1, reasoning_effort="low")
    await run_agent("Hi", llm, store, max_steps=1)

    assert completions.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in completions.requests[1]


async def test_agent_stops_at_the_step_limit(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[("list_wells", {})])])

    run = await run_agent("Loop forever", llm, store, max_steps=3)

    assert not run.finished
    assert run.summary is None
    assert len(completions.requests) == 3
    assert [e.id for e in run.evidence] == ["E1", "E2", "E3"]


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
