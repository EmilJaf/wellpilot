"""Agent tools and loop, driven by a scripted fake LLM: no tokens are spent."""

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import numpy as np
import openai
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


def validate(request: dict) -> None:
    """The wire rules the providers enforce (see CLAUDE.md): a request that breaks one is a
    400 from Baseten or Meta, so a test that breaks one fails here first."""
    assert "tool_choice" not in request, "Meta accepts only the default tool_choice"
    for tool in request.get("tools", []):
        assert re.fullmatch(r"[a-zA-Z0-9_.-]+", tool["function"]["name"])
    open_calls: list[str] = []
    for message in request["messages"]:
        role = message["role"]
        assert isinstance(message.get("content"), str), f"{role} content must be a string"
        if role == "tool":
            assert message["tool_call_id"] in open_calls, "a tool reply needs its call"
            open_calls.remove(message["tool_call_id"])
            continue
        assert not open_calls, f"tool calls left unanswered: {open_calls}"
        if role == "assistant":
            open_calls = [call["id"] for call in message.get("tool_calls", [])]
    assert not open_calls, f"tool calls left unanswered: {open_calls}"


class FakeCompletions:
    """Replays scripted responses; an Exception in the script is raised instead."""

    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def create(self, **kwargs) -> ChatCompletion:
        request = json.loads(json.dumps(kwargs, default=str))
        validate(request)
        self.requests.append(request)
        response = self.responses[min(len(self.requests), len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response


def fake_llm(responses: list[ChatCompletion]) -> tuple[LLM, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLM(client=client, model="fake", provider="fake"), completions


def submission(claim: str, evidence_ids=("E1",), confidence="fact", numbers=None) -> dict:
    """A submission as the live model sends it, with declared numbers (numbers=None leaves
    them out, as answers from before declared numbers did)."""
    finding = {"claim": claim, "evidence_ids": list(evidence_ids), "confidence": confidence}
    if numbers is not None:
        finding["numbers"] = [
            {"text": t, "value": v, "evidence_id": evidence_ids[0]} for t, v in numbers
        ]
    return {
        "summary": "F-12 is the only producer here.",
        "findings": [finding],
        **({"summary_numbers": []} if numbers is not None else {}),
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


def test_calculate_reads_grouped_thousands_and_refuses_decimal_commas(store) -> None:
    def value(expression: str) -> float:
        return execute(store, "calculate", json.dumps({"expression": expression}))["result"][
            "value"
        ]

    assert value(" (292363 - 259265) / 276078 * 100") == 11.99  # leading space, saved run
    assert value("4 050 / 2") == 2025
    assert value("1,133,053 / 1830") == 619.15
    assert execute(store, "calculate", '{"expression": 4050}')["result"]["value"] == 4050
    assert (
        "decimal separator" in execute(store, "calculate", '{"expression": "0,80 * 100"}')["error"]
    )


def test_execute_reports_errors_instead_of_raising(store) -> None:
    assert "unknown tool" in execute(store, "nope", "{}")["error"]
    assert "invalid arguments" in execute(store, "production_monthly", '{"wel": "F-12"}')["error"]
    assert "unknown well" in execute(store, "fit_decline", '{"well": "F-99"}')["error"]
    assert "single SELECT" in execute(store, "run_sql", '{"sql": "DROP TABLE x"}')["error"]


def test_unknown_argument_names_are_errors(store) -> None:
    result = execute(store, "fit_decline", '{"well": "F-12", "from": "2010-01-15"}')

    assert "invalid arguments" in result["error"] and "from" in result["error"]


def test_monthly_table_counts_days_and_averages_gauges_over_flowing_days(tmp_path) -> None:
    days = pd.date_range("2011-01-01", periods=31, freq="D")
    flowing = np.arange(31) < 15  # 15 days on stream, then shut in
    pd.DataFrame(
        {
            "date": days,
            "well": "F-12",
            "well_type": "OP",
            "flow_kind": "production",
            "on_stream_hrs": np.where(flowing, 24.0, 0.0),
            "oil_sm3": np.where(flowing, 1000.0, 0.0),
            "water_sm3": 0.0,
            "gas_sm3": 0.0,
            "water_inj_sm3": np.nan,
            "water_cut": 0.0,
            "bhp_bar": np.nan,
            "whp_bar": np.where(flowing, 50.0, 90.0),  # bled up while shut in
            "choke_pct": 50.0,
        }
    ).to_parquet(tmp_path / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        tmp_path / ISSUES_FILE, index=False
    )

    result = execute(ProductionStore(tmp_path), "production_monthly", '{"well": "F-12"}')["result"]

    row = dict(zip(result["columns"], result["rows"][0], strict=True))
    assert (row["days"], row["producing_days"]) == (31, 15)
    assert row["whp_bar"] == 50
    assert row["oil_rate_per_producing_day"] == 1000


def test_to_jsonable_cuts_long_text_at_a_space() -> None:
    text = "rate 1146 " * 300

    cut = to_jsonable(text)

    assert len(cut) <= 2001 and cut.endswith("…")
    assert cut[:-1].split()[-1] in ("rate", "1146")  # no number cut in half
    assert to_jsonable(text, max_text=None) == text


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


def test_system_prompt_describes_only_the_data_up_to_the_cutoff(store, tmp_path) -> None:
    from datetime import date

    full = system_prompt(store)
    cut = system_prompt(ProductionStore(tmp_path, as_of=date(2010, 1, 15)))

    assert "available only up to" not in full
    assert "F-12 (first oil 2010-01-01)" in full and "data ends on 2010-03-01" in full
    assert "available only up to 2010-01-15" in cut and "data ends on 2010-01-15" in cut
    assert "2016" not in cut


def test_make_llm_needs_a_key() -> None:
    settings = Settings(_env_file=None, llm_provider="meta", model_api_key="")

    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        make_llm(settings)

    llm = make_llm(Settings(_env_file=None, llm_provider="meta", model_api_key="test-key"))
    assert llm.model == "muse-spark-1.3"
    assert str(llm.client.base_url).startswith("https://api.meta.ai/v1")


async def test_provider_default_reasoning_and_visible_reasoning_in_events(store) -> None:
    llm, completions = fake_llm(
        [completion("Done.", reasoning_content="I compare F-12 with the plan.")]
    )
    llm = LLM(llm.client, "fake", "baseten", default_reasoning="medium")
    events = []

    await run_agent("Hi", llm, store, max_steps=1, on_event=events.append)

    assert completions.requests[0]["reasoning_effort"] == "medium"
    step = next(e for e in events if e["type"] == "step")
    assert step["reasoning"] == "I compare F-12 with the plan."


async def test_declared_numbers_pass_end_to_end(store) -> None:
    llm, _ = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(
                tool_calls=[(SUBMIT, submission("F-12 produced on 60 days.", numbers=[("60", 60)]))]
            ),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.verified
    assert run.findings[0].numbers[0].text == "60"


async def test_text_between_tool_calls_does_not_end_the_run(store) -> None:
    """The audit's scenario: a plan, a tool, a rejected answer, a remark, then the fix."""
    llm, completions = fake_llm(
        [
            completion("1. List wells. 2. Count days."),
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("On 61 days.", numbers=[("61", 61)]))]),
            completion("Let me recompute the day count."),
            completion(tool_calls=[(SUBMIT, submission("On 60 days.", numbers=[("60", 60)]))]),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.verified and run.findings[0].claim == "On 60 days."
    nudges = [
        r["messages"][-1]["content"] for r in completions.requests[1:2] + completions.requests[4:5]
    ]
    assert "Continue by calling the tools" in nudges[0]
    assert "Fix the problems the checker listed" in nudges[1]


async def test_two_text_replies_in_a_row_keep_the_last_submission(store) -> None:
    llm, _ = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("On 61 days.", numbers=[("61", 61)]))]),
            completion("Hmm."),
            completion("I am done."),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.finished and not run.verified
    assert run.findings[0].claim == "On 61 days."


async def test_a_submission_written_as_text_is_checked_like_a_call(store) -> None:
    """Run fdb7132d6a1b: Muse Spark Open wrote its answer as the reply three times and the
    run ended with the JSON as its summary. Now the text goes through the checker."""
    rejected = json.dumps(submission("On 61 days.", numbers=[("61", 61)]))
    llm, completions = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(rejected),
            completion(
                f"```json\n{json.dumps(submission('On 60 days.', numbers=[('60', 60)]))}\n```"
            ),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.verified and run.findings[0].claim == "On 60 days."
    assert len(run.check_rounds) == 2 and "61" in run.check_rounds[0][0]
    reply = completions.requests[2]["messages"][-1]
    assert reply["role"] == "user" and "submit_findings tool" in reply["content"]
    assert '"accepted": false' in reply["content"]
    event = run.steps[1].to_event()["tool_calls"][0]
    assert event["name"] == SUBMIT and event["as_text"] and event["accepted"] is False


async def test_text_that_only_mentions_json_is_still_a_text_reply(store) -> None:
    llm, _ = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion('Next I will query {"well": "F-12"}.'),
            completion("Done."),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert not run.verified and run.check_rounds == []
    assert run.steps[1].tool_calls == []


async def test_a_provider_error_keeps_steps_evidence_and_the_last_submission(store) -> None:
    request = httpx.Request("POST", "https://inference.example/v1/chat/completions")
    llm, _ = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("On 61 days.", numbers=[("61", 61)]))]),
            openai.APITimeoutError(request=request),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert not run.finished and not run.verified
    assert run.stopped == "model provider error (APITimeoutError)"
    assert [e.id for e in run.evidence] == ["E1"] and len(run.steps) == 2
    assert run.findings[0].claim == "On 61 days."
    assert run.to_dict()["stopped"] == run.stopped


async def test_a_checker_error_is_a_rejection_not_a_crash(store, monkeypatch) -> None:
    import wellpilot.agent.loop as loop

    real = loop.check_submission
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OverflowError("int too large to convert to float")
        return real(*args)

    monkeypatch.setattr(loop, "check_submission", flaky)
    llm, completions = fake_llm(
        [
            completion(tool_calls=[("list_wells", {})]),
            completion(tool_calls=[(SUBMIT, submission("On 60 days.", numbers=[("60", 60)]))]),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert run.verified and len(run.check_rounds) == 2
    assert "could not read this submission (OverflowError)" in run.check_rounds[0][0]


async def test_the_last_steps_are_announced(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[("list_wells", {})])])

    run = await run_agent("Loop", llm, store, max_steps=3)

    assert run.stopped == "step limit"
    said = [r["messages"][-1]["content"] for r in completions.requests]
    assert "Two steps left" in said[1] and "last step" in said[2]


async def test_a_run_without_an_answer_by_step_8_is_told_to_submit(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[("list_wells", {})])])

    await run_agent("Loop", llm, store, max_steps=14)

    said = [r["messages"][-1]["content"] for r in completions.requests]
    assert "Submit your findings now" in said[7]  # step 8
    assert not any("Submit your findings now" in m for m in said[:7] + said[8:])


async def test_past_step_10_without_an_answer_only_submit_is_offered(store) -> None:
    llm, completions = fake_llm([completion(tool_calls=[("list_wells", {})])])

    await run_agent("Loop", llm, store, max_steps=14)

    offered = [[t["function"]["name"] for t in r["tools"]] for r in completions.requests]
    assert "list_wells" in offered[8] and offered[9] == [SUBMIT]  # steps 9 and 10
    assert "Only submit_findings is available" in completions.requests[9]["messages"][-1]["content"]


async def test_tool_calls_per_step_are_capped(store) -> None:
    from wellpilot.agent.loop import MAX_CALLS_PER_STEP

    many = [("list_wells", {})] * (MAX_CALLS_PER_STEP + 2)
    llm, completions = fake_llm(
        [
            completion(tool_calls=many),
            completion(tool_calls=[(SUBMIT, submission("On 60 days.", numbers=[("60", 60)]))]),
        ]
    )

    run = await run_agent("Days?", llm, store)

    assert len(run.evidence) == MAX_CALLS_PER_STEP
    errors = [c["result"].get("error", "") for c in run.steps[0].tool_calls]
    assert sum("too many tool calls" in e for e in errors) == 2
    assert run.verified  # every call was answered, so the next request was valid


async def test_the_time_limit_stops_a_hung_provider(store) -> None:
    class Hung:
        async def create(self, **kwargs):
            await asyncio.sleep(60)

    llm = LLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Hung())), model="f", provider="f"
    )

    run = await run_agent("Days?", llm, store, timeout_s=0.2)

    assert run.stopped.startswith("time limit")
    assert not run.finished


async def test_an_empty_reply_is_sent_back_as_text_and_a_cut_off_is_explained(store) -> None:
    cut = completion(None)
    cut.choices[0].finish_reason = "length"
    llm, completions = fake_llm(
        [cut, completion(tool_calls=[(SUBMIT, submission("x", numbers=[]))])]
    )

    await run_agent("Hi", llm, store)

    second = completions.requests[1]["messages"]
    assert second[2] == {"role": "assistant", "content": ""}
    assert "cut off by the output limit" in second[-1]["content"]


def test_hyperbolic_b_never_exceeds_one() -> None:
    """A waterflooded conventional well declines with b at most 1 (harmonic). The time
    machine's first run picked b = 1.82 by an AIC margin of 0.4 over harmonic; harmonic
    would have forecast +6% instead of +15% (the reservoir engineer judge's point)."""
    from pathlib import Path

    from wellpilot.analysis.decline import fit_decline, forecast
    from wellpilot.data.store import ProductionStore

    fixtures = ProductionStore(Path(__file__).parent / "fixtures" / "processed")
    days = fixtures.well_daily("F-12", "2010-05-02", "2010-12-31")
    result = fit_decline(days[days["flow_kind"] == "production"])
    hyperbolic = next(c for c in result.candidates if c.model == "hyperbolic")

    assert hyperbolic.b <= 1.0 + 1e-9
    p50 = forecast(result.best, 365, start=pd.Timestamp("2011-01-01"))
    assert p50.cumulative["p50"] < 391_544  # what b = 1.82 forecast


def test_calculate_takes_several_expressions(store) -> None:
    result = execute(store, "calculate", json.dumps({"expressions": ["4207 - 1888", "1 / 4"]}))

    assert result["result"] == {
        "results": [
            {"expression": "4207 - 1888", "value": 2319.0},
            {"expression": "1 / 4", "value": 0.25},
        ]
    }
    both = execute(store, "calculate", json.dumps({"expression": "1", "expressions": ["2"]}))
    bad = execute(store, "calculate", json.dumps({"expressions": ["1 / 0"]}))
    assert "either expression or expressions" in both["error"]
    assert "'1 / 0': division by zero" in bad["error"]
