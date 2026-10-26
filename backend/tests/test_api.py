"""HTTP API with a scripted fake LLM: no tokens are spent."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import SUBMIT
from wellpilot.config import Settings
from wellpilot.ingest.production import DAILY_FILE, ISSUE_COLUMNS, ISSUES_FILE
from wellpilot.ingest.reports import PAGES_FILE
from wellpilot.llm import LLM
from wellpilot.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    processed = tmp_path / "processed"
    processed.mkdir()
    days = 60
    pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=days, freq="D"),
            "well": "F-12",
            "well_type": "OP",
            "flow_kind": "production",
            "on_stream_hrs": [24.0] * (days - 1) + [6.0],
            "oil_sm3": 2000.0 * np.exp(-0.002 * np.arange(days)),
            "water_sm3": 100.0,
            "gas_sm3": 300_000.0,
            "water_inj_sm3": np.nan,
            "water_cut": 0.05,
            "bhp_bar": 250.0,
            "whp_bar": 55.0,
            "choke_pct": 50.0,
        }
    ).to_parquet(processed / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        processed / ISSUES_FILE, index=False
    )
    pd.DataFrame(
        {
            "doc_id": ["volve_pud"],
            "page": [23],
            "pages": [53],
            "text": ["Maksimum vannkutt 0,80"],
            "has_text": [True],
            "image": ["reports/volve_pud/p023.png"],
            "width_px": [1240],
            "height_px": [1755],
        }
    ).to_parquet(processed / PAGES_FILE, index=False)
    (processed / "reports" / "volve_pud").mkdir(parents=True)
    (processed / "reports" / "volve_pud" / "p023.png").write_bytes(b"\x89PNG fake")
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        max_active_runs=1,
        runs_per_client_per_hour=100,
        daily_token_budget=10**9,
    )


def completion(content=None, tool_calls=()) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
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
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }
    )


SCRIPT = [
    completion("1. List wells.", tool_calls=[("list_wells", {})]),
    completion(
        tool_calls=[
            (
                SUBMIT,
                {
                    "summary": "F-12 is the only well.",
                    "findings": [
                        {
                            "claim": "F-12 produced on 60 days.",
                            "evidence_ids": ["E1"],
                            "confidence": "fact",
                        }
                    ],
                },
            )
        ]
    ),
]


def fake_llm(responses=SCRIPT, gate: asyncio.Event | None = None, fail: bool = False) -> LLM:
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if gate:
            await gate.wait()
        if fail:
            raise RuntimeError("model unavailable")
        return responses[min(len(calls), len(responses)) - 1]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLM(client=client, model="fake", provider="fake")


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse_sse(body: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


async def run_to_end(client: httpx.AsyncClient, question: str = "Which well?") -> tuple:
    started = await client.post("/api/runs", json={"question": question})
    assert started.status_code == 202
    run_id = started.json()["id"]
    events = parse_sse((await client.get(f"/api/runs/{run_id}/events")).text)
    return run_id, events


async def test_run_streams_events_and_ends_verified(settings) -> None:
    app = create_app(settings, llm_factory=fake_llm)
    async with client_for(app) as client:
        run_id, events = await run_to_end(client)

        assert [e["type"] for e in events] == [
            "run_started",
            "llm_call",
            "step",
            "llm_call",
            "step",
            "final",
        ]
        assert [e["seq"] for e in events] == list(range(6))
        first_step = events[2]
        assert first_step["tool_calls"][0]["evidence_id"] == "E1"
        assert "F-12" in first_step["tool_calls"][0]["preview"]
        assert events[4]["tool_calls"][0]["accepted"] is True
        final = events[-1]
        assert final["verified"] is True
        assert final["evidence"][0]["id"] == "E1"
        assert "result" not in final["evidence"][0]

        run = (await client.get(f"/api/runs/{run_id}")).json()
        assert run["status"] == "done" and run["result"]["summary"] == "F-12 is the only well."
        evidence = (await client.get(f"/api/runs/{run_id}/evidence/E1")).json()
        assert evidence["tool"] == "list_wells"
        assert evidence["result"]["rows"][0][0] == "F-12"
        assert (await client.get(f"/api/runs/{run_id}/evidence/E9")).status_code == 404
        assert (await client.get("/api/runs/0123456789ab")).status_code == 404
        assert (await client.get("/api/runs/../../etc")).status_code == 404
        assert [r["id"] for r in (await client.get("/api/runs")).json()] == [run_id]


async def test_finished_runs_are_saved_and_replayed_after_restart(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        run_id, events = await run_to_end(client)

    assert (settings.runs_dir / f"{run_id}.json").exists()
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        replay = parse_sse((await client.get(f"/api/runs/{run_id}/events")).text)
        evidence = await client.get(f"/api/runs/{run_id}/evidence/E1")

    assert replay == events
    assert evidence.json()["tool"] == "list_wells"


async def test_missing_api_key_is_503(settings) -> None:
    def no_key() -> LLM:
        raise RuntimeError("MODEL_API_KEY is not set")

    async with client_for(create_app(settings, llm_factory=no_key)) as client:
        response = await client.post("/api/runs", json={"question": "Which well?"})

    assert response.status_code == 503
    assert "MODEL_API_KEY" in response.json()["detail"]


async def test_too_many_active_runs_is_429(settings) -> None:
    gate = asyncio.Event()
    app = create_app(settings, llm_factory=lambda: fake_llm(gate=gate))
    async with client_for(app) as client:
        first = await client.post("/api/runs", json={"question": "First question"})
        second = await client.post("/api/runs", json={"question": "Second question"})
        gate.set()
        await client.get(f"/api/runs/{first.json()['id']}/events")

    assert first.status_code == 202
    assert second.status_code == 429


async def test_model_failure_ends_the_run_with_an_error_event(settings) -> None:
    async with client_for(create_app(settings, llm_factory=lambda: fake_llm(fail=True))) as client:
        run_id, events = await run_to_end(client)
        run = (await client.get(f"/api/runs/{run_id}")).json()

    assert events[-1]["type"] == "run_error"
    assert "model unavailable" in events[-1]["message"]
    assert run["status"] == "error"


async def test_question_is_validated(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        assert (await client.post("/api/runs", json={"question": "x"})).status_code == 422


async def test_data_endpoints(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        wells = (await client.get("/api/wells")).json()
        daily = (await client.get("/api/wells/F-12/daily?start=2010-01-01&end=2010-03-01")).json()
        missing = await client.get("/api/wells/F-99/daily")
        page = (await client.get("/api/reports/volve_pud/pages/23")).json()
        image = await client.get("/api/reports/volve_pud/pages/23/image")
        no_doc = await client.get("/api/reports/../pages/23/image")
        no_page = await client.get("/api/reports/volve_pud/pages/99/image")
        no_backtest = await client.get("/api/backtest")

    assert wells["rows"][0][0] == "F-12"
    rate = daily["columns"].index("oil_rate_per_producing_day")
    assert daily["rows"][0][rate] == pytest.approx(2000)
    assert daily["rows"][-1][rate] is None  # 6 h on stream: too short for a rate
    assert missing.status_code == 404
    assert page["text"] == "Maksimum vannkutt 0,80"
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert no_doc.status_code == 404
    assert no_page.status_code == 404
    assert no_backtest.status_code == 404


async def test_stream_resumes_after_last_event_id(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        run_id, events = await run_to_end(client)
        resumed = await client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "2"})

    assert [e["seq"] for e in parse_sse(resumed.text)] == [3, 4, 5]
    assert "event: run_error" not in resumed.text


async def test_runs_are_limited_per_client_and_by_the_daily_budget(settings) -> None:
    limited = settings.model_copy(update={"runs_per_client_per_hour": 1})
    async with client_for(create_app(limited, llm_factory=fake_llm)) as client:
        await run_to_end(client)
        second = await client.post("/api/runs", json={"question": "Which well again?"})
    assert second.status_code == 429
    assert "per hour" in second.json()["detail"]

    broke = settings.model_copy(update={"daily_token_budget": 150})
    async with client_for(create_app(broke, llm_factory=fake_llm)) as client:
        await run_to_end(client)  # spends 220 tokens
        third = await client.post("/api/runs", json={"question": "Which well again?"})
    assert third.status_code == 429
    assert "budget" in third.json()["detail"]


async def test_reasoning_above_high_is_refused(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        response = await client.post(
            "/api/runs", json={"question": "Which well?", "reasoning": "max"}
        )
    assert response.status_code == 422


def test_a_run_saved_mid_flight_loads_as_interrupted(settings) -> None:
    from wellpilot.api.runs import RunManager, RunRecord

    record = RunRecord("0123456789ab", "Which well?", None)
    record.emit({"type": "run_started"})
    manager = RunManager(settings.processed_dir, settings.runs_dir, fake_llm, max_active=1)
    manager._save(record)

    reloaded = RunManager(settings.processed_dir, settings.runs_dir, fake_llm, max_active=1)
    loaded = reloaded.get("0123456789ab")

    assert loaded.status == "error"
    assert loaded.events[-1]["type"] == "run_error"
    # It does not hold a slot: a new run can still start.
    assert reloaded._check_limits("someone") is None


async def test_featured_runs_hide_other_questions(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        run_id, _ = await run_to_end(client)
        await run_to_end(client, "A private question")

    public = settings.model_copy(update={"featured_runs": run_id})
    async with client_for(create_app(public, llm_factory=fake_llm)) as client:
        listed = (await client.get("/api/runs")).json()
    assert [r["id"] for r in listed] == [run_id]


async def test_tool_results_with_decimals_do_not_break_a_run(settings) -> None:
    sql = "SELECT count(*) * 100.0 AS pct, max(date) - min(date) AS span FROM production_daily"
    script = [
        completion(tool_calls=[("run_sql", {"sql": sql})]),
        SCRIPT[1],
    ]
    async with client_for(create_app(settings, llm_factory=lambda: fake_llm(script))) as client:
        _, events = await run_to_end(client)

    assert events[-1]["type"] == "final"
    assert events[2]["tool_calls"][0]["evidence_id"] == "E1"
