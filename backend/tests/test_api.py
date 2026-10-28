"""HTTP API with a scripted fake LLM: no tokens are spent."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.evidence import code_ref
from wellpilot.agent.loop import SUBMIT
from wellpilot.agent.tools import detect_events
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
        frontend_dir=tmp_path / "no-frontend",  # the API alone, whether or not a build exists
        max_active_runs=1,
        runs_per_client_per_hour=100,
        daily_token_budget=10**9,
        review_answers=False,  # scripted replies: the review is tested on its own
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
                    "summary_numbers": [],
                    "findings": [
                        {
                            "claim": "F-12 produced on 60 days.",
                            "evidence_ids": ["E1"],
                            "confidence": "fact",
                            "numbers": [{"text": "60", "value": 60, "evidence_id": "E1"}],
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
    # The declared numbers and their display spans survive the save and the reload.
    final = replay[-1]
    assert final["verified"]
    assert final["findings"][0]["numbers"] == [{"text": "60", "value": 60, "evidence_id": "E1"}]
    assert final["findings"][0]["spans"][0]["evidence_id"] == "E1"
    assert final["summary_numbers"] == [] and final["summary_spans"] == []


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


async def test_a_readers_edit_is_checked_against_the_runs_evidence(settings) -> None:
    """ "Try to fool the checker": the finding is edited and checked again, with no model
    call. A changed number is checked as written; a cause on a fact is caught; the
    unedited text is accepted again."""
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        run_id, _ = await run_to_end(client)

        async def check(claim: str, confidence: str = "fact") -> dict:
            response = await client.post(
                f"/api/runs/{run_id}/check",
                json={"findings": [{"index": 1, "claim": claim, "confidence": confidence}]},
            )
            assert response.status_code == 200, response.text
            return response.json()

        same = await check("F-12 produced on 60 days.")
        number = await check("F-12 produced on 61 days.")
        cause = await check("F-12 produced on 60 days because the choke was opened.")
        hedged = await check("F-12 produced on 60 days because the choke was opened.", "inference")
        missing = await client.post(
            f"/api/runs/{run_id}/check",
            json={"findings": [{"index": 4, "claim": "Nothing.", "confidence": "fact"}]},
        )

    assert same["accepted"] and same["problems"] == []
    assert same["findings"][0]["spans"][0]["evidence_id"] == "E1"
    assert not number["accepted"] and "'61' = 61 is not in E1" in number["problems"][0]
    assert number["findings"][0]["numbers"] == [{"text": "61", "value": 61, "evidence_id": "E1"}]
    assert any("states a cause" in p for p in cause["problems"])
    assert hedged["accepted"]
    assert missing.status_code == 422 and "no finding 4" in missing.json()["detail"]


async def test_an_edit_waits_for_the_run_to_end(settings) -> None:
    gate = asyncio.Event()
    app = create_app(settings, llm_factory=lambda: fake_llm(gate=gate))
    async with client_for(app) as client:
        started = await client.post("/api/runs", json={"question": "First question"})
        run_id = started.json()["id"]
        early = await client.post(
            f"/api/runs/{run_id}/check",
            json={"findings": [{"index": 1, "claim": "Edited.", "confidence": "fact"}]},
        )
        gate.set()
        await client.get(f"/api/runs/{run_id}/events")
        late = await client.post(f"/api/runs/{run_id}/check", json={"summary": "F-12 made 70."})

    assert early.status_code == 409
    assert late.status_code == 200 and not late.json()["accepted"]
    assert late.json()["problems"][0].startswith("Summary")


async def test_model_failure_ends_the_run_with_an_error_event(settings) -> None:
    async with client_for(create_app(settings, llm_factory=lambda: fake_llm(fail=True))) as client:
        run_id, events = await run_to_end(client)
        run = (await client.get(f"/api/runs/{run_id}")).json()

    assert events[-1]["type"] == "run_error"
    # The stream is public: the error's type, never its message (it may name server paths).
    assert "RuntimeError" in events[-1]["message"]
    assert "model unavailable" not in events[-1]["message"]
    assert run["status"] == "error"


async def test_a_failed_run_keeps_its_evidence_after_a_restart(settings) -> None:
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise RuntimeError("unexpected failure")
        return SCRIPT[0]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = LLM(client=client, model="fake", provider="fake")
    async with client_for(create_app(settings, llm_factory=lambda: llm)) as http:
        run_id, events = await run_to_end(http)
    assert events[-1]["type"] == "run_error"

    async with client_for(create_app(settings, llm_factory=fake_llm)) as http:
        evidence = await http.get(f"/api/runs/{run_id}/evidence/E1")

    assert evidence.status_code == 200
    assert evidence.json()["tool"] == "list_wells"


async def test_long_model_reasoning_reaches_the_stream_whole(settings) -> None:
    reasoning = "I compare F-12 with the plan. " * 160  # about 4800 characters
    step = SCRIPT[0].model_copy(deep=True)
    step.choices[0].message = step.choices[0].message.model_copy(
        update={"reasoning_content": reasoning}
    )
    llm = fake_llm([step, SCRIPT[1]])
    async with client_for(create_app(settings, llm_factory=lambda: llm)) as client:
        _, events = await run_to_end(client)

    first = next(e for e in events if e["type"] == "step")
    assert first["reasoning"] == reasoning


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
        no_plan = await client.get("/api/field/plan")
        no_design = await client.get("/api/field/design")
        screened = (await client.get("/api/field/screen?as_of=2010-02-15")).json()
        code = (await client.get(f"/api/code/{code_ref(detect_events)}")).json()
        old_code = (
            await client.get("/api/code/wellpilot.agent.tools.detect_events@000000000000")
        ).json()
        no_code = await client.get("/api/code/os.system@000000000000")
        no_quotes = (await client.get("/api/reports/volve_pud/pages/23/quotes")).json()

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
    assert no_plan.status_code == 404  # no report facts extracted: no plan to compare with
    assert no_design.status_code == 404
    assert screened["as_of"] == "2010-02-15" and screened["wells"][0]["well"] == "F-12"
    assert code["same_as_run"] and code["source"].startswith("def detect_events(")
    assert code["file"] == "backend/src/wellpilot/agent/tools.py"
    assert old_code["same_as_run"] is False  # shown, but marked as changed since the run
    assert no_code.status_code == 404  # only registered tools
    assert no_quotes == []


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

    # A restart (a new app on the same runs) still counts the 220 tokens spent today.
    broke = settings.model_copy(update={"daily_token_budget": 150})
    async with client_for(create_app(broke, llm_factory=fake_llm)) as client:
        third = await client.post("/api/runs", json={"question": "Which well again?"})
    assert third.status_code == 429
    assert "budget" in third.json()["detail"]

    # The operator is never locked out by the limits.
    operated = broke.model_copy(update={"operator_key": "s3cret"})
    async with client_for(create_app(operated, llm_factory=fake_llm)) as client:
        wrong = await client.post(
            "/api/runs", json={"question": "Which well?"}, headers={"x-operator-key": "nope"}
        )
        right = await client.post(
            "/api/runs", json={"question": "Which well?"}, headers={"x-operator-key": "s3cret"}
        )
    assert wrong.status_code == 429
    assert right.status_code == 202


async def test_proxy_headers_count_only_from_the_trusted_proxy(settings) -> None:
    from starlette.requests import Request as StarletteRequest

    from wellpilot.api.runs import client_id

    scope = {
        "type": "http",
        "headers": [(b"fly-client-ip", b"1.2.3.4"), (b"x-forwarded-for", b"5.6.7.8, 9.9.9.9")],
        "client": ("10.0.0.1", 1234),
    }
    request = StarletteRequest(scope)

    assert client_id(request, "none") == "10.0.0.1"
    assert client_id(request, "fly") == "1.2.3.4"
    assert client_id(request, "xff") == "9.9.9.9"


async def test_a_public_deployment_lists_only_featured_runs(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        await run_to_end(client)
    public = settings.model_copy(update={"public": True})
    async with client_for(create_app(public, llm_factory=fake_llm)) as client:
        listed = (await client.get("/api/runs")).json()
    assert listed == []


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
    record.save(settings.runs_dir)

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


async def test_shutdown_saves_runs_in_flight_as_interrupted(settings) -> None:
    gate = asyncio.Event()  # never set: the model call hangs
    app = create_app(settings, llm_factory=lambda: fake_llm(gate=gate))
    async with client_for(app) as client:
        started = await client.post("/api/runs", json={"question": "Which well?"})
        run_id = started.json()["id"]
        await asyncio.sleep(0.05)
        saved = json.loads((settings.runs_dir / f"{run_id}.json").read_text())
        assert saved["status"] == "running"  # on disk from the start

        await app.state.runs.aclose()

    saved = json.loads((settings.runs_dir / f"{run_id}.json").read_text())
    assert saved["status"] == "error"
    assert saved["events"][-1]["message"] == "interrupted: the server stopped"


async def test_the_built_frontend_is_served_beside_the_api(settings, tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>WellPilot</title>")
    (dist / "assets" / "app-1a2b.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("not for the web")
    app = create_app(settings.model_copy(update={"frontend_dir": dist}), llm_factory=fake_llm)

    async with client_for(app) as client:
        page = await client.get("/?run=0123456789ab&replay=1")
        asset = await client.get("/assets/app-1a2b.js")
        icon = await client.get("/favicon.svg")
        api_miss = await client.get("/api/nope")
        escape = await client.get("/..%2Fsecret.txt")
        health = await client.get("/api/health")

    assert page.status_code == 200 and "<title>WellPilot" in page.text
    assert asset.text == "console.log(1)" and icon.text == "<svg/>"
    assert api_miss.status_code == 404 and api_miss.headers["content-type"].startswith(
        "application/json"
    )
    assert "not for the web" not in escape.text
    assert health.json()["status"] == "ok"


async def test_link_previews_get_the_sites_own_address(settings, tmp_path) -> None:
    """Discord and X read og:image only as an absolute URL; the address is the request's,
    over https in public, where Fly's proxy ends TLS and the app sees http."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text('<meta property="og:image" content="__ORIGIN__/og.png">')

    async def page(public: bool) -> str:
        app = create_app(
            settings.model_copy(update={"frontend_dir": dist, "public": public}),
            llm_factory=fake_llm,
        )
        async with client_for(app) as client:
            return (await client.get("/", headers={"host": "wellpilot.fly.dev"})).text

    assert 'content="https://wellpilot.fly.dev/og.png"' in await page(public=True)
    assert 'content="http://wellpilot.fly.dev/og.png"' in await page(public=False)


async def test_hindsight_only_for_a_run_with_an_as_of_date(settings) -> None:
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        started = await client.post("/api/runs", json={"question": "Which?", "as_of": "2010-02-15"})
        past_id = started.json()["id"]
        await client.get(f"/api/runs/{past_id}/events")
        all_id, _ = await run_to_end(client)

        past = await client.get(f"/api/runs/{past_id}/hindsight")
        everything = await client.get(f"/api/runs/{all_id}/hindsight")

    assert past.status_code == 200
    body = past.json()
    assert body["as_of"] == "2010-02-15" and body["forecasts"] == []
    (f12,) = body["wells"]
    assert f12["well"] == "F-12" and f12["oil_year_after_sm3"] > 0  # the days after the cut
    assert everything.status_code == 404


async def test_a_run_is_saved_with_the_models_numbers_exact(settings) -> None:
    """A declared value is kept as the model sent it: run c81ba8340854 declared 3126.2 for
    "3126.2" and was saved as 3126 (rounded like a tool table), so its replay failed."""
    submission = json.loads(SCRIPT[-1].choices[0].message.tool_calls[0].function.arguments)
    submission["findings"][0]["numbers"][0]["value"] = 60.0001
    script = [*SCRIPT[:-1], completion(tool_calls=[(SUBMIT, submission)])]
    async with client_for(create_app(settings, llm_factory=lambda: fake_llm(script))) as client:
        run_id, events = await run_to_end(client)
    restarted = create_app(settings, llm_factory=fake_llm)
    async with client_for(restarted) as client:
        saved = (await client.get(f"/api/runs/{run_id}")).json()

    submitted = next(
        c for e in events if e["type"] == "step" for c in e["tool_calls"] if c["name"] == SUBMIT
    )
    assert submitted["arguments"]["findings"][0]["numbers"][0]["value"] == 60.0001
    assert saved["result"]["findings"][0]["numbers"][0]["value"] == 60.0001


async def test_an_answer_says_what_each_meta_model_did(settings) -> None:
    """The line under the status: the model, its steps and tool calls; saved runs get it
    on load."""
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        run_id, events = await run_to_end(client)
    restarted = create_app(settings, llm_factory=fake_llm)
    async with client_for(restarted) as client:
        saved = (await client.get(f"/api/runs/{run_id}")).json()

    live = events[-1]["meta_models"]
    assert live["muse"]["model"] == "fake" and live["muse"]["tool_calls"] >= 1
    assert live["sam"] == {"figures": [], "logs": []}
    assert saved["result"]["meta_models"] == live


async def test_the_field_views_hide_the_days_after_end(tmp_path) -> None:
    """The time machine's Plan tab and well list: with `end`, nothing after that day (the
    outside review saw actual production to 2016 beside an answer as of 2010)."""
    import shutil

    shutil.copytree(Path(__file__).parent / "fixtures" / "processed", tmp_path / "processed")
    settings = Settings(_env_file=None, data_dir=tmp_path, frontend_dir=tmp_path / "none")
    async with client_for(create_app(settings, llm_factory=fake_llm)) as client:
        plan = (await client.get("/api/field/plan?end=2010-12-31")).json()
        whole = (await client.get("/api/field/plan")).json()
        design = (await client.get("/api/field/design?end=2010-12-31")).json()
        wells = (await client.get("/api/wells?end=2010-12-31")).json()

    assert plan["actual_data_ends"] == "2010-12-31" and plan["actual_last_year"] == 2010
    assert whole["actual_last_year"] > 2010
    months = [str(m) for m in design["months"]["rows"]] if design.get("months") else []
    assert all("2011" not in m for m in months)
    oil = wells["columns"].index("cum_oil_sm3")
    producers = {r[0] for r in wells["rows"] if r[oil]}
    assert producers == {"F-12", "F-14"}  # F-11, F-15 D and F-1 C came later
