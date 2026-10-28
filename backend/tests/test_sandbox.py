"""run_python: the agent's own code over the field's data, cut at the run's date.

The runner runs for real here, in a local subprocess standing in for the Modal Sandbox (a
test double only: the server runs agent code nowhere but in Modal). Then the loop with its
call limit, and the checker on what the code returned.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from datetime import date
from functools import partial
from pathlib import Path

import pytest
from test_api import completion

from wellpilot.agent import sandbox as sandbox_module
from wellpilot.agent import sandbox_runner
from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import SUBMIT, run_agent
from wellpilot.agent.sandbox import OWN_CODE, RUN_PYTHON, run_python
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM

PROCESSED = Path(__file__).parent / "fixtures" / "processed"


class LocalSandbox:
    """The runner in a subprocess, with the image's two folders as temporary ones."""

    def __init__(self, root: Path) -> None:
        self.env = {"VOLVE_DIR": str(root / "volve"), "DATA_DIR": str(root / "data")}
        (root / "volve").mkdir(parents=True)
        for table in sandbox_runner.TABLES:
            if (PROCESSED / f"{table}.parquet").exists():
                shutil.copy(PROCESSED / f"{table}.parquet", root / "volve")
        self.closed = False

    async def exec(self, *args: str, limit_s: float) -> tuple[int, str, str]:
        command = [sys.executable, sandbox_runner.__file__, *args[2:]]

        def call() -> tuple[int, str, str]:
            try:
                done = subprocess.run(
                    command, capture_output=True, text=True, timeout=limit_s, env=self.env
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError from None
            return done.returncode, done.stdout, done.stderr

        return await asyncio.to_thread(call)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def box(tmp_path) -> LocalSandbox:
    sandbox = LocalSandbox(tmp_path)
    code, _, stderr = asyncio.run(
        sandbox.exec("python", "runner", "prepare", "2010-12-31", limit_s=60)
    )
    assert code == 0, stderr
    return sandbox


def call(box: LocalSandbox, code: str, purpose: str = "test") -> dict:
    return asyncio.run(run_python(box, json.dumps({"purpose": purpose, "code": code})))


def test_the_code_sees_the_data_up_to_the_runs_date_only(box, tmp_path) -> None:
    payload = call(
        box, "d = daily('F-12')\nresult = {'last': str(d['date'].max())[:10], 'days': len(d)}"
    )

    result = payload["result"]
    assert result["result"]["last"] == "2010-12-31" and result["rows_read"] > 1000
    assert result["warnings"] == [OWN_CODE]  # what code returns is never a fact
    # The full copy is gone: the code cannot read past the date by another path.
    assert not (tmp_path / "volve").exists()


def test_a_table_comes_back_column_wise_and_rounded_like_the_tools(box) -> None:
    payload = call(
        box,
        'result = sql("SELECT well, sum(oil_sm3) AS oil FROM production_daily '
        'GROUP BY well ORDER BY oil DESC LIMIT 2")',
    )

    table = payload["result"]["result"]
    assert table["columns"] == ["well", "oil"]
    assert table["rows"][0][0] in ("F-12", "F-14") and isinstance(table["rows"][0][1], int)


def test_errors_come_back_for_the_agent_to_fix(box, monkeypatch) -> None:
    failed = call(box, "result = 1 / 0")
    unset = call(box, "x = 5")
    monkeypatch.setattr(sandbox_module, "TIMEOUT_S", 1.0)
    slow = call(box, "import time\ntime.sleep(5)\nresult = 1")

    assert "ZeroDivisionError" in failed["error"] and "result = 1 / 0" in failed["error"]
    assert "sandbox_runner" not in failed["error"]  # only the agent's own frames
    assert "set no `result`" in unset["error"]
    assert "longer than 1 s" in slow["error"]


def test_code_that_reads_no_data_is_sent_back(box) -> None:
    """The fifth panel: the featured own-code run typed 36 monthly values into its code and
    read no data. Such a call now comes back as an error asking it to read the rows."""
    payload = call(box, "result = 4050 * 1.0")

    assert "read no rows" in payload["error"]


def test_numbers_typed_into_the_code_are_not_evidence(box) -> None:
    """The model's own 4050 comes back from its code; its data's oil total is evidence."""
    code = (
        "print('plateau 4050')\n"
        "result = {'typed': 4050, 'oil': float(daily('F-12')['oil_sm3'].sum())}"
    )
    payload = call(box, code)
    log = EvidenceLog()
    eid = log.add(RUN_PYTHON, {"purpose": "x", "code": code}, payload["result"], run_python).id
    oil = payload["result"]["result"]["oil"]

    def check(text: str, value: float) -> list[str]:
        finding = Finding(
            claim=f"F-12 {text} Sm3.",
            evidence_ids=[eid],
            confidence="inference",
            numbers=[DeclaredNumber(text=text.split()[-1], value=value, evidence_id=eid)],
        )
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert check(f"made {oil:.0f}", oil) == []
    assert check("plateaued at 4050", 4050)


def script(calls: list[str]) -> list:
    steps = [
        completion(tool_calls=[(RUN_PYTHON, {"purpose": "oil", "code": code})]) for code in calls
    ]
    return steps


def test_the_loop_offers_run_python_only_with_a_sandbox_and_limits_it(box) -> None:
    code = "result = float(daily('F-12')['oil_sm3'].sum())"
    submission = {
        "summary": "Done.",
        "summary_numbers": [],
        "findings": [
            {
                "claim": "F-12 had data.",
                "evidence_ids": ["E1"],
                "confidence": "inference",
                "numbers": [],
            }
        ],
    }
    replies = iter([*script([code] * 5), completion(tool_calls=[(SUBMIT, submission)])])
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return next(replies)

    from types import SimpleNamespace

    llm = LLM(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        model="fake",
        provider="fake",
    )
    store = ProductionStore(PROCESSED, as_of=date(2010, 12, 31))

    run = asyncio.run(run_agent("How much oil?", llm, store, python_fn=partial(run_python, box)))
    without = [t["function"]["name"] for t in requests[0]["tools"]]

    assert RUN_PYTHON in without
    calls = [c for s in run.steps for c in s.tool_calls if c["name"] == RUN_PYTHON]
    assert [("evidence_id" in c) for c in calls] == [True] * 4 + [False]
    assert "limited to 4 calls" in calls[4]["result"]["error"]
    assert run.verified and run.evidence[0].tool == RUN_PYTHON
    assert run.evidence[0].code.startswith("wellpilot.agent.sandbox.run_python@")


class Pool:
    """The run manager's pool, handing out local stand-ins."""

    available = True

    def __init__(self, root: Path) -> None:
        self.root, self.taken = root, []

    async def take(self, as_of):
        sandbox = LocalSandbox(self.root / f"box{len(self.taken)}")
        await sandbox.exec(
            "python", "runner", "prepare", as_of.isoformat() if as_of else "none", limit_s=60
        )
        self.taken.append(sandbox)
        return sandbox

    async def aclose(self) -> None:
        pass


async def test_a_run_through_the_api_gets_a_sandbox_and_closes_it(tmp_path, monkeypatch) -> None:
    from test_api import client_for, fake_llm, parse_sse, settings  # noqa: F401

    from wellpilot.config import Settings
    from wellpilot.main import create_app

    (tmp_path / "data").mkdir()
    shutil.copytree(PROCESSED, tmp_path / "data" / "processed")
    pool = Pool(tmp_path)
    monkeypatch.setattr("wellpilot.main.modal_pool", lambda *_: pool)
    code = "result = float(daily('F-12')['oil_sm3'].sum())"
    submission = {
        "summary": "Done.",
        "summary_numbers": [],
        "findings": [
            {
                "claim": "F-12 had data.",
                "evidence_ids": ["E1"],
                "confidence": "inference",
                "numbers": [],
            }
        ],
    }
    replies = [*script([code]), completion(tool_calls=[(SUBMIT, submission)])]
    app = create_app(
        Settings(
            _env_file=None,
            data_dir=tmp_path / "data",
            frontend_dir=tmp_path / "no-frontend",
            review_answers=False,
            python_sandbox=True,
            runs_per_client_per_hour=100,
            daily_token_budget=10**9,
        ),
        llm_factory=lambda: fake_llm(replies),
    )
    async with client_for(app) as client:
        started = await client.post(
            "/api/runs", json={"question": "How much oil?", "as_of": "2010-12-31"}
        )
        run_id = started.json()["id"]
        events = parse_sse((await client.get(f"/api/runs/{run_id}/events")).text)
        evidence = (await client.get(f"/api/runs/{run_id}/evidence/E1")).json()
        health = (await client.get("/api/health")).json()

    steps = [c for e in events if e["type"] == "step" for c in e["tool_calls"]]
    assert events[-1]["type"] == "final" and events[-1]["verified"], steps
    assert evidence["tool"] == RUN_PYTHON and evidence["arguments"]["code"] == code
    assert evidence["result"]["rows_read"] > 1000
    assert len(pool.taken) == 1 and pool.taken[0].closed
    assert health["python"] is True


async def test_after_a_failed_start_the_tool_is_not_offered_for_a_while(monkeypatch) -> None:
    """Modal down or the credits gone: runs go without run_python instead of being handed a
    tool whose every call fails; it is offered again after RETRY_AFTER_S."""
    from types import SimpleNamespace

    from wellpilot.agent.sandbox import SandboxPool

    pool = SandboxPool(PROCESSED)

    async def refused(*_, **__):
        raise RuntimeError("no credits")

    pool._modal = SimpleNamespace(App=SimpleNamespace(lookup=SimpleNamespace(aio=refused)))
    assert pool.available
    with pytest.raises(RuntimeError):
        await pool._start()

    assert not pool.available
    monkeypatch.setattr(SandboxPool, "RETRY_AFTER_S", 0)
    assert pool.available


async def test_the_cli_and_eval_tool_runs_in_a_sandbox_and_closes_it() -> None:
    """python_tool, for runs outside the API: the sandbox is taken for the run's date,
    the model's call runs in it, and it is closed when the run ends."""
    from datetime import date

    from wellpilot.agent.sandbox import python_tool

    taken, closed = [], []

    class Box:
        async def exec(self, *args, limit_s=30):
            return 0, "", ""

        async def close(self):
            closed.append(True)

    class Pool:
        available = True

        async def take(self, as_of):
            taken.append(as_of)
            return Box()

    async with python_tool(Pool(), date(2010, 12, 31)) as python_fn:
        assert python_fn is not None
        await python_fn('{"purpose": "a check", "code": "result = 1"}')
    assert taken == [date(2010, 12, 31)] and closed == [True]

    async with python_tool(None, None) as none:
        assert none is None


def test_a_finding_on_the_codes_result_is_an_inference(box) -> None:
    """The outside review: code that reads one row and returns a typed-in forecast passed
    as a fact. Whatever the code computes, a fact may not rest on it."""
    code = "d = sql('SELECT oil_sm3 FROM production_daily LIMIT 1')\nresult = 336023"
    payload = call(box, code)
    log = EvidenceLog()
    eid = log.add(RUN_PYTHON, {"purpose": "x", "code": code}, payload["result"], run_python).id
    finding = Finding(
        claim="F-12 will make 336,023 Sm3.",
        evidence_ids=[eid],
        confidence="fact",
        numbers=[DeclaredNumber(text="336,023", value=336023, evidence_id=eid)],
    )

    problems = check_submission(Submission(summary="Done.", findings=[finding]), log)
    assert any("marked fact" in p for p in problems), problems


def test_a_result_line_printed_by_the_code_is_not_taken(box) -> None:
    """Code that prints a line shaped like the runner's result and stops before the runner
    writes its own: the server only takes the line carrying its nonce."""
    code = (
        "import os, sys, json\n"
        "sys.__stdout__.write('__wellpilot_result__' + json.dumps({'result': 1, 'printed': '', "
        "'rows_read': 9999}) + '\\n')\n"
        "sys.__stdout__.flush()\n"
        "os._exit(0)"
    )
    payload = call(box, code)

    assert "error" in payload, payload
