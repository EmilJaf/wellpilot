"""run_python: the agent's own analysis, run in an isolated sandbox.

The fixed tools answer what they were built for. When a question needs a computation none
of them makes (two wells compared by a formula of the agent's, a share across a log, a
custom window), the agent writes Python and it runs in a Modal Sandbox: no network, its own
filesystem, a time and memory limit, and the field's tables cut at the run's as-of date
(sandbox_runner.prepare). The code and what it returned become evidence like any tool's: an
answer's numbers must be in that output, numbers typed into the code are not evidence, and
code that read no data carries a warning, so nothing it returns can be stated as fact.

One sandbox per run, closed at its end; the pool keeps one started ahead, so a run's first
call does not wait for a cold start. Without Modal (no package or no token) the tool is not
offered.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic import Field, ValidationError

from wellpilot.agent import sandbox_runner
from wellpilot.agent.tools import MAX_RESULT_CHARS, Args, Tool, to_jsonable

logger = logging.getLogger(__name__)

RUN_PYTHON = "run_python"
MAX_CALLS_PER_RUN = 4  # a fix-and-retry loop must not eat the run's time and credits
TIMEOUT_S = 30.0
NO_DATA = (
    "the code read no rows of data: its result comes from numbers written into the code, "
    "not from the field's data; treat it as an inference"
)


class PythonArgs(Args):
    purpose: str = Field(max_length=300, description="What the code computes, in one line.")
    code: str = Field(
        max_length=6000,
        description="Python. Assign the answer to `result`: a number, a dict, a list or a "
        "small DataFrame (at most a few dozen rows).",
    )


class Sandbox(Protocol):
    async def exec(self, *args: str, limit_s: float) -> tuple[int, str, str]:
        """Run a command within limit_s seconds; (return code, stdout, stderr)."""

    async def close(self) -> None: ...


def _parse(code: int, stdout: str, stderr: str) -> dict:
    marker = stdout.rfind(sandbox_runner.MARKER)
    if code != 0 or marker < 0:
        # The traceback's end says what went wrong; the agent fixes its code from it.
        tail = (stderr or stdout).strip()[-1500:]
        return {"error": f"the code failed:\n{tail}" if tail else "the code failed"}
    out = json.loads(stdout[marker + len(sandbox_runner.MARKER) :])
    if out["result"] is None:
        return {"error": "the code set no `result`; assign the answer to `result`"}
    return {"output": out}


async def run_python(sandbox: Sandbox, arguments: str) -> dict:
    """Run a run_python call from the model: {"result": ...} or {"error": ...}, like
    tools.execute. The result keeps the purpose and the code with what they returned."""
    try:
        args = PythonArgs.model_validate_json(arguments or "{}")
    except ValidationError as e:
        return {"error": f"invalid arguments for {RUN_PYTHON}: {e}"}
    try:
        parsed = _parse(
            *await sandbox.exec("python", RUNNER_PATH, "run", args.code, limit_s=TIMEOUT_S)
        )
    except TimeoutError:
        return {"error": f"the code ran longer than {TIMEOUT_S:.0f} s; make it smaller"}
    except Exception as e:  # the sandbox failed, not the code: cost the call, not the run
        logger.warning("sandbox call failed: %s", type(e).__name__)
        return {"error": f"{RUN_PYTHON} failed: {type(e).__name__}"}
    if "error" in parsed:
        return parsed
    out = parsed["output"]
    result = to_jsonable(
        {
            "purpose": args.purpose,
            "result": out["result"],
            "printed": out["printed"],
            "rows_read": out["rows_read"],
            "warnings": [] if out["rows_read"] else [NO_DATA],
        }
    )
    if len(json.dumps(result, ensure_ascii=False)) > MAX_RESULT_CHARS:
        return {"error": "the result is too large; return a summary or fewer rows"}
    return {"result": result}


PYTHON_TOOL = Tool(
    RUN_PYTHON,
    "Run your own Python analysis in an isolated sandbox, only when no other tool computes "
    "what you need (a comparison by your own formula, a custom window or ratio across the "
    "data). pandas as pd, numpy as np and scipy are there, with sql(query, params) for "
    "DuckDB SQL over production_daily, production_issues and production_logs (the same "
    "columns as run_sql's), and daily(well) for one well's days. Data after the as-of date "
    "is not there; there is no network. Assign the answer to `result`. At most "
    f"{MAX_CALLS_PER_RUN} calls per question, {TIMEOUT_S:.0f} s each. Numbers you type into "
    "the code are not evidence; on an error, fix the code from the traceback.",
    PythonArgs,
    run_python,
)

RUNNER_PATH = "/wellpilot/sandbox_runner.py"
VOLVE_PATH = "/volve"
# As backend/uv.lock has them, so the code runs on what the tools run on.
PACKAGES = ["pandas==3.1.0", "numpy==2.5.4", "scipy==1.18.1", "duckdb==2.0.0", "pyarrow==26.0.0"]


class ModalSandbox:
    """One Modal Sandbox: started from the image that holds the runner and the field's
    tables, prepared for one run's as-of date, closed at the run's end."""

    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox
        self.started = time.monotonic()

    async def exec(self, *args: str, limit_s: float) -> tuple[int, str, str]:
        async def call() -> tuple[int, str, str]:
            process = await self._sandbox.exec.aio(*args, timeout=int(limit_s))
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            return await process.wait.aio(), stdout, stderr

        # The sandbox ends the command at limit_s; a stuck connection must not hold the run.
        return await asyncio.wait_for(call(), limit_s + 15)

    async def close(self) -> None:
        await self._sandbox.terminate.aio()


class SandboxPool:
    """Modal sandboxes, one kept started ahead of the next run. A run takes it, prepares it
    for its date and closes it at its end; the next one starts at once."""

    APP = "wellpilot-python"
    # A sandbox's own time limit. One older than MAX_AGE_S is replaced, so a run that takes
    # it has at least 15 minutes (a run is stopped at 10).
    LIFETIME_S = 3600
    MAX_AGE_S = 2700
    KEEP_WARM_EVERY_S = 300
    RETRY_AFTER_S = 600  # after a failed start (Modal down, credits gone), runs go without

    def __init__(self, processed_dir: Path) -> None:
        import modal

        self._modal = modal
        image = modal.Image.debian_slim(python_version="3.13").pip_install(*PACKAGES)
        for table in sandbox_runner.TABLES:
            if (processed_dir / f"{table}.parquet").exists():
                image = image.add_local_file(
                    processed_dir / f"{table}.parquet", f"{VOLVE_PATH}/{table}.parquet", copy=True
                )
        self._image = image.add_local_file(sandbox_runner.__file__, RUNNER_PATH, copy=True)
        self._warm: asyncio.Task[ModalSandbox] | None = None
        self._failed_at: float | None = None

    @property
    def available(self) -> bool:
        """Offer run_python? Not for a while after a sandbox failed to start, so the agent
        is not handed a tool whose every call fails."""
        return self._failed_at is None or time.monotonic() - self._failed_at > self.RETRY_AFTER_S

    async def _start(self) -> ModalSandbox:
        try:
            app = await self._modal.App.lookup.aio(self.APP, create_if_missing=True)
            sandbox = await self._modal.Sandbox.create.aio(
                app=app,
                image=self._image,
                timeout=self.LIFETIME_S,
                block_network=True,
                cpu=1.0,
                memory=1024,
            )
        except Exception:
            self._failed_at = time.monotonic()
            raise
        self._failed_at = None
        return ModalSandbox(sandbox)

    def warm(self) -> None:
        """Start one ahead, if none is starting or ready."""
        if self._warm is None:
            self._warm = asyncio.create_task(self._start())

    async def keep_warm(self) -> None:
        """Forever: one started sandbox at hand, replaced before it gets old or after it
        failed, so no run waits for a cold start (about 20 s)."""
        while True:
            self.warm()
            task = self._warm
            if task is not None and task.done():
                failed = task.cancelled() or task.exception() is not None
                if failed or time.monotonic() - task.result().started > self.MAX_AGE_S:
                    self._warm = None
                    if not failed:
                        await _quietly(task.result().close())
                    self.warm()
            await asyncio.sleep(self.KEEP_WARM_EVERY_S)

    async def take(self, as_of: date | None) -> ModalSandbox:
        """A sandbox prepared for a run with data up to as_of; the next one starts warming."""
        sandbox = None
        if self._warm is not None:
            task, self._warm = self._warm, None
            try:
                sandbox = await task
            except Exception as e:  # a failed warm start: start one now instead
                logger.warning("warm sandbox failed: %s", type(e).__name__)
            if sandbox and time.monotonic() - sandbox.started > self.MAX_AGE_S:
                await _quietly(sandbox.close())
                sandbox = None
        sandbox = sandbox or await self._start()
        self.warm()
        code, _, stderr = await sandbox.exec(
            "python", RUNNER_PATH, "prepare", as_of.isoformat() if as_of else "none", limit_s=60
        )
        if code != 0:
            await _quietly(sandbox.close())
            raise RuntimeError(f"the sandbox could not prepare its data: {stderr[-300:]}")
        return sandbox

    async def aclose(self) -> None:
        if self._warm is not None:
            task, self._warm = self._warm, None
            task.cancel()
            with_result = await asyncio.gather(task, return_exceptions=True)
            if isinstance(with_result[0], ModalSandbox):
                await _quietly(with_result[0].close())


async def _quietly(close: Awaitable[None]) -> None:
    try:
        await close
    except Exception as e:
        logger.warning("closing a sandbox failed: %s", type(e).__name__)


def modal_pool(processed_dir: Path) -> SandboxPool | None:
    """The pool, or None when Modal is not installed or not logged in."""
    try:
        import modal  # noqa: F401
        from modal.config import config
    except ImportError:
        return None
    if not (config.get("token_id") and config.get("token_secret")):
        return None
    return SandboxPool(processed_dir)


PythonFn = Callable[[str], Awaitable[dict]]


async def _smoke() -> None:
    """Start a sandbox as a run would, run one piece of code, time each part, close."""
    from wellpilot.config import get_settings

    pool = modal_pool(get_settings().processed_dir)
    if pool is None:
        raise SystemExit("Modal is not set up: uv add modal, then uv run modal token new")
    started = time.monotonic()
    sandbox = await pool.take(date(2010, 12, 31))
    ready = time.monotonic()
    code = (
        "d = daily('F-12')\n"
        "result = {'last': str(d['date'].max())[:10], 'oil': float(d['oil_sm3'].sum())}"
    )
    payload = await run_python(sandbox, json.dumps({"purpose": "smoke", "code": code}))
    done = time.monotonic()
    await sandbox.close()
    warm = await pool.take(None)  # the one warmed meanwhile
    warmed = time.monotonic()
    await warm.close()
    await pool.aclose()
    print(json.dumps(payload, indent=1))
    print(
        f"cold start and data {ready - started:.1f} s, code {done - ready:.1f} s, "
        f"warm start and data {warmed - done:.1f} s"
    )


if __name__ == "__main__":
    asyncio.run(_smoke())
