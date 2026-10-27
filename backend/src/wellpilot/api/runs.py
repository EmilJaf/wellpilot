"""Agent runs over HTTP.

POST /api/runs starts a run in the background; GET /api/runs/{id}/events streams its
events as Server-Sent Events, replaying what already happened and then following live
(a reconnecting browser resumes after its Last-Event-ID). Finished runs are saved as
JSON under data/runs/, so results and evidence survive a restart and can be replayed.

The app is public, so every run costs us tokens: runs are limited per client per hour,
by a daily token budget, and by how many can run at once.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import AgentRun, run_agent
from wellpilot.agent.tools import to_jsonable
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM

logger = logging.getLogger(__name__)
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
STORE_CACHE_SIZE = 4  # stores per as_of date, about 14 MB each
FINISHED_IN_MEMORY = 50  # older finished runs are reloaded from disk on demand


class RunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    as_of: date | None = Field(None, description="Hide data after this day (backtests).")
    # xhigh and max are left out on purpose: they multiply a public run's cost.
    reasoning: Literal["minimal", "low", "medium", "high"] | None = None


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunRecord:
    def __init__(self, run_id: str, question: str, as_of: date | None) -> None:
        self.id = run_id
        self.question = question
        self.as_of = as_of
        self.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.status: Literal["running", "done", "error"] = "running"
        self.events: list[dict] = []
        self.result: dict | None = None
        self.evidence: dict[str, dict] = {}  # filled when the run ends (or from disk)
        self.log = EvidenceLog()  # live evidence while the run is going
        self._changed = asyncio.Event()

    @property
    def done(self) -> bool:
        return self.status != "running"

    def emit(self, event: dict) -> None:
        self.events.append({"seq": len(self.events), **to_jsonable(event)})
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def wait(self, seen: int) -> None:
        """Return once there are more than `seen` events or the run is over."""
        if len(self.events) > seen or self.done:
            return
        await self._changed.wait()

    def tokens(self) -> int:
        return sum(
            e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
            for e in self.events
            if e["type"] == "step"
        )

    def summary(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "created_at": self.created_at,
            "status": self.status,
            "verified": (self.result or {}).get("verified"),
        }

    def to_file(self) -> dict:
        return {
            **self.summary(),
            "events": self.events,
            "result": self.result,
            "evidence": self.evidence,
        }

    def start(self, llm: LLM) -> None:
        self.emit(
            {
                "type": "run_started",
                "id": self.id,
                "question": self.question,
                "as_of": self.as_of,
                "model": llm.model,
                "provider": llm.provider,
            }
        )

    def complete(self, run: AgentRun) -> None:
        self.evidence = {e.id: to_jsonable(asdict(e)) for e in run.evidence}
        self.result = to_jsonable(run.to_dict())
        self.status = "done"
        self.emit({"type": "final", **self.result})

    def fail(self, message: str) -> None:
        """End the run with an error, keeping the evidence gathered so far."""
        self.evidence = {e.id: to_jsonable(asdict(e)) for e in self.log}
        self.status = "error"
        self.emit({"type": "run_error", "message": message})

    def save(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{self.id}.json"
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(self.to_file(), ensure_ascii=False))
        os.replace(partial, path)  # atomic: readers never see a half-written file
        return path

    @classmethod
    def from_file(cls, data: dict) -> "RunRecord":
        as_of = date.fromisoformat(data["as_of"]) if data["as_of"] else None
        record = cls(data["id"], data["question"], as_of)
        record.created_at = data["created_at"]
        record.status = data["status"]
        record.events = data["events"]
        record.result = data["result"]
        record.evidence = data["evidence"]
        if record.status == "running":
            # Saved mid-run by a process that then stopped: it will never finish.
            record.status = "error"
            record.emit({"type": "run_error", "message": "interrupted by a server restart"})
        return record


class Refused(Exception):
    """The run was not started; the message says why."""


class RunManager:
    def __init__(
        self,
        processed_dir: Path,
        runs_dir: Path,
        llm_factory: Callable[[], LLM],
        *,
        max_active: int = 3,
        default_reasoning: str = "",
        runs_per_client_per_hour: int = 6,
        daily_token_budget: int = 3_000_000,
        run_timeout_s: float | None = 600.0,
        featured: tuple[str, ...] | None = None,
    ) -> None:
        self.processed_dir = processed_dir
        self.runs_dir = runs_dir
        self.llm_factory = llm_factory
        self.max_active = max_active
        self.default_reasoning = default_reasoning
        self.runs_per_client_per_hour = runs_per_client_per_hour
        self.daily_token_budget = daily_token_budget
        self.run_timeout_s = run_timeout_s
        # None lists every saved run (development); a tuple lists only those, so a public
        # deployment never shows other visitors' questions.
        self.featured = featured
        self.runs: dict[str, RunRecord] = {}
        self._stores: OrderedDict[date | None, ProductionStore] = OrderedDict()
        self._stores_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._starts: dict[str, deque[float]] = {}
        self._spent: tuple[date, int] = (datetime.now(UTC).date(), 0)

    def store(self, as_of: date | None = None) -> ProductionStore:
        with self._stores_lock:
            if as_of in self._stores:
                self._stores.move_to_end(as_of)
                return self._stores[as_of]
            store = ProductionStore(self.processed_dir, as_of=as_of)
            self._stores[as_of] = store
            while len(self._stores) > STORE_CACHE_SIZE:
                self._stores.popitem(last=False)
            return store

    def tokens_spent_today(self) -> int:
        day, spent = self._spent
        today = datetime.now(UTC).date()
        finished = spent if day == today else 0
        return finished + sum(r.tokens() for r in self.runs.values() if not r.done)

    def _check_limits(self, client: str) -> None:
        if sum(not r.done for r in self.runs.values()) >= self.max_active:
            raise Refused("too many runs in progress; try again in a minute")
        now = time.monotonic()
        starts = self._starts.setdefault(client, deque())
        while starts and now - starts[0] > 3600:
            starts.popleft()
        if len(starts) >= self.runs_per_client_per_hour:
            raise Refused(
                f"limit of {self.runs_per_client_per_hour} runs per hour reached; "
                "replay a recorded run meanwhile"
            )
        if self.tokens_spent_today() >= self.daily_token_budget:
            raise Refused("today's demo budget is used up; replay a recorded run instead")
        starts.append(now)

    def start(self, request: RunRequest, client: str = "local") -> RunRecord:
        self._check_limits(client)
        llm = self.llm_factory()  # raises RuntimeError when no API key is configured
        record = RunRecord(new_run_id(), request.question.strip(), request.as_of)
        self.runs[record.id] = record
        effort = request.reasoning or self.default_reasoning
        task = asyncio.create_task(self._run(record, llm, effort))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def _run(self, record: RunRecord, llm: LLM, reasoning_effort: str) -> None:
        record.start(llm)
        try:
            store = await asyncio.to_thread(self.store, record.as_of)
            run: AgentRun = await run_agent(
                record.question,
                llm,
                store,
                reasoning_effort=reasoning_effort,
                timeout_s=self.run_timeout_s,
                on_event=record.emit,
                evidence_log=record.log,
            )
            record.complete(run)
        except asyncio.CancelledError:
            # Shutdown or redeploy: record the run as stopped, not as running forever.
            record.fail("interrupted: the server stopped")
            raise
        except Exception as e:  # any failure ends the run with an error event
            logger.exception("run %s failed", record.id)
            record.fail(f"{type(e).__name__}: {e}")
        finally:
            self._finish(record)

    def _finish(self, record: RunRecord) -> None:
        day, spent = self._spent
        today = datetime.now(UTC).date()
        self._spent = (today, (spent if day == today else 0) + record.tokens())
        record.save(self.runs_dir)
        finished = [r for r in self.runs.values() if r.done]
        for old in finished[:-FINISHED_IN_MEMORY]:
            self.runs.pop(old.id, None)

    def get(self, run_id: str) -> RunRecord:
        if not RUN_ID.match(run_id):
            raise KeyError(run_id)
        if run_id not in self.runs:
            path = self.runs_dir / f"{run_id}.json"
            if not path.exists():
                raise KeyError(run_id)
            self.runs[run_id] = RunRecord.from_file(json.loads(path.read_text()))
        return self.runs[run_id]

    def recent(self, limit: int = 20) -> list[dict]:
        if self.featured is not None:
            ids = list(self.featured)
        else:
            ids = list(self.runs)
            if self.runs_dir.exists():
                ids += [p.stem for p in self.runs_dir.glob("*.json") if p.stem not in self.runs]
        summaries = []
        for run_id in ids:
            try:
                summaries.append(self.get(run_id).summary())
            except (KeyError, ValueError, OSError):  # missing or unreadable file: skip it
                continue
        return sorted(summaries, key=lambda s: s["created_at"], reverse=True)[:limit]

    async def stream(self, record: RunRecord, after: int = -1) -> AsyncIterator[dict]:
        seen = after + 1
        while True:
            while seen < len(record.events):
                event = record.events[seen]
                seen += 1
                # "error" is a built-in EventSource event, so failures go out as run_error.
                name = "run_error" if event["type"] == "error" else event["type"]
                yield {"event": name, "id": str(event["seq"]), "data": json.dumps(event)}
            if record.done:
                return
            await record.wait(seen)


router = APIRouter(prefix="/api/runs", tags=["runs"])


def _manager(request: Request) -> RunManager:
    return request.app.state.runs


def _record(request: Request, run_id: str) -> RunRecord:
    try:
        return _manager(request).get(run_id)
    except KeyError:
        raise HTTPException(404, f"no run {run_id}") from None


def client_id(request: Request) -> str:
    """The visitor's address. Behind Fly's proxy that is Fly-Client-IP; elsewhere the last
    X-Forwarded-For hop, which our own proxy appended, or the socket peer."""
    if ip := request.headers.get("fly-client-ip"):
        return ip
    if forwarded := request.headers.get("x-forwarded-for"):
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@router.post("", status_code=202)
async def start_run(body: RunRequest, request: Request) -> dict:
    try:
        record = _manager(request).start(body, client_id(request))
    except Refused as e:
        raise HTTPException(429, str(e)) from None
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    return {**record.summary(), "events_url": f"/api/runs/{record.id}/events"}


@router.get("")
async def list_runs(request: Request) -> list[dict]:
    return _manager(request).recent()


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request) -> dict:
    record = _record(request, run_id)
    return {**record.summary(), "result": record.result}


@router.get("/{run_id}/events")
async def run_events(run_id: str, request: Request) -> EventSourceResponse:
    record = _record(request, run_id)
    last = request.headers.get("last-event-id", "")
    after = int(last) if last.isdigit() else -1
    return EventSourceResponse(_manager(request).stream(record, after))


@router.get("/{run_id}/evidence/{evidence_id}")
async def get_evidence(run_id: str, evidence_id: str, request: Request) -> dict:
    record = _record(request, run_id)
    if evidence_id in record.evidence:
        return record.evidence[evidence_id]
    live = record.log.get(evidence_id)
    if live is None:
        raise HTTPException(404, f"run {run_id} has no evidence {evidence_id}")
    return to_jsonable(asdict(live))
