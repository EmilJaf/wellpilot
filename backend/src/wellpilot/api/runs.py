"""Agent runs over HTTP.

POST /api/runs starts a run in the background; GET /api/runs/{id}/events streams its
events as Server-Sent Events, replaying what already happened and then following live.
Finished runs are saved as JSON under data/runs/, so results and evidence survive a
restart and can be replayed later.
"""

import asyncio
import json
import logging
import re
import uuid
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


class RunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    as_of: date | None = Field(None, description="Hide data after this day (backtests).")
    reasoning: Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None = None


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

    @classmethod
    def from_file(cls, data: dict) -> "RunRecord":
        as_of = date.fromisoformat(data["as_of"]) if data["as_of"] else None
        record = cls(data["id"], data["question"], as_of)
        record.created_at = data["created_at"]
        record.status = data["status"]
        record.events = data["events"]
        record.result = data["result"]
        record.evidence = data["evidence"]
        return record


class Busy(Exception):
    pass


class RunManager:
    def __init__(
        self,
        processed_dir: Path,
        runs_dir: Path,
        llm_factory: Callable[[], LLM],
        *,
        max_active: int = 3,
        default_reasoning: str = "",
    ) -> None:
        self.processed_dir = processed_dir
        self.runs_dir = runs_dir
        self.llm_factory = llm_factory
        self.max_active = max_active
        self.default_reasoning = default_reasoning
        self.runs: dict[str, RunRecord] = {}
        self._stores: dict[date | None, ProductionStore] = {}
        self._tasks: set[asyncio.Task] = set()

    def store(self, as_of: date | None = None) -> ProductionStore:
        if as_of not in self._stores:
            self._stores[as_of] = ProductionStore(self.processed_dir, as_of=as_of)
        return self._stores[as_of]

    def start(self, request: RunRequest) -> RunRecord:
        if sum(not r.done for r in self.runs.values()) >= self.max_active:
            raise Busy
        llm = self.llm_factory()  # raises RuntimeError when no API key is configured
        record = RunRecord(uuid.uuid4().hex[:12], request.question, request.as_of)
        self.runs[record.id] = record
        effort = request.reasoning or self.default_reasoning
        task = asyncio.create_task(self._run(record, llm, effort))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def _run(self, record: RunRecord, llm: LLM, reasoning_effort: str) -> None:
        record.emit(
            {
                "type": "run_started",
                "id": record.id,
                "question": record.question,
                "as_of": record.as_of,
                "model": llm.model,
                "provider": llm.provider,
            }
        )
        try:
            store = await asyncio.to_thread(self.store, record.as_of)
            run: AgentRun = await run_agent(
                record.question,
                llm,
                store,
                reasoning_effort=reasoning_effort,
                on_event=record.emit,
                evidence_log=record.log,
            )
            record.evidence = {e.id: to_jsonable(asdict(e)) for e in run.evidence}
            record.result = to_jsonable(run.to_dict())
            record.status = "done"
            record.emit({"type": "final", **record.result})
        except Exception as e:  # any failure ends the run with an error event
            logger.exception("run %s failed", record.id)
            record.status = "error"
            record.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self._save(record)

    def _save(self, record: RunRecord) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{record.id}.json"
        path.write_text(json.dumps(record.to_file(), ensure_ascii=False))

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
        summaries = {r.id: r.summary() for r in self.runs.values()}
        if self.runs_dir.exists():
            for path in self.runs_dir.glob("*.json"):
                if path.stem not in summaries:
                    data = json.loads(path.read_text())
                    summaries[path.stem] = RunRecord.from_file(data).summary()
        return sorted(summaries.values(), key=lambda s: s["created_at"], reverse=True)[:limit]

    async def stream(self, record: RunRecord) -> AsyncIterator[dict]:
        seen = 0
        while True:
            while seen < len(record.events):
                event = record.events[seen]
                seen += 1
                yield {"event": event["type"], "id": str(event["seq"]), "data": json.dumps(event)}
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


@router.post("", status_code=202)
async def start_run(body: RunRequest, request: Request) -> dict:
    try:
        record = _manager(request).start(body)
    except Busy:
        raise HTTPException(429, "too many runs in progress; try again shortly") from None
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
    return EventSourceResponse(_manager(request).stream(record))


@router.get("/{run_id}/evidence/{evidence_id}")
async def get_evidence(run_id: str, evidence_id: str, request: Request) -> dict:
    record = _record(request, run_id)
    if evidence_id in record.evidence:
        return record.evidence[evidence_id]
    live = record.log.get(evidence_id)
    if live is None:
        raise HTTPException(404, f"run {run_id} has no evidence {evidence_id}")
    return to_jsonable(asdict(live))
