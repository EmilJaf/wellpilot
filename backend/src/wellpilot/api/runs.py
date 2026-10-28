"""Agent runs over HTTP.

POST /api/runs starts a run in the background; GET /api/runs/{id}/events streams its
events as Server-Sent Events, replaying what already happened and then following live
(a reconnecting browser resumes after its Last-Event-ID). Finished runs are saved as
JSON under data/runs/, so results and evidence survive a restart and can be replayed.

The app is public, so every run costs us tokens: runs are limited per client per hour,
by a daily token budget, and by how many can run at once.
"""

import asyncio
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import AgentRun, run_agent, with_spans
from wellpilot.agent.review import review
from wellpilot.agent.sandbox import Sandbox, SandboxPool, run_python
from wellpilot.agent.tools import to_jsonable
from wellpilot.data.store import ProductionStore
from wellpilot.hindsight import hindsight
from wellpilot.llm import LLM
from wellpilot.meta_models import meta_models

logger = logging.getLogger(__name__)
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
STORE_CACHE_SIZE = 4  # stores per as_of date, about 14 MB each
FINISHED_IN_MEMORY = 50  # older finished runs are reloaded from disk on demand


class RunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    as_of: date | None = Field(None, description="Hide data after this day (backtests).")
    # xhigh and max are left out on purpose: they multiply a public run's cost.
    reasoning: Literal["minimal", "low", "medium", "high"] | None = None


class EditedFinding(BaseModel):
    index: int = Field(ge=1, le=50, description="1-based, as the answer numbers them")
    claim: str = Field(min_length=1, max_length=2000)
    confidence: Literal["fact", "inference"]


class CheckRequest(BaseModel):
    """A reader's edit of a finished answer, to be checked like the agent's submission."""

    findings: list[EditedFinding] = Field(default_factory=list, max_length=10)
    summary: str | None = Field(None, max_length=3000)


def _record_of(value: dict) -> dict:
    """A run as it happened, made JSON-safe: nothing rounded or cut. The tools round and cut
    their own results already; what the model sent is kept exactly, so a replay feeds back
    the same submission and the checker on a saved run sees what it saw live."""
    return to_jsonable(value, max_text=None, rounded=False)


SANDBOX_START_S = 90.0  # a cold start and the data's preparation


async def _close_quietly(sandbox: Sandbox) -> None:
    try:
        await asyncio.wait_for(sandbox.close(), 10)
    except Exception as e:  # its own time limit ends it anyway
        logger.warning("closing a sandbox failed: %s", type(e).__name__)


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
        # Whole strings: step events already bound reasoning (6000) and previews (300). The
        # model's numbers as it sent them: a submission declaring 3126.2 must replay as 3126.2.
        self.events.append({"seq": len(self.events), **_record_of(event)})
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def wait(self, seen: int) -> None:
        """Return once there are more than `seen` events or the run is over."""
        if len(self.events) > seen or self.done:
            return
        await self._changed.wait()

    def tokens(self) -> int:
        steps = sum(
            e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
            for e in self.events
            if e["type"] == "step"
        )
        return steps + sum(e.get("tokens", 0) for e in self.events if e["type"] == "review")

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
        self.evidence = {e.id: _record_of(asdict(e)) for e in run.evidence}
        self.result = _record_of(run.to_dict())
        self.result["meta_models"] = meta_models(self.events, self.evidence)
        self.status = "done"
        self.emit({"type": "final", **self.result})

    def fail(self, message: str) -> None:
        """End the run with an error, keeping the evidence gathered so far."""
        self.evidence = {e.id: _record_of(asdict(e)) for e in self.log}
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
        # Runs saved before display spans existed get them now, by today's rules.
        log = EvidenceLog.restore(record.evidence.values())
        if record.result is not None:
            with_spans(record.result, log)
            record.result["meta_models"] = meta_models(record.events, record.evidence)
        for event in record.events:
            if event.get("type") == "final":
                with_spans(event, log)
                event["meta_models"] = meta_models(record.events, record.evidence)
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
        voice_per_client_per_hour: int = 20,
        voice_max_audio_b64: int = 1_500_000,
        attacks_per_client_per_hour: int = 10,
        checks_per_client_per_hour: int = 120,
        stores_per_client_per_hour: int = 30,
        review_answers: bool = True,
        run_timeout_s: float | None = 600.0,
        featured: tuple[str, ...] | None = None,
        public: bool = False,
        operator_key: str = "",
        trusted_proxy: str = "none",
        sandboxes: SandboxPool | None = None,
    ) -> None:
        self.processed_dir = processed_dir
        self.sandboxes = sandboxes  # run_python in Modal; None: the tool is not offered
        self.runs_dir = runs_dir
        self.llm_factory = llm_factory
        self.max_active = max_active
        self.default_reasoning = default_reasoning
        self.runs_per_client_per_hour = runs_per_client_per_hour
        self.daily_token_budget = daily_token_budget
        self.voice_per_client_per_hour = voice_per_client_per_hour
        self.voice_max_audio_b64 = voice_max_audio_b64
        self.attacks_per_client_per_hour = attacks_per_client_per_hour
        self.checks_per_client_per_hour = checks_per_client_per_hour
        self.stores_per_client_per_hour = stores_per_client_per_hour
        self.review_answers = review_answers
        self.run_timeout_s = run_timeout_s
        # A tuple lists only those runs. None lists every saved run in development and none
        # in public, so a public deployment never shows other visitors' questions.
        self.featured = featured
        self.public = public
        if public and featured is None:
            logger.warning("public deployment without FEATURED_RUNS: no run is listed")
        self.operator_key = operator_key
        self.trusted_proxy = trusted_proxy
        self.runs: dict[str, RunRecord] = {}
        self._stores: OrderedDict[date | None, ProductionStore] = OrderedDict()
        self._stores_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._starts: dict[str, deque[float]] = {}
        self._voice: dict[str, deque[float]] = {}  # voice questions per client, by start time
        self._attacks: dict[str, deque[float]] = {}  # the model's attempts on the checker
        self._checks: dict[str, deque[float]] = {}  # readers' edits checked
        self._builds: dict[str, deque[float]] = {}  # stores built for a client's new dates
        self._building: dict[date | None, threading.Lock] = {}
        # Checking is CPU-bound Python: two at a time keep the server answering the rest.
        self.check_slots = threading.BoundedSemaphore(2)
        self._hindsight: OrderedDict[str, dict] = OrderedDict()  # a finished run's never changes
        # A restart (a deploy, an idle machine stopped) must not reset today's spend.
        self._spent: tuple[date, int] = (datetime.now(UTC).date(), self._spent_on_disk())

    def _spent_on_disk(self) -> int:
        """Tokens of the runs saved today (UTC), read from their files."""
        today = datetime.now(UTC).date().isoformat()
        spent = 0
        for path in self.runs_dir.glob("*.json") if self.runs_dir.exists() else []:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if str(data.get("created_at", "")).startswith(today):
                spent += sum(
                    e.get("prompt_tokens", 0) + e.get("completion_tokens", 0) + e.get("tokens", 0)
                    for e in data.get("events", [])
                    if e.get("type") in ("step", "review")
                )
        return spent

    def is_operator(self, key: str | None) -> bool:
        return bool(self.operator_key and key) and hmac.compare_digest(
            key.encode(), self.operator_key.encode()
        )

    async def aclose(self) -> None:
        """On shutdown: stop the runs in flight, which records each as interrupted and saves
        it. Bounded, to finish inside the platform's kill timeout."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            with suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        if self.sandboxes is not None:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.sandboxes.aclose(), 3)

    def store(
        self, as_of: date | None = None, client: str | None = None, privileged: bool = False
    ) -> ProductionStore:
        """The data cut at as_of, from the cache or built (about a second). A build holds a
        lock for its date only, so readers of other dates wait for nothing. With a client
        (a date a visitor typed), a build counts against its hourly limit: Refused."""
        with self._stores_lock:
            if as_of in self._stores:
                self._stores.move_to_end(as_of)
                return self._stores[as_of]
            building = self._building.setdefault(as_of, threading.Lock())
        with building:
            with self._stores_lock:
                if as_of in self._stores:  # built while this one waited
                    return self._stores[as_of]
            if client is not None:
                self._allow(
                    self._builds,
                    client,
                    privileged,
                    self.stores_per_client_per_hour,
                    f"limit of {self.stores_per_client_per_hour} new dates per hour reached; "
                    "pick a date already shown",
                    budget=False,
                )
            store = ProductionStore(self.processed_dir, as_of=as_of)
            with self._stores_lock:
                self._stores[as_of] = store
                while len(self._stores) > STORE_CACHE_SIZE:
                    self._stores.popitem(last=False)
                self._building.pop(as_of, None)
            return store

    def allow_check(self, client: str, privileged: bool = False) -> None:
        """A reader's edit checked: no tokens, but CPU. Raises Refused."""
        self._allow(
            self._checks,
            client,
            privileged,
            self.checks_per_client_per_hour,
            f"limit of {self.checks_per_client_per_hour} checks per hour reached; try again later",
            budget=False,
        )

    def hindsight_of(self, record: "RunRecord", compute: Callable[[], dict]) -> dict:
        """A finished run's hindsight, computed once."""
        if record.id not in self._hindsight:
            self._hindsight[record.id] = compute()
            while len(self._hindsight) > 64:
                self._hindsight.popitem(last=False)
        return self._hindsight[record.id]

    def tokens_spent_today(self) -> int:
        day, spent = self._spent
        today = datetime.now(UTC).date()
        finished = spent if day == today else 0
        return finished + sum(r.tokens() for r in self.runs.values() if not r.done)

    def _check_limits(self, client: str, privileged: bool = False) -> None:
        active = sum(not r.done for r in self.runs.values())
        if privileged:  # the operator: one reserved slot, no per-client or daily limit
            if active > self.max_active:
                raise Refused("too many runs in progress; try again in a minute")
            return
        if active >= self.max_active:
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

    def start(
        self, request: RunRequest, client: str = "local", privileged: bool = False
    ) -> RunRecord:
        # No key: 503 before the visitor's hourly allowance is touched.
        llm = self.llm_factory()  # raises RuntimeError when no API key is configured
        self._check_limits(client, privileged)
        record = RunRecord(new_run_id(), request.question.strip(), request.as_of)
        self.runs[record.id] = record
        effort = request.reasoning or self.default_reasoning
        task = asyncio.create_task(self._run(record, llm, effort))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def _run(self, record: RunRecord, llm: LLM, reasoning_effort: str) -> None:
        record.start(llm)
        self._save(record)  # killed outright, the file still says it was running
        # The sandbox gets ready while the model takes its first steps; a run_python call
        # waits for it only if it comes first.
        offered = self.sandboxes is not None and self.sandboxes.available
        ready = asyncio.create_task(self._sandbox(record)) if offered else None

        async def python_fn(arguments: str) -> dict:
            sandbox = await ready
            if sandbox is None:
                return {"error": "the sandbox is not available now; answer with the other tools"}
            return await run_python(sandbox, arguments)

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
                review_fn=review if self.review_answers else None,
                python_fn=python_fn if ready else None,
            )
            record.complete(run)
        except asyncio.CancelledError:
            # Shutdown or redeploy: record the run as stopped, not as running forever.
            record.fail("interrupted: the server stopped")
            raise
        except Exception as e:  # any failure ends the run with an error event
            logger.exception("run %s failed", record.id)
            # The stream is public: the type, not the message, which may name server paths.
            record.fail(f"the run failed ({type(e).__name__}); see the server log")
        finally:
            if ready is not None:  # closed before the run ends, never left running
                if ready.done() and not ready.cancelled() and (sandbox := ready.result()):
                    await _close_quietly(sandbox)
                elif not ready.done():
                    ready.cancel()  # one still starting ends by its own time limit
            self._finish(record)

    async def _sandbox(self, record: RunRecord) -> Sandbox | None:
        """The run's sandbox, prepared for its date; a run without one goes on without
        run_python rather than failing."""
        if self.sandboxes is None:
            return None
        try:
            return await asyncio.wait_for(self.sandboxes.take(record.as_of), SANDBOX_START_S)
        except Exception as e:
            logger.warning("run %s goes without a sandbox: %s", record.id, type(e).__name__)
            return None

    def allow_voice(self, client: str, privileged: bool = False) -> None:
        """A voice question costs a fraction of a run, so it has its own hourly limit per
        client; the daily budget still applies. Raises Refused."""
        self._allow(
            self._voice,
            client,
            privileged,
            self.voice_per_client_per_hour,
            f"limit of {self.voice_per_client_per_hour} voice questions per hour reached; "
            "type the question instead",
        )

    def allow_attack(self, client: str, privileged: bool = False) -> None:
        """The model's attempts on the checker cost about a cent each time; their own hourly
        limit per client, and the daily budget. Raises Refused."""
        self._allow(
            self._attacks,
            client,
            privileged,
            self.attacks_per_client_per_hour,
            f"limit of {self.attacks_per_client_per_hour} attempts per hour reached; "
            "edit the finding yourself instead",
        )

    def _allow(
        self,
        bucket: dict[str, deque[float]],
        client: str,
        privileged: bool,
        limit: int,
        refusal: str,
        budget: bool = True,
    ) -> None:
        if privileged:
            return
        now = time.monotonic()
        starts = bucket.setdefault(client, deque())
        while starts and now - starts[0] > 3600:
            starts.popleft()
        if len(starts) >= limit:
            raise Refused(refusal)
        if budget and self.tokens_spent_today() >= self.daily_token_budget:
            raise Refused("today's demo budget is used up")
        starts.append(now)

    def charge(self, tokens: int) -> None:
        """Tokens spent outside a run (a transcription) count against today's budget. Not
        written to disk: a restart forgets them, and they are small."""
        day, spent = self._spent
        today = datetime.now(UTC).date()
        self._spent = (today, (spent if day == today else 0) + tokens)

    def _finish(self, record: RunRecord) -> None:
        self.charge(record.tokens())
        self._save(record)
        finished = [r for r in self.runs.values() if r.done]
        for old in finished[:-FINISHED_IN_MEMORY]:
            self.runs.pop(old.id, None)

    def _save(self, record: RunRecord) -> None:
        try:
            record.save(self.runs_dir)
        except OSError:  # a full or read-only disk must not crash the run or the stream
            logger.exception("could not save run %s", record.id)

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
        elif self.public:
            ids = []
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


def client_id(request: Request, trusted_proxy: str = "none") -> str:
    """The visitor's address for rate limits. Proxy headers count only from the proxy we
    run behind: Fly-Client-IP on Fly, the last X-Forwarded-For hop (which our own proxy
    appended) with "xff". Otherwise a visitor could set them to dodge the limit."""
    if trusted_proxy == "fly" and (ip := request.headers.get("fly-client-ip")):
        return ip
    if trusted_proxy == "xff" and (forwarded := request.headers.get("x-forwarded-for")):
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@router.post("", status_code=202)
async def start_run(body: RunRequest, request: Request) -> dict:
    manager = _manager(request)
    privileged = manager.is_operator(request.headers.get("x-operator-key"))
    try:
        record = manager.start(body, client_id(request, manager.trusted_proxy), privileged)
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
    after = int(last) if last.isascii() and last.isdigit() else -1  # "²" is a digit too
    return EventSourceResponse(_manager(request).stream(record, after))


@router.get("/{run_id}/hindsight")
def run_hindsight(run_id: str, request: Request) -> dict:
    """What happened after a run's as-of date: its forecasts against the actual oil and the
    classic fit, and every well's next year. Only once the run has ended."""
    record = _record(request, run_id)
    if not record.done:
        raise HTTPException(409, "the run is still going")
    manager = _manager(request)
    run = {
        "question": record.question,
        "as_of": record.as_of.isoformat() if record.as_of else None,
        "verified": bool((record.result or {}).get("verified")),
        "result": record.result,
        "evidence": record.evidence,
        "events": record.events,  # the model's reasoning, read for the hidden period
    }
    try:
        return manager.hindsight_of(
            record, lambda: hindsight(run, manager.store(None), manager.store(record.as_of))
        )
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.post("/{run_id}/check")
def check_edit(run_id: str, body: CheckRequest, request: Request) -> dict:
    """The checker on a reader's edit of a finished answer, against the run's saved
    evidence: no model call. Declared numbers are carried to the edited text (a replaced
    number is checked as written, with the agent's evidence id). Only the edited parts'
    problems come back, so an unedited finding never changes the verdict."""
    record = _record(request, run_id)
    if not record.done:
        raise HTTPException(409, "the run is still going")
    result = record.result or {}
    if not result.get("findings"):
        raise HTTPException(409, "the run has no answer to edit")
    manager = _manager(request)
    try:
        manager.allow_check(
            client_id(request, manager.trusted_proxy),
            manager.is_operator(request.headers.get("x-operator-key")),
        )
    except Refused as e:
        raise HTTPException(429, str(e)) from None
    if not manager.check_slots.acquire(timeout=10):
        raise HTTPException(429, "the checker is busy; try again in a moment")
    try:
        return check_edits(
            record.question,
            result,
            EvidenceLog.restore(record.evidence.values()),
            [(e.index, e.claim, e.confidence) for e in body.findings],
            body.summary,
        )
    except IndexError as e:
        raise HTTPException(422, str(e)) from None
    finally:
        manager.check_slots.release()


@router.get("/{run_id}/evidence/{evidence_id}")
async def get_evidence(run_id: str, evidence_id: str, request: Request) -> dict:
    record = _record(request, run_id)
    if evidence_id in record.evidence:
        return record.evidence[evidence_id]
    live = record.log.get(evidence_id)
    if live is None:
        raise HTTPException(404, f"run {run_id} has no evidence {evidence_id}")
    return _record_of(asdict(live))
