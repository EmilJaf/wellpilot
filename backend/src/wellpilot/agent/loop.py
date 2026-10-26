"""The agent loop: plan, call tools, submit findings, pass the claim checker.

Every successful tool result is logged as evidence with an id. The model finishes by
calling submit_findings; the checker verifies each finding's numbers and dates against
the evidence it cites, and rejected submissions go back to the model to fix.
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from pydantic import ValidationError

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.agent.tools import TOOLS_BY_NAME, execute, tool_schemas
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM, json_schema

SUBMIT = "submit_findings"
MAX_CHECK_ROUNDS = 3
PREVIEW_CHARS = 300
MAX_REASONING_CHARS = 6000  # Muse Spark Open's visible reasoning, per step, in the stream

SYSTEM_PROMPT = """\
You are WellPilot, an analysis agent for petroleum engineers working on the Volve oil \
field (Norwegian North Sea, operated by Equinor). {wells} Reports are indexed too, \
such as the February 2005 development plan (PUD, in Norwegian): \
use search_reports to compare what happened with what was planned, and quote the original \
wording with an English translation. Prefer facts whose quote_check is "exact".

How you work:
- Start with a short numbered plan, then call tools. Get every number from a tool; never \
invent or estimate a number you have not computed.
- Every successful tool result carries an evidence_id (E1, E2, ...).
- Check results before concluding. Tool warnings mean a result is not trustworthy as is: \
for a poor decline fit, refit on a window that starts after the last regime change. If a \
pressure is missing, check the events for a gauge failure before drawing any conclusion \
about the reservoir.
- Compute every derived number (ratios, differences, percent changes) with the calculate \
tool, so it has evidence too.
- Finish by calling submit_findings: a short summary and findings. Each finding cites the \
evidence ids its numbers and dates come from and is marked "fact" (shown directly by the \
evidence) or "inference" (an interpretation, such as a likely cause). A finding that \
rests on evidence with warnings must be an inference.
- Declare every number you write: in each finding's "numbers" (and the summary's \
"summary_numbers") list it with its text exactly as written, its full value, and the \
evidence id it comes from, e.g. {{"text": "4 050", "value": 4050, "evidence_id": "E3"}} \
or {{"text": "72%", "value": 0.72, "evidence_id": "E9"}}. Dates, years and well names need \
no declaration.
- A checker verifies every declared number, date and quote against the evidence. If it \
rejects your submission, fix the listed problems and submit again.{as_of_note}"""

NUDGE = (
    "Submit your answer with the submit_findings tool, citing the evidence ids of the tool "
    "results each finding rests on."
)


def submit_tool_schema() -> dict:
    parameters = json_schema(Submission)
    # Declared numbers are optional in the model only so old answers still load; the agent
    # must always send them.
    parameters["required"] = [*parameters["required"], "summary_numbers"]
    finding = parameters["properties"]["findings"]["items"]
    finding["required"] = [*finding["required"], "numbers"]
    return {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": "Submit the final answer: a summary and findings that cite "
            "evidence ids, with every number declared. A checker verifies them; fix any "
            "problems it returns.",
            "parameters": parameters,
        },
    }


@dataclass
class Step:
    index: int
    content: str | None
    reasoning: str | None
    tool_calls: list[dict] = field(default_factory=list)  # name, arguments, result, evidence_id
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def to_event(self) -> dict:
        """A compact, JSON-ready view for streaming: results become short previews; full
        results stay in the evidence log."""
        calls = []
        for call in self.tool_calls:
            result = call["result"]
            view: dict = {"name": call["name"], "arguments": _parse_arguments(call["arguments"])}
            if call["name"] == SUBMIT:
                view["accepted"] = result.get("accepted", False)
                view["problems"] = result.get("problems", [])
            elif "error" in result:
                view["error"] = result["error"]
            else:
                inner = result.get("result")
                text = json.dumps(inner, ensure_ascii=False)
                view["evidence_id"] = call.get("evidence_id")
                view["warnings"] = inner.get("warnings", []) if isinstance(inner, dict) else []
                view["preview"] = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
            calls.append(view)
        reasoning = self.reasoning or None
        if reasoning and len(reasoning) > MAX_REASONING_CHARS:
            reasoning = reasoning[:MAX_REASONING_CHARS] + "…"
        return {
            "index": self.index,
            "content": self.content,
            "reasoning": reasoning,
            "tool_calls": calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class AgentRun:
    question: str
    summary: str | None
    findings: list[Finding]
    evidence: list[Evidence]
    check_rounds: list[list[str]]  # problems found at each submission; [] = accepted
    verified: bool
    finished: bool
    steps: list[Step]
    summary_numbers: list[DeclaredNumber] = field(default_factory=list)

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.prompt_tokens for s in self.steps),
            sum(s.completion_tokens for s in self.steps),
        )

    def to_dict(self) -> dict:
        """The answer and an index of its evidence (results are served separately)."""
        tokens_in, tokens_out = self.total_tokens
        return {
            "question": self.question,
            "summary": self.summary,
            "findings": [f.model_dump() for f in self.findings],
            "summary_numbers": [n.model_dump() for n in self.summary_numbers],
            "verified": self.verified,
            "finished": self.finished,
            "check_rounds": self.check_rounds,
            "steps": len(self.steps),
            "tokens": {"input": tokens_in, "output": tokens_out},
            "evidence": [
                {k: v for k, v in asdict(e).items() if k != "result"} for e in self.evidence
            ],
        }


def _reasoning(message: Any) -> str | None:
    """Providers return reasoning under different non-standard fields, if at all."""
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if extra.get(key):
            return str(extra[key])
    return None


def describe_wells(store: ProductionStore) -> str:
    """The wells as the data shows them, so a run with an as-of date learns nothing about
    later years (not which wells come later, nor when the field stops)."""
    rows = store.query(
        """
        SELECT well,
               min(date) FILTER (WHERE oil_sm3 > 0) AS first_oil,
               bool_or(water_inj_sm3 > 0) AS injects,
               max(date) AS last_date
        FROM production_daily GROUP BY well ORDER BY well
        """
    ).rows
    if rows.empty:
        return "No production data is available."
    producers = [
        f"{r.well} (first oil {r.first_oil.date().isoformat()})"
        for r in rows.itertuples()
        if r.first_oil is not None and not pd.isna(r.first_oil)
    ]
    injectors = [r.well for r in rows.itertuples() if r.injects]
    last = rows["last_date"].max().date().isoformat()
    return (
        f"Producers in the data: {', '.join(producers) or 'none yet'}. "
        f"Water injectors: {', '.join(injectors) or 'none yet'}. The data ends on {last}."
    )


def system_prompt(store: ProductionStore) -> str:
    as_of = store.as_of
    note = f"\n\nData is available only up to {as_of.isoformat()}." if as_of else ""
    return SYSTEM_PROMPT.format(wells=describe_wells(store), as_of_note=note)


def _parse_arguments(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


async def run_agent(
    question: str,
    llm: LLM,
    store: ProductionStore,
    *,
    max_steps: int = 12,
    reasoning_effort: str = "",
    on_step: Callable[[Step], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    evidence_log: EvidenceLog | None = None,
) -> AgentRun:
    """Run the agent on one question.

    on_step gets each finished Step; on_event gets JSON-ready events for streaming:
    {"type": "llm_call", "step": n} before each model call, {"type": "step", ...} after.
    Pass evidence_log to read evidence while the run is still going.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(store)},
        {"role": "user", "content": question},
    ]
    tools = [*tool_schemas(), submit_tool_schema()]
    log = evidence_log if evidence_log is not None else EvidenceLog()
    steps: list[Step] = []
    check_rounds: list[list[str]] = []
    last_submission: Submission | None = None
    nudged = False

    def finish(summary, findings, verified, finished=True, submission=None) -> AgentRun:
        numbers = (submission.summary_numbers or []) if submission else []
        return AgentRun(
            question,
            summary,
            findings,
            list(log),
            check_rounds,
            verified,
            finished,
            steps,
            numbers,
        )

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    effort = reasoning_effort or llm.default_reasoning

    for index in range(1, max_steps + 1):
        emit({"type": "llm_call", "step": index})
        started = time.monotonic()
        response = await llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            tools=tools,
            **({"reasoning_effort": effort} if effort else {}),
        )
        message = response.choices[0].message
        usage = response.usage
        step = Step(
            index=index,
            content=message.content,
            reasoning=_reasoning(message),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=time.monotonic() - started,
        )

        assistant: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        messages.append(assistant)

        accepted: Submission | None = None
        for call in message.tool_calls or []:
            name, arguments = call.function.name, call.function.arguments
            record: dict = {"name": name, "arguments": arguments}
            if name == SUBMIT:
                try:
                    submission = Submission.model_validate_json(arguments or "{}")
                except ValidationError as e:
                    payload = {"accepted": False, "problems": [f"invalid submission: {e}"]}
                else:
                    last_submission = submission
                    problems = check_submission(submission, log)
                    check_rounds.append(problems)
                    payload = {"accepted": not problems, "problems": problems}
                    if not problems:
                        accepted = submission
            else:
                # Tools are synchronous (DuckDB, scipy); keep the event loop free for streams.
                payload = await asyncio.to_thread(execute, store, name, arguments)
                if "result" in payload:
                    evidence = log.add(
                        name,
                        _parse_arguments(arguments),
                        payload["result"],
                        TOOLS_BY_NAME[name].run,
                    )
                    payload = {"evidence_id": evidence.id, **payload}
                    record["evidence_id"] = evidence.id
            record["result"] = payload
            step.tool_calls.append(record)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)}
            )

        steps.append(step)
        if on_step:
            on_step(step)
        emit({"type": "step", **step.to_event()})

        if accepted:
            return finish(accepted.summary, accepted.findings, True, submission=accepted)
        if last_submission and len(check_rounds) >= MAX_CHECK_ROUNDS:
            # Out of correction rounds: return the last answer, marked unverified.
            return finish(
                last_submission.summary,
                last_submission.findings,
                False,
                submission=last_submission,
            )
        if not message.tool_calls:
            if nudged:
                return finish(message.content, [], verified=False)
            nudged = True
            messages.append({"role": "user", "content": NUDGE})

    # Out of steps: keep the last answer the model submitted, if any, marked unverified.
    if last_submission:
        return finish(
            last_submission.summary,
            last_submission.findings,
            False,
            finished=False,
            submission=last_submission,
        )
    return finish(None, [], verified=False, finished=False)
