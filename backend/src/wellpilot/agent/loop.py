"""The agent loop: plan, call tools, submit findings, pass the claim checker.

Every successful tool result is logged as evidence with an id. The model finishes by
calling submit_findings; the checker verifies each finding's numbers and dates against
the evidence it cites, and rejected submissions go back to the model to fix.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import ValidationError

from wellpilot.agent.checker import Finding, Submission, check_submission
from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.agent.tools import TOOLS_BY_NAME, execute, parameters_schema, tool_schemas
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM

SUBMIT = "submit_findings"
MAX_CHECK_ROUNDS = 3

SYSTEM_PROMPT = """\
You are WellPilot, an analysis agent for petroleum engineers working on the Volve oil \
field (Norwegian North Sea, operated by Equinor, produced 2008-2016). Producers: F-1 C, \
F-11, F-12, F-14, F-15 D (F-5 produced briefly in 2016). Water injectors: F-4, F-5.

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
- A checker verifies every number and date in each finding against the evidence it \
cites. If it rejects your submission, fix the listed problems and submit again.{as_of_note}"""

NUDGE = (
    "Submit your answer with the submit_findings tool, citing the evidence ids of the tool "
    "results each finding rests on."
)


def submit_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": "Submit the final answer: a summary and findings that cite "
            "evidence ids. A checker verifies them; fix any problems it returns.",
            "parameters": parameters_schema(Submission),
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

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.prompt_tokens for s in self.steps),
            sum(s.completion_tokens for s in self.steps),
        )


def _reasoning(message: Any) -> str | None:
    """Providers return reasoning under different non-standard fields, if at all."""
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if extra.get(key):
            return str(extra[key])
    return None


def system_prompt(as_of: date | None) -> str:
    note = f"\n\nData is available only up to {as_of.isoformat()}." if as_of else ""
    return SYSTEM_PROMPT.format(as_of_note=note)


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
    on_step: Callable[[Step], None] | None = None,
) -> AgentRun:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(store.as_of)},
        {"role": "user", "content": question},
    ]
    tools = [*tool_schemas(), submit_tool_schema()]
    log = EvidenceLog()
    steps: list[Step] = []
    check_rounds: list[list[str]] = []
    last_submission: Submission | None = None
    nudged = False

    def finish(summary, findings, verified, finished=True) -> AgentRun:
        return AgentRun(
            question, summary, findings, list(log), check_rounds, verified, finished, steps
        )

    for index in range(1, max_steps + 1):
        started = time.monotonic()
        response = await llm.client.chat.completions.create(
            model=llm.model, messages=messages, tools=tools
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
                payload = execute(store, name, arguments)
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

        if accepted:
            return finish(accepted.summary, accepted.findings, verified=True)
        if last_submission and len(check_rounds) >= MAX_CHECK_ROUNDS:
            # Out of correction rounds: return the last answer, marked unverified.
            return finish(last_submission.summary, last_submission.findings, verified=False)
        if not message.tool_calls:
            if nudged:
                return finish(message.content, [], verified=False)
            nudged = True
            messages.append({"role": "user", "content": NUDGE})

    return finish(None, [], verified=False, finished=False)
