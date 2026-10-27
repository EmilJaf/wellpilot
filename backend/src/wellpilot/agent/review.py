"""A second opinion: once the checker has accepted an answer, the same model reads it as a
senior reservoir engineer would and questions the reasoning, which the checker does not
check (it checks that every number, date and quote is in its source, not that the
conclusion follows). The agent gets one chance to settle the points with its tools and
submit again; if the revision does not pass the checker, the checked answer stands.
"""

import asyncio
import json
from typing import Literal

from pydantic import BaseModel, Field

from wellpilot.agent.checker import Submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.llm import LLM, json_schema

TIMEOUT_S = 90.0
MAX_POINTS = 3

SYSTEM = """\
You are a senior reservoir and production engineer reviewing a colleague's answer about an \
oil field. A checker has already verified that every number, date and quote in it is in \
the tool result it cites, so do not question the numbers. Judge only the reasoning:
- an explanation stated while another was not ruled out (a gauge fault, a shut-in, a choke \
change, work done on the well, a change in a neighbouring injector);
- a conclusion that goes beyond what the evidence shows, or a cause marked as fact;
- a check the tools could have made and did not: well_operations for work on the well, \
detect_events for shut-ins and gauge failures, waterflood for liquid and choke, a decline \
fit on a window after the last change, the plan's limits for the whole field rather than \
one well.
Raise at most three points, only ones that could change the conclusion and that the tools \
can settle. If the reasoning is sound, return verdict "sound" and no points. Write plainly; \
each point names the check that would settle it."""


class Point(BaseModel):
    point: str = Field(description="What in the reasoning may not hold, in one sentence.")
    check: str = Field(description="The tool call or comparison that would settle it.")


class Review(BaseModel):
    verdict: Literal["sound", "concerns"]
    points: list[Point] = Field(default_factory=list, max_length=MAX_POINTS)


def _brief(question: str, answer: Submission, log: EvidenceLog) -> str:
    """The answer and what its evidence came from, for the reviewer: claims with their
    labels and sources, and every tool the agent called."""
    tools = {e.id: e for e in log}
    findings = [
        {
            "claim": f.claim,
            "confidence": f.confidence,
            "sources": [
                f"{i} {tools[i].tool}({json.dumps(tools[i].arguments, default=str)})"
                for i in f.evidence_ids
                if i in tools
            ],
        }
        for f in answer.findings
    ]
    called = sorted({f"{e.tool}({json.dumps(e.arguments, default=str)})" for e in log})
    return json.dumps(
        {
            "question": question,
            "summary": answer.summary,
            "findings": findings,
            "tools_called": called,
        },
        ensure_ascii=False,
        indent=1,
    )


async def review(
    llm: LLM, question: str, answer: Submission, log: EvidenceLog
) -> tuple[Review, int]:
    """The review and the tokens it cost."""
    response = await asyncio.wait_for(
        llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _brief(question, answer, log)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "review", "schema": json_schema(Review)},
            },
            reasoning_effort="low",
        ),
        TIMEOUT_S,
    )
    result = Review.model_validate_json(response.choices[0].message.content or "")
    if result.verdict == "sound":
        result = Review(verdict="sound", points=[])
    usage = response.usage
    return result, (usage.prompt_tokens + usage.completion_tokens) if usage else 0


REVISE = """\
A reviewer read your answer (its numbers already passed the checker) and questions the \
reasoning:
{points}
For each point, run the tool that settles it, then submit your answer again with what you \
found: change, add or drop findings. If a point does not hold, add a finding that says why, \
with its evidence. Your checked answer stands if the revision does not pass the checker."""


def revise_message(result: Review) -> str:
    points = "\n".join(
        f"{n}. {p.point} Check: {p.check}" for n, p in enumerate(result.points, start=1)
    )
    return REVISE.format(points=points)
