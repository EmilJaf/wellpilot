"""A second opinion: once the checker has accepted an answer, the same model reads it as a
senior reservoir engineer would and questions the reasoning, which the checker does not
check (it checks that every number, date and quote is in its source, not that the
conclusion follows). The agent gets one chance to settle the points with its tools and
submit again; if the revision does not pass the checker, the checked answer stands.
"""

import asyncio
import json
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from wellpilot.agent.checker import Submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.llm import LLM, json_schema

TIMEOUT_S = 90.0
MAX_POINTS = 3
NEAR_DAYS = 45  # an event this close to a date the answer names is worth the reviewer's look
MAX_LEADS = 8
DAY = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")

SYSTEM = """\
You are a senior reservoir and production engineer reviewing a colleague's answer about an \
oil field. A checker has already verified that every number, date and quote in it is in \
the tool result it cites, so do not question the numbers. Judge only the reasoning:
- an explanation stated while another was not ruled out (a gauge fault, a shut-in, a choke \
change, work done on the well, a change in a neighbouring injector);
- a conclusion that goes beyond what the evidence shows, or a cause marked as fact;
- a period's average given as the level on one day ("to 640 around 2018-11-07" for the \
mean of the months after), or a forecast stated as what will happen;
- advice on a plug that ignores what the production log says it would leave flowing (its \
water_cut_left_above against the well's water cut now);
- a check the tools could have made and did not: well_operations for work on the well, \
detect_events for shut-ins and gauge failures, waterflood for liquid and choke, a decline \
fit on a window after the last change, the plan's limits for the whole field rather than \
one well.
Raise at most three points, only ones that could change the conclusion and that the tools \
can settle. You are also given what the agent's own tool results recorded near the dates \
the answer names (a choke moved with a rate change, a shut-in, a gauge failure, work on \
the well): an answer that explains such a change by the reservoir without saying why the \
recorded event does not explain it has a point to answer. If the reasoning is sound, \
return verdict "sound" and no points. Write plainly; each point names the check that \
would settle it."""


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
            "recorded_near_the_answers_dates": _leads(answer, log),
        },
        ensure_ascii=False,
        indent=1,
    )


def _days(text: str) -> list[date]:
    found = []
    for y, m, d in DAY.findall(text):
        try:
            found.append(date(int(y), int(m), int(d or 15)))
        except ValueError:
            continue
    return found


def _records(result) -> list[tuple[str, str]]:
    """(date, text) of the events and issues in a tool result: detect_events' list,
    field_screen's issues and well_operations' jobs."""
    if isinstance(result, list):
        return [
            (str(e.get("start", "")), str(e.get("summary", "")))
            for e in result
            if isinstance(e, dict)
        ]
    if not isinstance(result, dict):
        return []
    found = []
    for well in result.get("wells", []) if isinstance(result.get("wells"), list) else []:
        for issue in well.get("issues", []):
            if issue.get("since"):
                found.append((str(issue["since"]), f"{well.get('well')}: {issue.get('text', '')}"))
    for job in result.get("jobs", []) if isinstance(result.get("jobs"), list) else []:
        found.append(
            (
                str(job.get("start", "")),
                f"{job.get('kind')} job {job.get('start')} to {job.get('end')}",
            )
        )
    return found


def _leads(answer: Submission, log: EvidenceLog) -> list[str]:
    """What the agent's own results recorded within NEAR_DAYS of a date the answer names:
    the deterministic part of a second opinion, for the reviewer to weigh."""
    text = " ".join([answer.summary, *(f.claim + " " + (f.detail or "") for f in answer.findings)])
    named = _days(text)
    leads: list[str] = []
    # Plugs already in the hole, whatever dates the answer names: advice to set a plug where
    # one already sits (or below it) changes nothing.
    for e in log:
        if e.tool == "production_log" and isinstance(e.result, dict):
            for plug in e.result.get("plugs_set") or []:
                leads.append(
                    f"{e.id} production_log: a plug is already in the hole at "
                    f"{plug.get('depth_m')} m (set {plug.get('date')}); a new plug at or below "
                    "it, or in the blank pipe around it, shuts off nothing new"
                )
            # What a plug in each gap would leave flowing, on the latest log.
            surveys = e.result.get("surveys") or []
            for p in surveys[-1:] and surveys[-1].get("passes") or []:
                for gap in p.get("at_blank_pipe") or []:
                    if gap.get("already_isolated") or gap.get("water_cut_left_above") is None:
                        continue
                    leads.append(
                        f"{e.id} production_log: on the {surveys[-1].get('date')} {p.get('pass')} "
                        f"pass a plug in the {gap['blank_pipe_m']} m gap would leave a flow at "
                        f"water cut {gap['water_cut_left_above']}"
                    )
    for e in log:
        for day, what in _records(e.result):
            when = _days(day)
            if not when or not any(abs((when[0] - d).days) <= NEAR_DAYS for d in named):
                continue
            lead = f"{e.id} {e.tool}: {what[:300]}"
            if lead not in leads:
                leads.append(lead)
    return leads[:MAX_LEADS]


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
