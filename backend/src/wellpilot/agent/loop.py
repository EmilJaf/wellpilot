"""The agent loop: ask the model, run the tools it calls, repeat until it answers.

This is the first, minimal version (plan + act). Evidence ids, the claim checker and
self-correction rounds build on it.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from wellpilot.agent.tools import execute, tool_schemas
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM

SYSTEM_PROMPT = """\
You are WellPilot, an analysis agent for petroleum engineers working on the Volve oil \
field (Norwegian North Sea, operated by Equinor, produced 2008-2016). Producers: F-1 C, \
F-11, F-12, F-14, F-15 D (F-5 produced briefly in 2016). Water injectors: F-4, F-5.

How you work:
- Start with a short numbered plan, then call tools. Get every number from a tool; never \
invent or estimate a number you have not computed.
- Check your results before concluding. If a decline fit is poor (r2_log below 0.7) or \
residual_trend is far from 0, refit on a window that starts after the last regime \
change. If a pressure is missing, check the events for a gauge failure before drawing \
any conclusion about the reservoir.
- Label inferences as hypotheses, never as facts from the data.
- Finish with concise findings. Each finding states the numbers and dates it rests on \
and the tool that produced them.{as_of_note}"""


@dataclass
class Step:
    index: int
    content: str | None
    reasoning: str | None
    tool_calls: list[dict] = field(default_factory=list)  # name, arguments, result
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


@dataclass
class AgentRun:
    question: str
    answer: str | None
    steps: list[Step]
    finished: bool

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


async def run_agent(
    question: str,
    llm: LLM,
    store: ProductionStore,
    *,
    max_steps: int = 10,
    on_step: Callable[[Step], None] | None = None,
) -> AgentRun:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(store.as_of)},
        {"role": "user", "content": question},
    ]
    tools = tool_schemas()
    steps: list[Step] = []

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

        for call in message.tool_calls or []:
            result = execute(store, call.function.name, call.function.arguments)
            step.tool_calls.append(
                {"name": call.function.name, "arguments": call.function.arguments, "result": result}
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )

        steps.append(step)
        if on_step:
            on_step(step)
        if not message.tool_calls:
            return AgentRun(question, message.content, steps, finished=True)

    return AgentRun(question, None, steps, finished=False)
