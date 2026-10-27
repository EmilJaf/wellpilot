"""Replay a saved run through the agent loop, without spending tokens.

The model's recorded replies are fed back as a scripted client; the tools and the checker
run for real, on today's code and data. Comparing the replay with the recording shows
whether a change in tools, data or checker would change what the agent saw or what the
checker said: evidence results, check rounds and the verdict. Featured demo runs are
replayed in CI (tests/test_replay_runs.py) once re-recorded on the frozen code.

    uv run python -m wellpilot.agent.replay <run id or path>   # print the differences
"""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import AgentRun, run_agent
from wellpilot.agent.vision import LOOK
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM


def _completion(step: dict) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": step.get("content")}
    if step.get("reasoning"):
        message["reasoning_content"] = step["reasoning"]
    if step["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": f"call_{step['index']}_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            }
            for i, call in enumerate(step["tool_calls"])
        ]
    return ChatCompletion.model_validate(
        {
            "id": f"replay-{step['index']}",
            "object": "chat.completion",
            "created": 0,
            "model": "replay",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {
                "prompt_tokens": step.get("prompt_tokens", 0),
                "completion_tokens": step.get("completion_tokens", 0),
                "total_tokens": step.get("prompt_tokens", 0) + step.get("completion_tokens", 0),
            },
        }
    )


async def replay(run: dict, processed_dir: Path) -> AgentRun:
    steps = [e for e in run["events"] if e.get("type") == "step"]
    replies = iter([_completion(step) for step in steps])

    async def create(**_):
        return next(replies)

    # A page reading is the model's answer too: replay what it said, in order.
    readings = iter(
        {"result": run["evidence"][c["evidence_id"]]["result"]}
        if c.get("evidence_id")
        else {"error": c.get("error", "recorded error")}
        for step in steps
        for c in step["tool_calls"]
        if c["name"] == LOOK
    )

    async def recorded_look(*_):
        return next(readings)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = LLM(client=client, model="replay", provider="replay")
    as_of = date.fromisoformat(run["as_of"]) if run.get("as_of") else None
    store = ProductionStore(processed_dir, as_of=as_of)
    return await run_agent(run["question"], llm, store, max_steps=len(steps), look_fn=recorded_look)


def differences(run: dict, replayed: AgentRun) -> list[str]:
    """What the replay changed: each evidence result, the check rounds, the verdict."""
    found = []
    saved = run["evidence"]
    for item in replayed.evidence:
        before = saved.get(item.id)
        if before is None:
            found.append(f"{item.id} ({item.tool}) is new")
        elif json.dumps(before["result"], sort_keys=True) != json.dumps(
            item.result, sort_keys=True
        ):
            found.append(f"{item.id} ({item.tool}) returns something else now")
    missing = set(saved) - {e.id for e in replayed.evidence}
    found += [f"{eid} is no longer produced" for eid in sorted(missing)]
    recorded = run["result"]["check_rounds"]
    if [bool(r) for r in recorded] != [bool(r) for r in replayed.check_rounds]:
        found.append(f"check rounds {recorded!r} -> {replayed.check_rounds!r}")
    if run["result"]["verified"] != replayed.verified:
        found.append(f"verified {run['result']['verified']} -> {replayed.verified}")
    return found


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if source is None:
        raise SystemExit("usage: python -m wellpilot.agent.replay <run id or path>")
    if not source.exists():
        source = get_settings().runs_dir / f"{sys.argv[1]}.json"
    run = json.loads(source.read_text())
    replayed = asyncio.run(replay(run, get_settings().processed_dir))
    found = differences(run, replayed)
    print("\n".join(found) if found else "the replay matches the recording")


if __name__ == "__main__":
    main()
