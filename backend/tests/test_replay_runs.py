"""Replay the frozen demo runs through the agent loop on today's code (no tokens spent).

See tests/fixtures/runs/README.md: a run here is one the public site replays, so its
evidence results, check rounds and verdict must not change silently.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wellpilot.agent.loop import run_agent
from wellpilot.agent.replay import _completion, differences, replay
from wellpilot.api.runs import RunRecord
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM

FIXTURES = Path(__file__).parent / "fixtures"
RUNS = {p.stem: json.loads(p.read_text()) for p in sorted((FIXTURES / "runs").glob("*.json"))}


@pytest.mark.skipif(not RUNS, reason="no frozen demo runs yet (see tests/fixtures/runs)")
@pytest.mark.parametrize("run_id", sorted(RUNS))
async def test_a_frozen_run_replays_as_recorded(run_id: str) -> None:
    run = RUNS[run_id]

    replayed = await replay(run, FIXTURES / "processed")

    assert differences(run, replayed) == []


CLAIM = (
    "F-12's mean oil rate fell from 4 050 to 1 146 Sm3/d between the periods before and "
    "after 2010-07-18."
)
SUBMISSION = {
    "summary": "F-12's rate shifted down around 2010-07-18.",
    "summary_numbers": [],
    "findings": [
        {
            "claim": CLAIM,
            "evidence_ids": ["E1"],
            "confidence": "fact",
            "numbers": [
                {"text": "4 050", "value": 4050.37, "evidence_id": "E1"},
                {"text": "1 146", "value": 1145.72, "evidence_id": "E1"},
            ],
        }
    ],
}
EVENTS_CALL = {"name": "detect_events", "arguments": {"well": "F-12"}}
SUBMIT_CALL = {"name": "submit_findings", "arguments": SUBMISSION}
STEPS = [
    {"index": 1, "content": "Plan.", "tool_calls": [EVENTS_CALL]},
    {"index": 2, "content": None, "tool_calls": [SUBMIT_CALL]},
]


async def test_the_replay_harness_reproduces_a_run_recorded_now() -> None:
    """Record a run with a scripted model on the fixture, then replay it: no differences."""
    replies = iter([_completion(step) for step in STEPS])

    async def create(**_):
        return next(replies)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    record = RunRecord("0123456789ab", "Why did F-12 fall?", None)
    run = await run_agent(
        record.question,
        LLM(client, "scripted", "scripted"),
        ProductionStore(FIXTURES / "processed"),
        on_event=record.emit,
        evidence_log=record.log,
    )
    record.complete(run)
    saved = json.loads(json.dumps(record.to_file()))

    assert run.verified
    assert differences(saved, await replay(saved, FIXTURES / "processed")) == []


async def test_a_run_with_a_second_opinion_replays_as_recorded() -> None:
    """The review is the model's answer too: the replay feeds the recorded one back."""
    from wellpilot.agent.review import Point, Review

    script = [*STEPS]
    # After the first accepted answer, the reviewer asks; the model submits the same again.
    script.append(
        next(s for s in STEPS if any(c["name"] == "submit_findings" for c in s["tool_calls"]))
    )
    replies = iter([_completion({**step, "index": i}) for i, step in enumerate(script, start=1)])

    async def create(**_):
        return next(replies)

    async def review_fn(*_):
        return Review(
            verdict="concerns", points=[Point(point="Shut-ins?", check="detect_events")]
        ), 700

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    record = RunRecord("0123456789ac", "Why did F-12 fall?", None)
    run = await run_agent(
        record.question,
        LLM(client, "scripted", "scripted"),
        ProductionStore(FIXTURES / "processed"),
        on_event=record.emit,
        evidence_log=record.log,
        review_fn=review_fn,
    )
    record.complete(run)
    saved = json.loads(json.dumps(record.to_file()))

    assert run.verified and run.review["outcome"] == "revised"
    assert record.tokens() == sum(s.prompt_tokens + s.completion_tokens for s in run.steps) + 700
    assert differences(saved, await replay(saved, FIXTURES / "processed")) == []


def test_a_result_that_only_gained_fields_replays_as_recorded() -> None:
    """A tool that adds fields (the log's perforations, its age) or rewords its note leaves
    a recorded run's view of the old fields, and the numbers it cited, as they were."""
    from wellpilot.agent.replay import _holds

    then = {"well": "F-12", "surveys": [{"date": "2010-09-22", "share": 0.325}], "note": "old"}
    now = {**then, "surveys": [{"date": "2010-09-22", "share": 0.325}], "note": "new", "age": 3}

    assert _holds(now, then)
    assert not _holds({**now, "surveys": [{"date": "2010-09-22", "share": 0.33}]}, then)
    assert not _holds({k: v for k, v in now.items() if k != "well"}, then)
