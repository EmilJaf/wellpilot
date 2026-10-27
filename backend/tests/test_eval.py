"""The agent eval's scoring, driven by a scripted model on the real-data fixture (no tokens)."""

import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import SUBMIT
from wellpilot.eval import CASES, Case, run_case
from wellpilot.llm import LLM

FIXTURE = Path(__file__).parent / "fixtures" / "processed"


def reply(tool_calls) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                            for i, (name, args) in enumerate(tool_calls)
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10_000, "completion_tokens": 1_000, "total_tokens": 11_000},
        }
    )


ANSWER = {
    "summary": "F-12's oil rate stepped down around 2010-07-18.",
    "summary_numbers": [],
    "findings": [
        {
            "claim": "F-12's oil rate fell from 4 050 to 1 146 Sm3/d around 2010-07-18.",
            "evidence_ids": ["E1"],
            "confidence": "fact",
            "numbers": [
                {"text": "4 050", "value": 4050.37, "evidence_id": "E1"},
                {"text": "1 146", "value": 1145.72, "evidence_id": "E1"},
            ],
        }
    ],
}


def scripted() -> LLM:
    replies = iter([reply([("detect_events", {"well": "F-12"})]), reply([(SUBMIT, ANSWER)])])

    async def create(**_):
        return next(replies)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLM(client=client, model="scripted", provider="scripted")


async def test_a_case_passes_when_every_check_does(tmp_path) -> None:
    case = Case("step", "Why did F-12 fall?", tools=("detect_events",), mentions=(r"2010-07-18",))

    result = await run_case(case, scripted(), tmp_path, FIXTURE, add_to_corpus=False)

    assert result["passed"], result["checks"]
    assert result["steps"] == 2 and result["cost"] == 0.0335  # 20k in + 2k out
    assert (tmp_path / f"{result['run']}.json").exists()  # saved, so it can be replayed


async def test_each_failed_check_is_named(tmp_path) -> None:
    case = Case(
        "strict",
        "Why did F-12 fall?",
        tools=("look_at_page",),
        mentions=(r"2012-06",),
        max_steps=1,
        max_cost=0.01,
    )

    result = await run_case(case, scripted(), tmp_path, FIXTURE, add_to_corpus=False)

    failed = {c["check"] for c in result["checks"] if not c["passed"]}
    assert failed == {"calls look_at_page", "says /2012-06/", "<= 1 steps", "<= $0.01"}


def test_the_cases_cover_every_promise() -> None:
    tools = {t for c in CASES for t in (*c.tools, *c.any_tools)}
    assert {"look_at_page", "field_vs_design", "forecast_production", "detect_events"} <= tools
    assert any(c.as_of for c in CASES)  # one forecast made with data cut off
    assert len({c.id for c in CASES}) == len(CASES)


def test_summary_numbers_are_read_as_a_reader_meets_them() -> None:
    from wellpilot.eval import unreadable_numbers

    summary = (
        "F-12 fell from 4,050 to 1,146 Sm3/d on 2010-07-18 (-72%), declining 0.4461 per "
        'year after 1,133,053 Sm3; the plan said "Platåraten på 9.000 Sm3 /d" in 2005, '
        "and 0.45 per year, 1.13 million Sm3 or 10.04 MSm3 read fine."
    )

    assert unreadable_numbers(summary) == ["0.4461", "1,133,053"]


def test_dense_claims_are_flagged() -> None:
    from wellpilot.eval import dense_claims

    judged = (  # the time machine's first answer, as the judges read it
        "F-12 (attention 3, highest priority) produced 1 551 Sm3/d oil at 0.69 water cut on "
        "2010-12-31 with 16.5% decline over the last 90 days; its oil rate stepped down 55% "
        "from 4 207 to 1 888 Sm3/d around 2010-05-02."
    )
    stats = "The fit after 2010-07-18 is exponential with r2_log 0.87."
    plain = "F-12 needs a look first: its downhole gauge has been dead since 2010-10-12."

    assert dense_claims([judged, stats, plain]) == [
        "finding 1 (7 numbers, jargon)",
        "finding 2 (1 numbers, jargon)",
    ]
