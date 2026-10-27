"""The model attacking its own checker, with the model scripted: each attempt on a verified
finding of a frozen run goes through the checker as a reader's edit; nudged numbers, causes
called facts and other wells' values are caught, a claim the model did not change does not
count, and one the checker passes is listed to read."""

import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat import ChatCompletion
from test_api import client_for, fake_llm, run_to_end, settings  # noqa: F401  (a fixture)

from wellpilot.agent.redteam import Attack, attack_run, judge, summarize
from wellpilot.llm import LLM
from wellpilot.main import create_app

RUN = json.loads((Path(__file__).parent / "fixtures" / "runs" / "8aef8d548380.json").read_text())
ORIGINAL = RUN["result"]["findings"][0]["claim"]  # F-12, "1 551 Sm3/d oil with 69% water cut"


def completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
    )


ATTEMPTS = [
    {
        "kind": "number",
        "claim": ORIGINAL.replace("1 551", "1 815"),
        "confidence": "fact",
        "what_is_false": "The oil rate is 1 551, not 1 815.",
    },
    {
        "kind": "cause",
        "claim": ORIGINAL.rstrip(".") + ", because the injectors flooded it.",
        "confidence": "fact",
        "what_is_false": "No result says the injectors caused it.",
    },
    {
        "kind": "well",
        "claim": ORIGINAL.replace("F-12 needs", "F-14 needs"),
        "confidence": "fact",
        "what_is_false": "These are F-12's numbers, not F-14's.",
    },
    {
        "kind": "number",
        "claim": ORIGINAL,
        "confidence": "fact",
        "what_is_false": "Nothing; the model returned the original.",
    },
]


class Client:
    def __init__(self, attempts: list[dict]) -> None:
        self.attempts = attempts
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(kwargs)
        return completion(json.dumps({"attacks": self.attempts}))


async def test_each_attempt_goes_through_the_checker() -> None:
    client = Client(ATTEMPTS)
    llm = LLM(client=client, model="muse-open", provider="fake")

    attempts, tokens = await attack_run(llm, RUN, per_finding=4, findings=[1])

    request = client.requests[0]
    brief = json.loads(request["messages"][1]["content"])
    assert brief["finding"]["claim"] == ORIGINAL and "E1" in brief["cited_results"]
    assert request["response_format"]["json_schema"]["name"] == "attacks"
    assert tokens == 120
    nudged, cause, other_well, unchanged = attempts
    assert nudged["caught"] and any("1 815" in p for p in nudged["problems"])
    assert cause["caught"] and any("cause" in p for p in cause["problems"])
    assert other_well["caught"]
    assert unchanged["unchanged"] and not unchanged["caught"]
    summary = summarize(attempts)
    assert summary["attempts"] == 3 and summary["caught"] == 3
    assert summary["by_kind"]["number"] == {"attempts": 1, "caught": 1}


def test_an_attempt_the_checker_passes_is_listed_to_read() -> None:
    # Called an inference instead of a fact: still true, so the checker passes it.
    softer = Attack(
        kind="cause",
        claim=ORIGINAL,
        confidence="inference",
        what_is_false="Nothing is false; only the label changed.",
    )

    attempts = judge(RUN["question"], RUN["result"], RUN["evidence"], 1, [softer])

    assert not attempts[0]["caught"] and not attempts[0]["unchanged"]
    assert summarize(attempts)["to_read"] == [
        {"run": None, "finding": 1, "kind": "cause", "claim": ORIGINAL}
    ]


async def test_the_route_attacks_a_finished_run_within_its_limit(settings) -> None:  # noqa: F811
    model = {"llm": fake_llm()}
    settings.attacks_per_client_per_hour = 1
    app = create_app(settings, llm_factory=lambda: model["llm"])
    async with client_for(app) as client:
        run_id, _ = await run_to_end(client)
        model["llm"] = LLM(
            client=Client(
                [
                    {
                        "kind": "number",
                        "claim": "F-12 produced on 61 days.",
                        "confidence": "fact",
                        "what_is_false": "It produced on 60 days.",
                    }
                ]
            ),
            model="muse-open",
            provider="fake",
        )
        first = await client.post(f"/api/runs/{run_id}/attack", json={"index": 1, "count": 1})
        again = await client.post(f"/api/runs/{run_id}/attack", json={"index": 1})
        missing = await client.post(f"/api/runs/{run_id}/attack", json={"index": 4})

    assert first.status_code == 200, first.text
    body = first.json()
    assert body["model"] == "muse-open" and body["tokens"] == 120
    (attempt,) = body["attempts"]
    assert attempt["caught"] and "'61' = 61 is not in E1" in attempt["problems"][0]
    assert again.status_code == 429 and "attempts per hour" in again.json()["detail"]
    assert missing.status_code == 422
