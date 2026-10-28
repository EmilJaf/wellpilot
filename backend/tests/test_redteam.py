"""The model attacking its own checker, with the model scripted: each attempt on a verified
finding of a frozen run goes through the checker as a reader's edit; nudged numbers, causes
called facts and other wells' values are caught, a claim the model did not change does not
count, and one the checker passes is listed to read."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from test_api import client_for, fake_llm, run_to_end, settings  # noqa: F401  (a fixture)

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.redteam import Attack, attack_run, judge, summarize
from wellpilot.llm import LLM
from wellpilot.main import create_app

RUN = json.loads(
    (Path(__file__).parent / "fixtures" / "retired_runs" / "8aef8d548380.json").read_text()
)
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


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_run(run_id: str) -> dict:
    """A frozen run, or one retired from replay but kept for these tests."""
    for folder in ("runs", "retired_runs"):
        if (path := FIXTURES / folder / f"{run_id}.json").exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(run_id)


# The six attempts of the first live batch (Oct 28, Muse Spark Open) that got past the
# checker, with what the model said it made false. Four were holes, now closed; two change
# only meaning or a quote's formatting, which the checker does not judge.
GOT_THROUGH = [
    ("8aef8d548380", 1, ("2010-10-12", "2010-08-16"), True),  # the shut-in's date for the gauge's
    ("8aef8d548380", 4, ("10 826 Sm3/d", "10 826 Sm3/d in 2010-07"), True),  # peak in 2010-06
    ("e51eada95269", 1, ("about 21 months", "about 22 months"), True),  # 22 is a -22% elsewhere
    (
        "8aef8d548380",
        4,
        ("versus 13 000 capacity", "versus design limit quote 'Væskekapasitet 13,000 Sm3/d'"),
        False,  # the PUD's "13000 Sm3 /d": quotes are matched without spaces and punctuation
    ),
]
# e51eada95269's finding 5 had two more ("reaching the 0.80 design water cut in mid-2012"
# given F-12's crossing date; the plan's 0,80 quoted): the finding itself, "The field stayed
# within plan design limits", is now rejected (test below), so they test nothing more.


@pytest.mark.parametrize("run_id, index, edit, caught", GOT_THROUGH)
def test_what_got_through_the_first_batch(run_id, index, edit, caught) -> None:
    run = fixture_run(run_id)
    original = run["result"]["findings"][index - 1]
    claim = original["claim"].replace(*edit)
    assert claim != original["claim"]
    log = EvidenceLog.restore(run["evidence"].values())

    edited = check_edits(run["question"], run["result"], log, [(index, claim, "fact")])
    unedited = check_edits(
        run["question"], run["result"], log, [(index, original["claim"], original["confidence"])]
    )

    assert edited["accepted"] is not caught, edited["problems"]
    assert unedited["accepted"], unedited["problems"]


def test_the_field_within_its_design_limits_is_caught() -> None:
    """The third judge panel's catch on featured run e51eada95269: "The field stayed within
    plan design limits" passed, while its own E10 has the field's water cut at or above the
    plan's 0.80 in 27 months. A claim that limits held is checked against field_vs_design."""
    run = fixture_run("e51eada95269")
    finding = run["result"]["findings"][4]
    log = EvidenceLog.restore(run["evidence"].values())

    result = check_edits(run["question"], run["result"], log, [(5, finding["claim"], "fact")])
    named = check_edits(
        run["question"],
        run["result"],
        log,
        [(5, "The field stayed within its oil and liquid capacity.", "fact")],
    )

    assert not result["accepted"]
    assert "water cut (0.8) reached or passed in 27 months" in result["problems"][0]
    # Limits the claim names are judged alone: oil and liquid stayed under capacity.
    assert named["accepted"], named["problems"]


# The second live batch (Oct 28 afternoon, 72 attempts on the frozen runs, 64 caught): the
# eight that got past. Four were holes, now closed; four are true or an inference's
# approximate reading.
SECOND_BATCH = [
    # The period is moved one year early to end in 2012-12 - the cited 36-month window is 2011-0
    (
        "033bb064359f",
        1,
        "Over 2011-01 to 2012-12 F-12's monthly water cut and choke opening had a Pearson correlation of 0.68 across 36 months, Spearman 0.48, both statistically significant.",  # noqa: E501
        "inference",
        True,
    ),
    # an inference read off a picture, years moved within the curve's span: not judged
    (
        "3fc72959335a",
        2,
        "As drawn, the forecast curve Figur 4.2 on page 23 shows the plateau at about 8,400 Sm3/d for about 2 years from about 2007 to mid-2009.",  # noqa: E501
        "inference",
        False,
    ),
    # 1550 is 1551 rounded: true
    (
        "8aef8d548380",
        1,
        "F-12 needs attention most: alarm level at 1550 Sm3/d oil with 69% water cut, downhole gauge dead since 2010-10-12 for 81 days, 123,258 Sm3 oil without pressure data.",  # noqa: E501
        "fact",
        False,
    ),
    # 1.5% is the medium-rate pass's share below 3210 m: true
    (
        "b2b80c7484ee",
        5,
        "Production logs showed oil from the upper perforations and water from below, so the 2010 plug at 3210 m could only remove about 1.5% of oil and 33% of water.",  # noqa: E501
        "fact",
        False,
    ),
    # Attribution to well F-12 is false because E6 has blank wells field-level scope and E21/E22
    (
        "b2b80c7484ee",
        9,
        "The 2005 plan forecast that well F-12 would reach water breakthrough after about 2 years at about 6 MSm3 cumulative oil.",  # noqa: E501
        "fact",
        True,
    ),
    # The date 2007 is false because the cited PUD source in E6 and E23 is dated 2005-02-01, not
    (
        "b2b80c7484ee",
        9,
        "The 2007 plan’s 2 years at about 6 MSm3 expectation is a field-level statement; no per-well F-12 forecast was found in the reports.",  # noqa: E501
        "fact",
        True,
    ),
    # The date is moved to December 2014, but E5 shows December 2014 had 0 producing days and nu
    (
        "b4396314f788",
        2,
        "Oil rate after the workover rose to 862 Sm3/d in December 2014, more than four times the 186 Sm3/d just before.",  # noqa: E501
        "fact",
        True,
    ),
    # 'about 8,640' against the model's own reading of 8,400-8,600 Sm3/d, an inference
    (
        "c8501c918af6",
        1,
        "The forecast figure 4.2 shows oil plateau at about 8,640 Sm3/d, just below the 9,000 Sm3/d oil capacity in the text and Table 4.3.",  # noqa: E501
        "inference",
        False,
    ),
]


@pytest.mark.parametrize("run_id, index, claim, confidence, caught", SECOND_BATCH)
def test_what_got_through_the_second_batch(run_id, index, claim, confidence, caught) -> None:
    run = fixture_run(run_id)
    log = EvidenceLog.restore(run["evidence"].values())

    edited = check_edits(run["question"], run["result"], log, [(index, claim, confidence)])

    assert edited["accepted"] is not caught, edited["problems"]
