"""The paraphrase fuzzer with a scripted model: each rewording goes through the checker with
its own declarations, against the run's saved evidence; rejected ones are listed to read."""

import json
from pathlib import Path

from test_redteam import completion

from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.paraphrase import check_rewording, reword_run, summarize
from wellpilot.llm import LLM

RUN = json.loads(
    (Path(__file__).parent / "fixtures" / "retired_runs" / "69519dd0a995.json").read_text()
)
FIRST = RUN["result"]["findings"][0]


class Client:
    """Answers every passage with the same rewordings of finding 1."""

    def __init__(self, rewordings: list[dict]) -> None:
        self.rewordings = rewordings
        self.requests: list[dict] = []
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return completion(json.dumps({"rewordings": self.rewordings}))


def test_the_original_wording_is_accepted() -> None:
    log = EvidenceLog.restore(RUN["evidence"].values())
    numbers = FIRST["numbers"]
    assert check_rewording(RUN["question"], RUN["result"], log, 1, FIRST["claim"], numbers) == []


async def test_each_rewording_is_checked_with_its_own_declarations() -> None:
    same = {"text": FIRST["claim"], "numbers": FIRST["numbers"], "varied": "nothing"}
    typography = {
        "text": FIRST["claim"].replace("F-12", "F‑12").replace("1 551", "1 551"),
        "numbers": [{**n, "text": n["text"].replace(" ", " ")} for n in FIRST["numbers"]],
        "varied": "typography",
    }
    wrong = {
        "text": FIRST["claim"].replace("1 551", "1 815"),
        "numbers": [
            {**n, "text": "1 815", "value": 1815} if n["text"] == "1 551" else n
            for n in FIRST["numbers"]
        ],
        "varied": "a number (not true: the fuzzer's model is told to keep them)",
    }
    client = Client([same, typography, wrong])
    llm = LLM(client=client, model="muse-open", provider="fake")

    found, tokens = await reword_run(llm, RUN, per_passage=3)

    brief = json.loads(client.requests[0]["messages"][1]["content"])
    assert brief["passage"]["text"] == FIRST["claim"] and brief["passage"]["numbers"]
    assert client.requests[-1]["response_format"]["json_schema"]["name"] == "rewordings"
    assert len(client.requests) == len(RUN["result"]["findings"]) + 1  # and the summary
    assert tokens == 120 * len(client.requests)
    first = [r for r in found if r["index"] == 1]
    assert [r["accepted"] for r in first] == [True, True, False]
    assert any("1 815" in p for p in first[2]["problems"])
    report = summarize(found)
    assert {"passage": "finding 1", "text": wrong["text"]}.items() <= report["to_read"][0].items()
