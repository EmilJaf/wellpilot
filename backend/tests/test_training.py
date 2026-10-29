"""The writer training data and its eval: held-out questions stay out of training, and the
scorer accepts the gold answers and rejects a broken one, with no model call."""

import json

import pytest

from wellpilot.agent.checker import Submission
from wellpilot.training import HOLDOUT, build, parse, score

# Reads the whole corpus: CI runs it with the other real-data suites.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    out = tmp_path_factory.mktemp("writer")
    manifest = build(out=out)
    rows = {
        name: [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "test")
    }
    return manifest, rows


def test_no_held_out_question_is_trained_on(split) -> None:
    manifest, rows = split
    assert manifest["train_examples"] > 50 and manifest["test_runs"] > 15
    for example in rows["train"]:
        question = example["messages"][1]["content"].split("\n", 1)[0].removeprefix("Question: ")
        assert not any(question.startswith(h) for h in HOLDOUT), question
    assert all(any(t["question"].startswith(h) for h in HOLDOUT) for t in rows["test"])


def test_every_training_answer_is_a_submission(split) -> None:
    _, rows = split
    for example in rows["train"]:
        roles = [m["role"] for m in example["messages"]]
        assert roles == ["system", "user", "assistant"]
        Submission(**json.loads(example["messages"][2]["content"]))


def test_the_scorer_accepts_gold_and_sends_a_wrong_number_back(split) -> None:
    _, rows = split
    item = next(t for t in rows["test"] if t["gold"])
    assert score(item, item["gold"])["accepted"]
    broken = json.loads(json.dumps(item["gold"]))
    finding = next(f for f in broken["findings"] if f.get("numbers"))
    number = next(n for n in finding["numbers"] if n["text"] in finding["claim"])
    finding["claim"] = finding["claim"].replace(number["text"], "98765", 1)
    number.update(text="98765", value=98765)
    result = score(item, broken)
    assert not result["accepted"] and result["kinds"]


def test_a_reply_is_read_through_fences_and_thinking() -> None:
    assert parse('<think>plan</think>\n```json\n{"summary": "x", "findings": []}\n```') == {
        "summary": "x",
        "findings": [],
    }
    assert parse("no json here") is None
