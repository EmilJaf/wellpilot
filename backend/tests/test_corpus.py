"""Replay every real submission in tests/corpus through the current checker.

A case fails when the checker's verdict on a real answer changes: accepted or not, and
which findings it flags. See wellpilot.agent.corpus for adding cases.
"""

import json
from pathlib import Path

import pytest

from wellpilot.agent.corpus import CORPUS_DIR, add, add_new, replay, verdict

# Real-data regression suite: CI runs it; locally `uv run pytest -m "slow or not slow"`.
pytestmark = pytest.mark.slow

CASES = sorted(CORPUS_DIR.glob("*.json"))


def test_the_corpus_is_not_empty() -> None:
    assert CASES, f"no cases in {CORPUS_DIR}"


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_real_submission_keeps_its_verdict(path: Path) -> None:
    case = json.loads(path.read_text())

    problems = replay(case)

    assert verdict(problems) == case["expected"], (
        f"{case['note'] or 'no note'}\nproblems now:\n" + "\n".join(problems)
    )


def _run_file(directory: Path, run_id: str) -> Path:
    def submission(rate: str) -> dict:
        claim = f"Rate fell to {rate} Sm3/d."
        number = {"text": rate, "value": float(rate), "evidence_id": "E1"}
        return {
            "summary": "Rate fell.",
            "summary_numbers": [],
            "findings": [
                {"claim": claim, "evidence_ids": ["E1"], "confidence": "fact", "numbers": [number]}
            ],
        }

    run = {
        "id": run_id,
        "question": "Why?",
        "events": [
            {"type": "run_started", "model": "test-model"},
            {
                "type": "step",
                "tool_calls": [
                    {"name": "production_monthly", "evidence_id": "E1"},
                    {"name": "submit_findings", "arguments": submission("1500")},
                    {"name": "submit_findings", "arguments": submission("1146")},
                ],
            },
        ],
        "evidence": {
            "E1": {"id": "E1", "tool": "production_monthly", "arguments": {}, "result": [1146]}
        },
    }
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(run))
    return path


def test_rejections_are_queued_for_review_and_runs_are_added_once(tmp_path: Path) -> None:
    runs, corpus = tmp_path / "runs", tmp_path / "corpus"
    runs.mkdir()
    _run_file(runs, "0123456789ab")

    first = add_new(runs, corpus)
    again = add_new(runs, corpus)

    cases = [json.loads(p.read_text()) for p in first]
    assert [(c["expected"]["accepted"], c["to_review"]) for c in cases] == [
        (False, True),
        (True, False),
    ]
    assert cases[1]["source"]["model"] == "test-model"
    assert again == []
    assert len(add(runs / "0123456789ab.json", corpus)) == 2  # an explicit add redoes a run


def test_the_corpus_exports_as_checker_labelled_training_data(tmp_path) -> None:
    """Within a run, the answer sent back and the one accepted after it form a pair; the
    reviewed verdict is the label, so a corrected false rejection is not taught as wrong."""
    import json
    import shutil

    from wellpilot.agent.corpus import CORPUS_DIR, export

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("e8c3374117a8-1.json", "e8c3374117a8-2.json"):
        shutil.copy(CORPUS_DIR / name, corpus / name)

    counts = export(corpus, tmp_path / "training")

    assert counts["preference_pairs"] == 1 and counts["accepted_answers"] == 1
    (pair,) = [json.loads(line) for line in (tmp_path / "training" / "preferences.jsonl").open()]
    assert pair["checker_said"] and pair["rejected"] != pair["chosen"]
    assert pair["question"].startswith("F-12's water cut fell")
