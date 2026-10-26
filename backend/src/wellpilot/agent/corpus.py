"""A corpus of real submissions for the claim checker, replayed in CI.

Every submit_findings call a real run made becomes a case: the submission, the evidence
that existed when it was made, and the verdict a human agrees with. Unit tests cover the
formats we thought of; the corpus covers what models actually write, so a checker change
that would wrongly reject (or wrongly pass) a real answer fails the build.

    uv run python -m wellpilot.agent.corpus add <run id or path>   # one case per submission
    uv run python -m wellpilot.agent.corpus check                   # replay all, show diffs

`add` records the checker's current verdict as expected. Read each case it prints: if the
checker was wrong (a false rejection like "4 050"), set "expected" to the right verdict
and say why in "note"; the case then fails until the checker is fixed.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from wellpilot.agent.checker import Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import SUBMIT
from wellpilot.config import get_settings

CORPUS_DIR = Path(__file__).resolve().parents[3] / "tests" / "corpus"
LABEL = re.compile(r"^(Finding \d+|Summary)\b")


def verdict(problems: list[str]) -> dict:
    """What a case pins down: accepted or not, and which parts were flagged. Wording of the
    problems may change freely."""
    flagged = sorted({m.group(1) for p in problems if (m := LABEL.match(p))})
    return {"accepted": not problems, "flagged": flagged}


def replay(case: dict) -> list[str]:
    log = EvidenceLog.restore(case["evidence"])
    return check_submission(Submission.model_validate(case["submission"]), log)


def submissions(run: dict) -> Iterator[tuple[dict, list[str]]]:
    """Each submission in a saved run, with the evidence ids that existed when it was made."""
    seen: list[str] = []
    for event in run["events"]:
        if event.get("type") != "step":
            continue
        for call in event["tool_calls"]:
            if call["name"] == SUBMIT:
                yield call["arguments"], list(seen)
            elif call.get("evidence_id"):
                seen.append(call["evidence_id"])


def model_of(run: dict) -> str:
    started = next((e for e in run["events"] if e.get("type") == "run_started"), {})
    return started.get("model", "unknown")


def add(source: str, out: Path) -> list[Path]:
    path = Path(source)
    if not path.exists():
        path = get_settings().runs_dir / f"{source}.json"
    run = json.loads(path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n, (arguments, ids) in enumerate(submissions(run), start=1):
        case = {
            "source": {
                "run": run["id"],
                "submission": n,
                "model": model_of(run),
                "question": run["question"],
            },
            "note": "",
            "expected": {},
            "submission": arguments,
            "evidence": [run["evidence"][i] for i in ids],
        }
        try:
            problems = replay(case)
        except ValidationError:
            print(f"skipped submission {n}: not a valid submission")
            continue
        case["expected"] = verdict(problems)
        target = out / f"{run['id']}-{n}.json"
        target.write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n")
        written.append(target)
        print(f"\n{target.name}: {'accepted' if not problems else 'rejected'}")
        for problem in problems:
            print(f"  - {problem}")
    return written


def check(corpus: Path) -> int:
    failures = 0
    for path in sorted(corpus.glob("*.json")):
        case = json.loads(path.read_text())
        problems = replay(case)
        if verdict(problems) != case["expected"]:
            failures += 1
            print(f"{path.name}: expected {case['expected']}, got {verdict(problems)}")
            for problem in problems:
                print(f"  - {problem}")
    print(f"{failures} of {len(list(corpus.glob('*.json')))} cases changed verdict")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Checker corpus of real submissions.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    add_parser = commands.add_parser(
        "add", parents=[common], help="add every submission of a saved run"
    )
    add_parser.add_argument("run", help="run id (from data/runs) or path to a run file")
    commands.add_parser(
        "check", parents=[common], help="replay every case against the current checker"
    )
    args = parser.parse_args()
    if args.command == "add":
        add(args.run, args.corpus)
    else:
        sys.exit(1 if check(args.corpus) else 0)


if __name__ == "__main__":
    main()
