"""A corpus of real submissions for the claim checker, replayed in CI.

Every submit_findings call a real run made becomes a case: the submission, the evidence
that existed when it was made, and the expected verdict. Unit tests cover the formats we
thought of; the corpus covers what models actually write, so a checker change that would
wrongly reject (or wrongly pass) a real answer fails the build.

The agent CLI adds each run it makes. For runs made in the web app:

    uv run python -m wellpilot.agent.corpus add            # every run not in the corpus yet
    uv run python -m wellpilot.agent.corpus add <run id>   # one run (again)
    uv run python -m wellpilot.agent.corpus check          # replay all; list cases to review

A new case expects the verdict the checker gives now. A rejection is marked "to_review":
someone who understands the answer decides whether the checker was right. If it was
wrong (a false rejection like "4 050"), set "expected" to the right verdict, say why in
"note", and fix the checker until the case passes. Then set "to_review" to false.
Acceptances are not queued: a false pass is found by reviewing the rules, not each case.
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


def add(path: Path, out: Path = CORPUS_DIR) -> list[Path]:
    """Add every submission of the saved run at path; return the case files written."""
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
            "to_review": False,
            "note": "",
            "expected": {},
            "submission": arguments,
            "evidence": [run["evidence"][i] for i in ids],
        }
        try:
            problems = replay(case)
        except ValidationError:
            continue  # the model sent an invalid submission; nothing to check
        case["expected"] = verdict(problems)
        case["to_review"] = bool(problems)
        target = out / f"{run['id']}-{n}.json"
        target.write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n")
        written.append(target)
    return written


def add_new(runs_dir: Path, out: Path = CORPUS_DIR) -> list[Path]:
    """Add every saved run that has no case in the corpus yet."""
    known = {p.stem.rsplit("-", 1)[0] for p in out.glob("*.json")}
    written = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.stem not in known:
            written += add(path, out)
    return written


def describe(paths: list[Path]) -> None:
    for path in paths:
        case = json.loads(path.read_text())
        state = "accepted" if case["expected"]["accepted"] else "rejected, to review"
        print(f"corpus: {path.name} {state}")


def check(corpus: Path = CORPUS_DIR) -> int:
    """Replay every case; print changed verdicts and the cases still to review."""
    changed, to_review = 0, []
    cases = sorted(corpus.glob("*.json"))
    for path in cases:
        case = json.loads(path.read_text())
        problems = replay(case)
        if verdict(problems) != case["expected"]:
            changed += 1
            print(f"CHANGED {path.name}: expected {case['expected']}, got {verdict(problems)}")
            for problem in problems:
                print(f"  - {problem}")
        elif case.get("to_review"):
            to_review.append((path, problems))
    for path, problems in to_review:
        print(f"TO REVIEW {path.name}:")
        for problem in problems:
            print(f"  - {problem}")
    print(f"{len(cases)} cases: {changed} changed verdict, {len(to_review)} to review")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Checker corpus of real submissions.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    add_parser = commands.add_parser(
        "add", parents=[common], help="add saved runs (default: every run not added yet)"
    )
    add_parser.add_argument("run", nargs="?", help="run id (from data/runs) or path to a file")
    commands.add_parser("check", parents=[common], help="replay every case; list cases to review")
    args = parser.parse_args()
    if args.command == "check":
        sys.exit(1 if check(args.corpus) else 0)
    runs_dir = get_settings().runs_dir
    if args.run is None:
        describe(add_new(runs_dir, args.corpus))
        return
    path = Path(args.run)
    describe(add(path if path.exists() else runs_dir / f"{args.run}.json", args.corpus))


if __name__ == "__main__":
    main()
