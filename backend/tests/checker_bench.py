"""How often the checker lets a wrong number through, and how often it rejects a true one.

    uv run python tests/checker_bench.py [--cases N] [--show N]

False passes, measured without anyone writing the attacks: every accepted answer in the
corpus has each declared number in its claims swapped for another value of the result it
cites (at least 10% off, same order of size, written in the same style and declared exactly,
as a model that copied the wrong cell would). Each swap is false in its sentence; the share
the checker accepts is the false-pass rate an attacker gets from the cited result itself.
The ninth panel's judge measured 56% this way on the six featured runs.

False rejections: the same accepted answers reworded by the metamorphic transforms that
keep their meaning (tests/metamorphic.py); each rejection is a true answer sent back.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metamorphic import PRESERVING  # noqa: E402

from wellpilot.agent.checker import (  # noqa: E402
    _leaves,
    _standalone,
)
from wellpilot.agent.corpus import CORPUS_DIR, replay  # noqa: E402
from wellpilot.agent.evidence import EvidenceLog  # noqa: E402

SEED = 7


def accepted_cases() -> dict[str, dict]:
    return {
        p.stem: case
        for p in sorted(CORPUS_DIR.glob("*.json"))
        if (case := json.loads(p.read_text()))["expected"]["accepted"]
    }


def _style(text: str, value: float, new: float) -> str | None:
    """The new value written the way the old one is: its separator, decimals, percent."""
    body = text.strip()
    pct = body.endswith("%")
    digits = re.sub(r"[^\d.,\s]", "", body.rstrip("%")).strip()
    if pct and abs(value) <= 1.5:
        new *= 100
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if decimals:
        out = f"{new:.{decimals}f}"
    else:
        n = round(new)
        out = f"{n:,}" if "," in digits else f"{n:,}".replace(",", " ") if " " in digits else str(n)
    return out + ("%" if pct else "")


def swaps(case: dict, rnd: random.Random, per_number: int = 3):
    """(finding index, new submission, what changed) for each swap of a declared number."""
    log = EvidenceLog.restore(case["evidence"])
    sub = case["submission"]
    for i, finding in enumerate(sub["findings"]):
        claim = finding["claim"]
        numbers = finding.get("numbers") or []
        declared_values = {n["value"] for n in numbers}
        for n in numbers:
            text, value, eid = n["text"], n["value"], n["evidence_id"]
            item = log.get(eid)
            if item is None or not value or not _standalone(claim, text):
                continue
            pct = text.strip().endswith("%") and abs(value) <= 1.5
            pool = sorted(
                {
                    leaf.value
                    for leaf in _leaves(item)
                    if leaf.value is not None
                    and leaf.value > 0
                    and not (1900 <= leaf.value <= 2100 and leaf.value == int(leaf.value))
                    and (not pct or leaf.value <= 1)
                }
            )
            candidates = [
                v
                for v in pool
                if abs(v - value) / abs(value) >= 0.10
                and 0.4 <= v / value <= 2.5
                and v not in declared_values
            ]
            rnd.shuffle(candidates)
            for v in candidates[:per_number]:
                new_text = _style(text, value, v)
                if not new_text or new_text == text or re.fullmatch(r"\d%?", new_text):
                    continue
                start, end = _standalone(claim, text)[0]
                new_sub = json.loads(json.dumps(sub))
                f = new_sub["findings"][i]
                f["claim"] = claim[:start] + new_text + claim[end:]
                f["numbers"] = [
                    m if m is not n else {"text": new_text, "value": v, "evidence_id": eid}
                    for m in numbers
                ]
                yield i, new_sub, f"{text} -> {new_text} ({eid})"


def measure_swaps(cases: dict[str, dict], show: int) -> tuple[int, int, list]:
    rnd = random.Random(SEED)
    total = passed = 0
    passes = []
    for case_id, case in cases.items():
        for i, new_sub, what in swaps(case, rnd):
            problems = replay(case | {"submission": new_sub})
            total += 1
            if not any(p.startswith(f"Finding {i + 1}") for p in problems):
                passed += 1
                passes.append((case_id, i + 1, what, new_sub["findings"][i]["claim"]))
    for p in passes[:show]:
        print("  PASS", *p)
    return total, passed, passes


def measure_rewordings(cases: dict[str, dict], show: int) -> tuple[int, int]:
    total = rejected = 0
    shown = 0
    for case_id, case in cases.items():
        for name, transform in sorted(PRESERVING.items()):
            reworded = transform(case["submission"])
            if reworded is None:
                continue
            total += 1
            problems = replay(case | {"submission": reworded})
            if problems:
                rejected += 1
                if shown < show:
                    print("  REJECTED", case_id, name, problems[:2])
                    shown += 1
    return total, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=0, help="only the first N accepted cases")
    parser.add_argument("--show", type=int, default=0, help="print N examples of each kind")
    parser.add_argument("--json", help="write the passes to this file")
    args = parser.parse_args()
    cases = accepted_cases()
    if args.cases:
        cases = dict(list(cases.items())[: args.cases])
    total, passed, passes = measure_swaps(cases, args.show)
    print(f"swaps: {passed} of {total} accepted ({passed / max(total, 1):.0%} false passes)")
    kinds = Counter(p[2].split(" (")[0].split(" -> ")[0] for p in passes)
    print("  most passed numbers:", kinds.most_common(8))
    t, r = measure_rewordings(cases, args.show)
    print(f"rewordings: {r} of {t} rejected ({r / max(t, 1):.1%} false rejections)")
    if args.json:
        Path(args.json).write_text(json.dumps(passes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
