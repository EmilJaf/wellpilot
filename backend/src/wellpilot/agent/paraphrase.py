"""True rewordings of accepted answers, to find what the checker rejects wrongly.

The mirror of the red team (redteam.py): there the model writes false versions of a finding
and the checker must catch them; here it writes TRUE versions, the same facts in other words,
and the checker must accept them. The checker reads English by rules, and the model's
phrasing varies run to run ("to over 50%", "when water was 30%", "at 21 months", dates with
non-breaking hyphens); each new phrasing a rule misreads used to surface only in a live run,
as a false rejection that could cost the run its verification. Run this before re-recording
demo runs, and after any change to the checker.

Each rewording carries its own declarations ({text, value, evidence_id}, as submit_findings
does) and goes through the checker against the run's saved evidence, with the finding (or
the summary) replaced; only its own problems count. A rejected rewording is not proof of a
checker bug: the model may have changed the meaning. Those are the ones to read.

    uv run python -m wellpilot.agent.paraphrase [run ...] [--per-finding 4]   # spends tokens
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.llm import LLM, json_schema

TIMEOUT_S = 120.0
MAX_REWORDINGS = 6
EVIDENCE_CHARS = 6000  # of each cited result shown to the model
REPORTS_DIR = "paraphrase"  # under data/
FROZEN_RUNS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "runs"

SYSTEM = """\
You help test a fact checker that guards an AI agent's answers about an oil field. You get \
one passage the checker accepted (a finding or a summary), its declared numbers and the \
tool results it cites. Rewrite it in different words, the way different engineers would \
write the same thing, keeping it TRUE: the same facts, the same numbers at the same \
precision, the same wells, dates and directions, the same hedges, no cause stated more \
firmly than the original. Vary as much else as you can: sentence order and structure, \
active or passive, synonyms ("rose" / "climbed" / "went up"), how a level is introduced \
("water cut was 0.97", "at a 97% water cut", "with water at 97%"), "from X to Y" or "X \
before and Y after", dates as YYYY-MM-DD or in words ("18 July 2010"), thousands as 4,050 \
or 4 050, a unit before or after a number, a number's unit inside or outside its \
declaration. Keep any quotation from a report word for word. Declare every number you \
write as {"text": exactly as written, "value": its value, "evidence_id": where it comes \
from}, as the original does (dates, years, well names and integers up to 10 need none). \
Write the rewording as plain prose, the numbers as a reader sees them; the declarations \
go only in the "numbers" list, never inside the text. For each rewording, say in a few \
words what you varied."""

# Declarations the model wrote inside the text ("{55%|0.5512|E5}") instead of in the list.
INLINE = re.compile(r"\{([^|{}]+)\|\s*(-?[\d.eE+-]+)\s*\|\s*(E\d+)\s*\}")


# Other annotations the model put after a number: "{text: 32%, value: 0.325, ...}",
# "[0.6715 E3]", "{false}", "{0}"; and around one: "{42,800 (value 42798.0, ...)}",
# "{32 days:32.0:E3:a}", "declared 97% as {0}".
AFTER = re.compile(r"\s*(?:\{text:[^{}]*\}|\[-?[\d.eE+]+\s+E\d+\]|\{(?:false|true|\d+)\})")
AROUND = re.compile(
    r"\{([^{}():]+?)\s*\((?:value|declared)[^)]*\)\}|\{([^{}:|]+):[^{}]*\}|declared (\S+) as"
)


def unwrap(text: str, numbers: list[dict]) -> tuple[str, list[dict]]:
    """Text with the model's inline declarations and annotations taken out; declarations
    written inline join the list (kept once each)."""
    found = [
        {"text": m.group(1).strip(), "value": float(m.group(2)), "evidence_id": m.group(3)}
        for m in INLINE.finditer(text)
    ]
    texts = {n["text"] for n in numbers}
    text = AFTER.sub("", INLINE.sub(lambda m: m.group(1).strip(), text))
    text = AROUND.sub(lambda m: next(g for g in m.groups() if g).strip(), text)
    return text, numbers + [n for n in found if n["text"] not in texts]


class Rewording(BaseModel):
    text: str = Field(max_length=3000)
    numbers: list[DeclaredNumber]
    varied: str = Field(description="What was varied, a few words.")


class Rewordings(BaseModel):
    rewordings: list[Rewording] = Field(max_length=MAX_REWORDINGS)


def _brief(question: str, passage: dict, cited: list[dict], count: int) -> str:
    results = {
        e["id"]: {
            "tool": e["tool"],
            "arguments": e["arguments"],
            "result": json.dumps(e["result"], ensure_ascii=False, default=str)[:EVIDENCE_CHARS],
        }
        for e in cited
    }
    return json.dumps(
        {"question": question, "passage": passage, "cited_results": results, "write": count},
        ensure_ascii=False,
        indent=1,
    )


async def reword(
    llm: LLM, question: str, passage: dict, cited: list[dict], count: int = 4
) -> tuple[list[Rewording], int]:
    """The model's rewordings of one passage, and the tokens they cost."""
    response = await asyncio.wait_for(
        llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _brief(question, passage, cited, count)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rewordings", "schema": json_schema(Rewordings)},
            },
            reasoning_effort="low",
        ),
        TIMEOUT_S,
    )
    found = Rewordings.model_validate_json(response.choices[0].message.content or "").rewordings
    usage = response.usage
    return found[:count], (usage.prompt_tokens + usage.completion_tokens) if usage else 0


def check_rewording(
    question: str, result: dict, log: EvidenceLog, index: int, text: str, numbers: list
) -> list[str]:
    """The checker's problems with one rewording: finding `index` (from 1) or, for 0, the
    summary replaced, with the rewording's own declarations; only its own problems."""
    findings = [Finding.model_validate(f) for f in result["findings"]]
    summary = result.get("summary") or ""
    summary_numbers = [
        DeclaredNumber.model_validate(n) for n in result.get("summary_numbers") or []
    ]
    declared = [DeclaredNumber.model_validate(n) for n in numbers]
    if index:
        findings[index - 1] = findings[index - 1].model_copy(
            update={"claim": text, "numbers": declared}
        )
        label = f"Finding {index}"
    else:
        summary, summary_numbers, label = text, declared, "Summary"
    submission = Submission(summary=summary, findings=findings, summary_numbers=summary_numbers)
    return [p for p in check_submission(submission, log, question) if p.startswith(label)]


def _passages(result: dict, evidence: dict) -> list[tuple[int, dict, list[dict]]]:
    """(index, passage, cited results): each finding's claim, then the summary (index 0)."""
    out = []
    for n, f in enumerate(result["findings"], start=1):
        cited = [evidence[i] for i in f["evidence_ids"] if i in evidence]
        passage = {
            "kind": "finding",
            "text": f["claim"],
            "confidence": f["confidence"],
            "evidence_ids": f["evidence_ids"],
            "numbers": f.get("numbers") or [],
        }
        out.append((n, passage, cited))
    numbers = result.get("summary_numbers") or []
    ids = list(dict.fromkeys(n["evidence_id"] for n in numbers))
    passage = {"kind": "summary", "text": result.get("summary") or "", "numbers": numbers}
    out.append((0, passage, [evidence[i] for i in ids if i in evidence]))
    return out


async def reword_run(llm: LLM, record: dict, per_passage: int = 4) -> tuple[list[dict], int]:
    """Rewordings of every finding and the summary of one saved run, each checked."""
    result, evidence = record["result"], record["evidence"]
    log = EvidenceLog.restore(evidence.values())
    out, tokens = [], 0
    for index, passage, cited in _passages(result, evidence):
        try:
            found, used = await reword(llm, record["question"], passage, cited, per_passage)
        except Exception as e:  # one passage the model could not reword; go on
            print(f"  passage {index}: no rewordings ({type(e).__name__})", file=sys.stderr)
            continue
        tokens += used
        for r in found:
            text, numbers = unwrap(r.text, [n.model_dump() for n in r.numbers])
            problems = check_rewording(record["question"], result, log, index, text, numbers)
            out.append(
                {
                    "passage": "summary" if index == 0 else f"finding {index}",
                    "index": index,
                    "original": passage["text"],
                    "text": text,
                    "numbers": numbers,
                    "varied": r.varied,
                    "accepted": not problems,
                    "problems": problems,
                }
            )
    return out, tokens


def summarize(rewordings: list[dict]) -> dict:
    return {
        "rewordings": len(rewordings),
        "accepted": sum(r["accepted"] for r in rewordings),
        "to_read": [
            {
                "run": r.get("run"),
                "passage": r["passage"],
                "text": r["text"],
                "problems": r["problems"],
            }
            for r in rewordings
            if not r["accepted"]
        ],
    }


def main() -> None:
    from wellpilot.config import get_settings
    from wellpilot.llm import make_llm

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", help="run ids (default: the frozen featured runs)")
    parser.add_argument("--per-finding", type=int, default=4)
    args = parser.parse_args()
    settings = get_settings()
    runs = args.runs or sorted(p.stem for p in FROZEN_RUNS.glob("*.json"))
    llm = make_llm(settings)
    at = datetime.now(UTC).isoformat(timespec="seconds")
    out = settings.data_dir / REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{at.replace(':', '')}.json"
    rewordings: list[dict] = []
    tokens = 0

    def save() -> None:
        report = {
            "at": at,
            "model": llm.model,
            "runs": runs,
            "tokens": tokens,
            "summary": summarize(rewordings),
            "rewordings": rewordings,
        }
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    async def reword_all() -> None:
        nonlocal tokens
        for run_id in runs:
            source = settings.runs_dir / f"{run_id}.json"
            record = json.loads(
                (source if source.exists() else FROZEN_RUNS / source.name).read_text()
            )
            found, used = await reword_run(llm, record, args.per_finding)
            tokens += used
            rewordings.extend({"run": run_id, **r} for r in found)
            save()
            print(f"{run_id}: {sum(r['accepted'] for r in found)} of {len(found)} accepted")

    asyncio.run(reword_all())
    report = summarize(rewordings)
    print(f"{report['accepted']} of {report['rewordings']} accepted; {tokens} tokens; {path}")


if __name__ == "__main__":
    main()
