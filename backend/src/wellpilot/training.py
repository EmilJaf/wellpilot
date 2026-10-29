"""The checker as a teacher: training data for an open-weight model, and a test of whether
the training worked.

    uv run python -m wellpilot.training build               # free: data/training/writer/
    uv run python -m wellpilot.training eval --fake gold     # free: the harness scores itself
    LLM_PROVIDER=fireworks FIREWORKS_MODEL=<model or deployment> \\
        uv run python -m wellpilot.training eval --label tuned   # spends tokens

The task is the agent's last step: given the question and every tool result its run
gathered, write the submit_findings answer. It needs no tool calling, so any chat model can
be compared on the same inputs. Training examples are the corpus's accepted answers; the
test holds out whole questions (never an answer to a question seen in training) and scores
each model's first answer with the checker itself, by the kinds of problem it finds, with
the eval's own regexes for that question where it has them. Passing the checker is what
the model is taught, so the regexes and a read of the answers are the check on that.
"""

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from wellpilot.agent.checker import Submission, check_submission, readability
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import SYSTEM_PROMPT, submit_tool_schema
from wellpilot.agent.vision import for_the_model
from wellpilot.config import get_settings
from wellpilot.eval import CASES_BY_ID, PRICE_IN, PRICE_OUT, dense_claims, unreadable_numbers

# Whole questions the model never sees in training: a sensor question, a design limit, a
# restart, a refusal (F-99 does not exist), the end-2010 screen and a synthetic field's well.
HOLDOUT = (
    "Is F-12's missing downhole pressure after October 2010",
    "When did the Volve field reach the maximum water cut",
    "Why did F-12's oil rate jump after the December 2014 shutdown",
    "Why did well F-99 stop producing in 2013",
    "Which wells needed attention at the end of 2010",
    "Why did P-1's oil decline speed up in late 2019",
)
WRITER_NOTE = """

You have already called the tools: their results are below, each with its evidence_id. \
Do not ask for more. Reply with one JSON object, the arguments of submit_findings, and \
nothing else. {submit}
Its JSON schema: {schema}"""
# The kinds of problem the checker sends back, as rounds.py counted them on the corpus.
KINDS = [
    ("bookkeeping", r"undeclared|Declared but|does not show its declared|declar|not written"),
    ("number not in source", r" is not in E\d|not found in E\d|Closest there"),
    ("cause or hedge", r"cause|hedge|uncertain|settled|conclusion"),
    ("fact on warned evidence", r"warning|marked fact|inference"),
    ("quote", r"quote|word for word|verbatim"),
    (
        "wrong cell or attribute",
        r"fits .* better|only for |is named |written in |written as |"
        r"wrong way|belongs to|row|its date|two ends|beside the quote|at most \d+ months",
    ),
    ("meaning rule", r"limit|pressure|gauge|trend|within|exceed|SAM"),
]


def _out() -> Path:
    return get_settings().data_dir / "training" / "writer"


def _held_out(question: str) -> bool:
    return any(question.strip().startswith(h) for h in HOLDOUT)


def system_prompt(as_of: str | None) -> str:
    """The agent's own instructions, without the field description (the results carry the
    facts), and the submit schema the loop gives it."""
    tool = submit_tool_schema()["function"]
    base = SYSTEM_PROMPT.format(
        field="field the question is about",
        drive="",
        wells="",
        reports="",
        as_of_note=f"\n\nData is available only up to {as_of}." if as_of else "",
    )
    schema = json.dumps(tool["parameters"], separators=(",", ":"))
    return base + WRITER_NOTE.format(submit=tool["description"], schema=schema)


def user_message(question: str, evidence: list[dict]) -> str:
    shown = []
    for item in evidence:
        result = item["result"]
        if item["tool"] == "look_at_page" and isinstance(result, dict):
            result = for_the_model(result)  # the model never sees SAM's values, as live
        shown.append(
            {
                "evidence_id": item["id"],
                "tool": item["tool"],
                "arguments": item["arguments"],
                "warnings": item.get("warnings") or [],
                "result": result,
            }
        )
    results = "\n".join(json.dumps(s, ensure_ascii=False, default=str) for s in shown)
    return f"Question: {question}\n\nTool results:\n{results}"


def _as_of(run: str) -> str | None:
    path = get_settings().runs_dir / f"{run}.json"
    if path.exists():
        return json.loads(path.read_text()).get("as_of")
    return None


def _example(question: str, evidence: list[dict], answer: dict, as_of: str | None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt(as_of)},
            {"role": "user", "content": user_message(question, evidence)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


def build(corpus: Path = CORPUS_DIR, out: Path | None = None) -> dict:
    """train.jsonl (chat examples), test.jsonl (inputs, with the run's accepted answer as
    gold where it has one), split by question, and validation.jsonl: the test's gold answers
    as chat examples, for a trainer's validation loss on questions it never trains on (a
    monitor only: nothing is chosen by it)."""
    out = out or _out()
    out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list[dict]] = {}
    for path in sorted(corpus.glob("*.json")):
        case = json.loads(path.read_text())
        runs.setdefault(case["source"]["run"], []).append(case)
    train, test = [], []
    for run, cases in sorted(runs.items()):
        cases.sort(key=lambda c: c["source"]["submission"])
        question = cases[0]["source"]["question"]
        accepted = [c for c in cases if c["expected"]["accepted"]]
        as_of = _as_of(run)
        if _held_out(question):
            test.append(
                {
                    "run": run,
                    "question": question,
                    "as_of": as_of,
                    "evidence": cases[-1]["evidence"],
                    "gold": accepted[-1]["submission"] if accepted else None,
                    "live_first_accepted": cases[0]["expected"]["accepted"],
                }
            )
        elif accepted:
            # The run's last accepted answer, with the evidence it was written from.
            case = accepted[-1]
            train.append(_example(question, case["evidence"], case["submission"], as_of))
    validation = [
        _example(t["question"], t["evidence"], t["gold"], t["as_of"]) for t in test if t["gold"]
    ]
    for name, rows in (("train", train), ("test", test), ("validation", validation)):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in train]
    manifest = {
        "train_examples": len(train),
        "train_questions": len({r["messages"][1]["content"].split("\n", 1)[0] for r in train}),
        "test_runs": len(test),
        "test_questions": len({t["question"] for t in test}),
        "test_with_gold": sum(t["gold"] is not None for t in test),
        "validation_examples": len(validation),
        "held_out": list(HOLDOUT),
        # About 3.5 characters a token for this mix of JSON and English.
        "train_tokens_estimate": round(sum(chars) / 3.5),
        "longest_example_tokens_estimate": round(max(chars, default=0) / 3.5),
        "out": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def parse(reply: str) -> dict | None:
    """The JSON object in a reply: fences and a model's thinking block taken off."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def kind(problem: str) -> str:
    return next((k for k, rx in KINDS if re.search(rx, problem, re.I)), "other")


def score(item: dict, answer: dict | None) -> dict:
    """One answer, as the checker and the eval's regexes judge it."""
    if answer is None:
        return {"run": item["run"], "valid_json": False, "accepted": False, "problems": []}
    try:
        submission = Submission(**answer)
    except Exception as e:  # noqa: BLE001 - a malformed answer is a result, not a crash
        return {
            "run": item["run"],
            "valid_json": True,
            "schema_ok": False,
            "accepted": False,
            "problems": [f"schema: {type(e).__name__}"],
        }
    log = EvidenceLog.restore(item["evidence"])
    try:
        problems = check_submission(submission, log, item["question"])
    except Exception as e:  # noqa: BLE001
        problems = [f"checker error: {type(e).__name__}"]
    text = " ".join(
        [submission.summary, *(f"{f.claim} {f.detail or ''}" for f in submission.findings)]
    )
    case = next(
        (c for c in CASES_BY_ID.values() if c.question.strip() == item["question"].strip()), None
    )
    regexes = []
    if case:
        regexes += [(f"says /{p}/", bool(re.search(p, text, re.I))) for p in case.mentions]
        regexes += [(f"never says /{p}/", not re.search(p, text, re.I)) for p in case.avoids]
    return {
        "run": item["run"],
        "valid_json": True,
        "schema_ok": True,
        "accepted": not problems,
        "problems": problems,
        "kinds": dict(Counter(kind(p) for p in problems)),
        "style": readability(submission)
        + unreadable_numbers(submission.summary)
        + dense_claims([f.claim for f in submission.findings]),
        "regexes": regexes,
        "findings": len(submission.findings),
    }


async def _ask(llm, messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
    response = await llm.client.chat.completions.create(
        model=llm.model, messages=messages, temperature=0, max_tokens=max_tokens
    )
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


async def evaluate(items: list[dict], fake: str | None, rounds: int, max_tokens: int) -> list[dict]:
    llm = None
    if fake is None:
        from wellpilot.llm import make_llm

        llm = make_llm(get_settings(), timeout=240.0)
    results = []
    for item in items:
        messages = [
            {"role": "system", "content": system_prompt(item["as_of"])},
            {"role": "user", "content": user_message(item["question"], item["evidence"])},
        ]
        tokens_in = tokens_out = 0
        started = time.monotonic()
        attempts = []
        for attempt in range(rounds):
            if fake == "gold":
                reply = json.dumps(item["gold"]) if item["gold"] else ""
            elif fake == "broken":
                reply = _broken(item["gold"])
            else:
                reply, i, o = await _ask(llm, messages, max_tokens)
                tokens_in, tokens_out = tokens_in + i, tokens_out + o
            result = score(item, parse(reply))
            attempts.append(result)
            if result["accepted"] or attempt == rounds - 1:
                break
            # One more round the way the loop gives it: the checker's problems back.
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": json.dumps({"accepted": False, "problems": result["problems"]}),
                },
            ]
        results.append(
            {
                **attempts[0],
                "accepted_by_round": next(
                    (i + 1 for i, a in enumerate(attempts) if a["accepted"]), None
                ),
                "live_first_accepted": item["live_first_accepted"],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "seconds": round(time.monotonic() - started, 1),
                "last": attempts[-1],
            }
        )
        print(
            f"{item['run']}: first {'ACCEPTED' if attempts[0]['accepted'] else 'rejected'}"
            f" ({', '.join(f'{k} {v}' for k, v in attempts[0].get('kinds', {}).items())})"
        )
    return results


def _broken(gold: dict | None) -> str:
    """The gold answer with its first declared number made 17% larger, text and value."""
    if not gold:
        return ""
    answer = json.loads(json.dumps(gold))
    for finding in answer["findings"]:
        for n in finding.get("numbers") or []:
            if n["value"] and n["text"] in finding["claim"]:
                new = f"{n['value'] * 1.17:.4g}"
                finding["claim"] = finding["claim"].replace(n["text"], new, 1)
                n.update(text=new, value=float(new))
                return json.dumps(answer)
    return json.dumps(answer)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    first = sum(r["accepted"] for r in results)
    kinds = Counter()
    for r in results:
        kinds.update(r.get("kinds", {}))
    regex = [ok for r in results for _, ok in r.get("regexes", [])]
    return {
        "items": n,
        "valid_json": sum(r["valid_json"] for r in results),
        "accepted_first": first,
        "accepted_within_rounds": sum(r["accepted_by_round"] is not None for r in results),
        "live_muse_first_accepted": sum(bool(r["live_first_accepted"]) for r in results),
        "problem_kinds": dict(kinds.most_common()),
        "style_flags": sum(bool(r.get("style")) for r in results),
        "regex_checks_passed": f"{sum(regex)} of {len(regex)}",
        "cost_usd_at_muse_prices": round(
            sum(r["tokens_in"] for r in results) * PRICE_IN
            + sum(r["tokens_out"] for r in results) * PRICE_OUT,
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="write train.jsonl and test.jsonl (free)")
    run = sub.add_parser("eval", help="score a model's answers on the held-out questions")
    run.add_argument("--fake", choices=["gold", "broken"], help="score fixed answers (free)")
    run.add_argument("--label", default="", help="a name for the scorecard (e.g. base, tuned)")
    run.add_argument(
        "--rounds", type=int, default=2, help="answers per item, with the checker's reply between"
    )
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(), indent=1))
        return
    path = _out() / "test.jsonl"
    if not path.exists():
        build()
    items = [json.loads(line) for line in path.read_text().splitlines()]
    if args.fake:
        items = [i for i in items if i["gold"]]
    if args.limit:
        items = items[: args.limit]
    results = asyncio.run(evaluate(items, args.fake, args.rounds, args.max_tokens))
    summary = summarize(results)
    settings = get_settings()
    label = args.label or args.fake or settings.llm_provider
    card = {
        "label": label,
        "model": None
        if args.fake
        else (
            settings.fireworks_model
            if settings.llm_provider == "fireworks"
            else settings.baseten_model
        ),
        "summary": summary,
        "results": results,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = settings.data_dir / "evals" / f"writer-{label}-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"scorecard: {target}")


if __name__ == "__main__":
    main()
