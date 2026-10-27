"""Agent eval: real questions, each with checks a good answer must pass.

The checker verifies one answer's numbers; this checks the agent's behaviour across the
questions the product promises to answer: that it verifies, calls the tools the question
needs (look_at_page for a figure, field_vs_design for the plan's limits), says the facts
any correct answer contains, refuses a well that does not exist, and stays within a step
and cost budget. Run it before a deploy, a demo recording or a prompt change.

Spends tokens (about $2 for all cases on Muse Spark Open; Emil runs it):
    uv run python -m wellpilot.eval                 # every case
    uv run python -m wellpilot.eval --only gauge    # one case
    uv run python -m wellpilot.eval --list          # print the cases, no tokens

Each run is saved to data/runs (replayable) and added to the checker corpus, like a CLI
run; the scorecard goes to data/evals/<time>.json. Exits 1 if any case fails.
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from wellpilot.agent import corpus
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.loop import AgentRun, run_agent
from wellpilot.api.runs import RunRecord, new_run_id
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.llm import LLM, make_llm

# Muse Spark Open on Baseten, USD per token (the same as Meta's hosted Muse Spark).
PRICE_IN, PRICE_OUT = 1.25e-6, 4.25e-6


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    as_of: date | None = None
    tools: tuple[str, ...] = ()  # each must be called successfully
    any_tools: tuple[str, ...] = ()  # at least one of these must be
    mentions: tuple[str, ...] = ()  # regexes the answer (summary and findings) must match
    max_steps: int = 10
    max_cost: float = 0.30


CASES = [
    Case(
        "decline_vs_plan",
        "Why did well F-12 decline faster after 2010? Compare with the 2005 development plan.",
        tools=("detect_events",),
        any_tools=("search_reports", "plan_vs_actual", "read_report_page", "field_vs_design"),
        mentions=(r"2010-07-18|4,?050",),
        max_steps=12,
        max_cost=0.35,
    ),
    Case(
        "gauge",
        "Is F-12's missing downhole pressure after October 2010 a reservoir problem or a "
        "sensor problem?",
        tools=("detect_events",),
        mentions=(r"2010-10-1[12]", r"gauge|sensor"),
    ),
    Case(
        "forecast_as_of",
        "Forecast F-14 oil production for the next 12 months and say how confident the "
        "forecast is.",
        as_of=date(2013, 1, 1),
        tools=("forecast_production",),
        mentions=(r"P50",),
    ),
    Case(
        "design_limit",
        "When did the Volve field reach the maximum water cut the 2005 plan designed the "
        "platform for?",
        tools=("field_vs_design",),
        mentions=(r"2012-06|June 2012",),
    ),
    Case(
        "plan_figure",
        "What oil plateau rate and duration does the forecast figure on page 23 of the 2005 "
        "development plan show?",
        tools=("look_at_page",),
        mentions=(r"plateau",),
    ),
    Case(
        "restart_2014",
        "Why did F-12's oil rate jump after the December 2014 shutdown?",
        tools=("detect_events",),
        mentions=(r"2014-12|2015-01|December 2014",),
    ),
    Case(
        "unknown_well",
        "Why did well F-99 stop producing in 2013?",
        any_tools=("list_wells", "run_sql"),
        mentions=(
            r"F-99",
            r"\bnot\b|\bno (data|well|record|such)\b|does(n't| not) exist",
        ),
        max_steps=6,
        max_cost=0.15,
    ),
    Case(
        "top_producers",
        "Which wells produced most of Volve's oil, and how much did each produce?",
        any_tools=("list_wells", "run_sql", "production_monthly"),
        mentions=(r"F-12", r"F-14"),
    ),
]
CASES_BY_ID = {c.id: c for c in CASES}


def cost(run: AgentRun) -> float:
    tokens_in, tokens_out = run.total_tokens
    return tokens_in * PRICE_IN + tokens_out * PRICE_OUT


def score(case: Case, run: AgentRun) -> list[tuple[str, bool, str]]:
    """Each check as (name, passed, detail)."""
    called = {e.tool for e in run.evidence}
    answer = " ".join([run.summary or "", *(f.claim for f in run.findings)])
    checks = [
        ("verified", run.verified, "" if run.verified else (run.stopped or "checker rejected")),
    ]
    for tool in case.tools:
        checks.append((f"calls {tool}", tool in called, ", ".join(sorted(called))))
    if case.any_tools:
        hit = called & set(case.any_tools)
        checks.append(("calls one of " + "/".join(case.any_tools), bool(hit), ", ".join(hit)))
    for pattern in case.mentions:
        found = re.search(pattern, answer, re.IGNORECASE)
        checks.append((f"says /{pattern}/", bool(found), found.group(0) if found else ""))
    steps = len(run.steps)
    checks.append((f"<= {case.max_steps} steps", steps <= case.max_steps, str(steps)))
    spent = cost(run)
    checks.append((f"<= ${case.max_cost:.2f}", spent <= case.max_cost, f"${spent:.2f}"))
    if case.as_of:  # a forecast starts after the cut and never sees the data behind it
        starts = [
            str(e.result.get("forecast_from"))
            for e in run.evidence
            if e.tool == "forecast_production" and isinstance(e.result, dict)
        ]
        late = all(s > case.as_of.isoformat() for s in starts)
        checks.append(("forecast starts after the cut", late, ", ".join(starts)))
    return checks


async def run_case(
    case: Case,
    llm: LLM,
    runs_dir: Path,
    processed_dir: Path | None = None,
    add_to_corpus: bool = True,
) -> dict:
    settings = get_settings()
    store = ProductionStore(processed_dir or settings.processed_dir, as_of=case.as_of)
    record = RunRecord(new_run_id(), case.question, case.as_of)
    record.start(llm)
    run = await run_agent(
        case.question,
        llm,
        store,
        reasoning_effort=settings.agent_reasoning_effort,
        timeout_s=settings.run_timeout_s,
        on_event=record.emit,
        evidence_log=record.log,
    )
    record.complete(run)
    path = record.save(runs_dir)
    if add_to_corpus and CORPUS_DIR.parent.is_dir():
        corpus.add(path)
    checks = score(case, run)
    return {
        "case": case.id,
        "run": record.id,
        "passed": all(ok for _, ok, _ in checks),
        "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in checks],
        "steps": len(run.steps),
        "tokens": dict(zip(("input", "output"), run.total_tokens, strict=True)),
        "cost": round(cost(run), 4),
    }


def print_report(results: list[dict], model: str) -> None:
    print(f"\neval on {model}: {sum(r['passed'] for r in results)}/{len(results)} cases passed")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n{mark}  {r['case']}  (run {r['run']}, {r['steps']} steps, ${r['cost']:.2f})")
        for c in r["checks"]:
            if not c["passed"]:
                print(f"      x {c['check']}: {c['detail']}")
    print(f"\ntotal ${sum(r['cost'] for r in results):.2f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent eval (spends tokens).")
    parser.add_argument("--only", action="append", choices=sorted(CASES_BY_ID))
    parser.add_argument("--list", action="store_true", help="print the cases and exit")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    cases = [CASES_BY_ID[c] for c in args.only] if args.only else CASES
    if args.list:
        for c in cases:
            print(f"{c.id}: {c.question}" + (f" (data up to {c.as_of})" if c.as_of else ""))
        return

    settings = get_settings()
    llm = make_llm(settings)
    print(f"provider: {llm.provider}, model: {llm.model}, {len(cases)} cases")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case: Case) -> dict:
        async with semaphore:
            result = await run_case(case, llm, settings.runs_dir)
            print(f"  {case.id}: {'pass' if result['passed'] else 'FAIL'}, run {result['run']}")
            return result

    results = await asyncio.gather(*(one(c) for c in cases))
    print_report(results, llm.model)
    out = settings.data_dir / "evals"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {"model": llm.model, "at": stamp, "results": results}
    (out / f"{stamp}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"scorecard: {out / f'{stamp}.json'}")
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
