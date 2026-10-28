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
from wellpilot.agent.checker import (
    MAX_CLAIM_NUMBERS,
    claim_numbers,
)
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.loop import SUBMIT, AgentRun, run_agent
from wellpilot.agent.review import review
from wellpilot.agent.sandbox import SandboxPool, modal_pool, python_tool
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
    max_steps: int = 12
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
        mentions=(r"P50|central estimate|most likely",),
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
        "water_shutoff_f12",
        "F-12's water cut fell from about 0.97 to about 0.3 at the end of 2014. What was done "
        "to the well, and why did the plug set in 2010 not do the same?",
        tools=("well_operations", "production_log"),
        mentions=(r"3150", r"3210", r"2009|2010-09"),
        max_steps=10,
    ),
    Case(
        "plug_decision_2014",
        "F-12's water cut is near 0.97 and its oil rate has fallen below 200 Sm3/d. What would "
        "you recommend doing to the well, and why?",
        as_of=date(2014, 10, 31),
        tools=("production_log",),
        mentions=(r"3[,\s]?1(?:4[7-9]|5\d|60)", r"3[,\s]?210|2010"),  # blank pipe
        max_steps=10,
    ),
    Case(
        "own_code_f12",
        "Over 2011 to 2013, how closely did F-12's monthly water cut follow its choke opening? "
        "Give the correlation and what it suggests.",
        tools=("run_python",),
        mentions=(r"correlat", r"choke"),
        max_steps=10,
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
        "waterflood_f12",
        "Was F-12's falling oil rate in 2011-2013 a loss of reservoir pressure or water taking "
        "over the well? How much oil was left before an uneconomic water cut?",
        as_of=date(2013, 12, 31),
        tools=("waterflood",),
        mentions=(r"liquid|choke", r"water cut|WOR"),
        max_steps=12,
        max_cost=0.35,
    ),
    Case(
        "screen_2010",
        "Which wells needed attention at the end of 2010, and why?",
        as_of=date(2010, 12, 31),
        tools=("field_screen",),
        mentions=(r"F-12", r"gauge|pressure"),
    ),
    Case(
        "f11b_completion",
        "When was well F-11 B completed, and how does its completion report's timing compare "
        "with when it started producing?",
        any_tools=("search_reports", "read_report_page", "look_at_page"),
        mentions=(r"2013-07", r"F-11"),
    ),
    Case(
        "time_machine_2010",
        "Which wells need attention right now, and why? Forecast the next 12 months of oil "
        "for the well that needs it most.",
        as_of=date(2010, 12, 31),
        tools=("field_screen", "forecast_production"),
        mentions=(r"F-12", r"gauge|pressure"),
        max_steps=10,
        max_cost=0.30,
    ),
    Case(
        "top_producers",
        "Which wells produced most of Volve's oil, and how much did each produce?",
        any_tools=("list_wells", "run_sql", "production_monthly"),
        mentions=(r"F-12", r"F-14"),
    ),
]
# The synthetic field (python -m wellpilot.synthetic; run with DATA_DIR=data/synthetic):
# questions whose answers were planted, so a right answer and a confident wrong one differ.
SYNTHETIC_CASES = [
    Case(
        "syn_screen",
        "Which wells need attention right now, and why?",
        tools=("field_screen",),
        mentions=(r"P-1", r"gauge"),
        max_steps=10,
    ),
    Case(
        "syn_step",
        "Why did P-2's oil rate drop in November 2018?",
        tools=("detect_events",),
        mentions=(r"2018-11", r"choke"),
        max_steps=10,
    ),
    Case(
        "syn_depletion",
        "Is P-3's decline caused by water or by falling reservoir pressure?",
        mentions=(r"P-3", r"pressure|deplet"),
        max_steps=10,
    ),
    Case(
        "syn_injector",
        "Why did P-1's oil decline speed up in late 2019?",
        mentions=(r"I-2", r"2019-0[89]"),
        max_steps=12,
    ),
    Case(
        "syn_forecast",
        "Forecast P-2 oil production for the next 12 months and say how confident the forecast is.",
        as_of=date(2019, 3, 31),
        tools=("forecast_production",),
        mentions=(r"P50|P90|P10|central estimate|range",),
        max_steps=10,
    ),
]
SUITES = {"volve": CASES, "synthetic": SYNTHETIC_CASES}
CASES_BY_ID = {c.id: c for c in [*CASES, *SYNTHETIC_CASES]}


# A number in the summary as a reader meets it; dates, years and well names are not.
SUMMARY_NUMBER = re.compile(r"(?<![\w.-])\d[\d,]*(?:\.\d+)?")
MASK = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b[A-Z]{1,3}-\d+\b|[\"“][^\"”]*[\"”]|\b(?:19|20)\d{2}\b"
)


def unreadable_numbers(summary: str) -> list[str]:
    """Summary numbers with more than 3 significant digits that read as noise: a whole
    number of 10,000 or more ("1,133,053") or a decimal under 10 ("0.4461"). "4,050" and
    "10.04" read fine."""
    found = []
    for match in SUMMARY_NUMBER.finditer(MASK.sub(" ", summary)):
        written = match.group(0).rstrip(",")
        digits = written.replace(",", "")
        whole = "." not in digits
        significant = digits.replace(".", "").lstrip("0").rstrip("0" if whole else "")
        value = float(digits)
        noisy = value >= 10_000 if whole else value < 10
        if len(significant) > 3 and noisy:
            found.append(written)
    return found


# What a finding's claim must not carry: fit statistics belong in its detail, and internal
# fields (a score, a key name) mean nothing to a reader.
JARGON = re.compile(r"\br2(?:_log)?\b|r²|\bAIC\b|\bqi\b|\bdi\b|\battention \d|_sm3|_bar\b")


def dense_claims(claims: list[str]) -> list[str]:
    """Claims a manager could not follow: jargon, or more numbers than a sentence holds."""
    found = []
    for n, claim in enumerate(claims, start=1):
        numbers = claim_numbers(claim)
        if JARGON.search(claim) or len(numbers) > MAX_CLAIM_NUMBERS:
            found.append(
                f"finding {n} ({len(numbers)} numbers{', jargon' if JARGON.search(claim) else ''})"
            )
    return found


def cost(run: AgentRun) -> float:
    tokens_in, tokens_out = run.total_tokens
    reviewed = (run.review or {}).get("tokens", 0)  # mostly input
    return (tokens_in + reviewed) * PRICE_IN + tokens_out * PRICE_OUT


def to_answer(run: AgentRun) -> tuple[int, float]:
    """Steps and cost up to the first answer the checker accepted: the agent's own work, which
    the budgets measure. A second opinion and the revision it asks for come on top (they
    were added after the budgets were set); the run's totals are reported apart."""
    for i, step in enumerate(run.steps, start=1):
        if any(c["name"] == SUBMIT and c["result"].get("accepted") for c in step.tool_calls):
            done = run.steps[:i]
            spent = (
                sum(s.prompt_tokens for s in done) * PRICE_IN
                + sum(s.completion_tokens for s in done) * PRICE_OUT
            )
            return i, spent
    return len(run.steps), cost(run)


def score(case: Case, run: AgentRun) -> list[tuple[str, bool, str]]:
    """Each check as (name, passed, detail)."""
    called = {e.tool for e in run.evidence}
    answer = " ".join([run.summary or "", *(f"{f.claim} {f.detail or ''}" for f in run.findings)])
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
    long = unreadable_numbers(run.summary or "")
    checks.append(("summary numbers <= 3 significant digits", not long, ", ".join(long)))
    dense = dense_claims([f.claim for f in run.findings])
    checks.append(("claims in plain words", not dense, ", ".join(dense)))
    steps, spent = to_answer(run)
    checks.append(
        (f"<= {case.max_steps} steps to the checked answer", steps <= case.max_steps, str(steps))
    )
    checks.append(
        (f"<= ${case.max_cost:.2f} to the checked answer", spent <= case.max_cost, f"${spent:.2f}")
    )
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
    sandboxes: SandboxPool | None = None,
) -> dict:
    settings = get_settings()
    store = ProductionStore(processed_dir or settings.processed_dir, as_of=case.as_of)
    record = RunRecord(new_run_id(), case.question, case.as_of)
    record.start(llm)
    async with python_tool(sandboxes, case.as_of) as python_fn:
        run = await run_agent(
            case.question,
            llm,
            store,
            reasoning_effort=settings.agent_reasoning_effort,
            timeout_s=settings.run_timeout_s,
            on_event=record.emit,
            evidence_log=record.log,
            review_fn=review if settings.review_answers else None,  # the product as served
            python_fn=python_fn,
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
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="volve",
        help="synthetic: the planted field; run with DATA_DIR=data/synthetic",
    )
    parser.add_argument("--list", action="store_true", help="print the cases and exit")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    cases = [CASES_BY_ID[c] for c in args.only] if args.only else SUITES[args.suite]
    if args.list:
        for c in cases:
            print(f"{c.id}: {c.question}" + (f" (data up to {c.as_of})" if c.as_of else ""))
        return

    settings = get_settings()
    synthetic = (settings.processed_dir / "field.json").exists()
    if any(c in SYNTHETIC_CASES for c in cases) != synthetic:
        sys.exit("synthetic cases run on the synthetic field and only there: set DATA_DIR")
    llm = make_llm(settings)
    # run_python as served: a Modal sandbox per case, when Modal is set up.
    sandboxes = modal_pool(settings.processed_dir) if settings.python_sandbox else None
    print(
        f"provider: {llm.provider}, model: {llm.model}, {len(cases)} cases, "
        f"run_python: {'on' if sandboxes else 'off'}"
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case: Case) -> dict:
        async with semaphore:
            result = await run_case(case, llm, settings.runs_dir, sandboxes=sandboxes)
            print(f"  {case.id}: {'pass' if result['passed'] else 'FAIL'}, run {result['run']}")
            return result

    try:
        results = await asyncio.gather(*(one(c) for c in cases))
    finally:
        if sandboxes is not None:
            await sandboxes.aclose()  # the one warming for a next case
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
