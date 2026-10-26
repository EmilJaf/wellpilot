"""Ask the agent a question from the terminal and watch every step.

Run: uv run python -m wellpilot.agent "Why did F-12 decline faster after 2010?"
This calls the LLM in .env (LLM_PROVIDER) and spends tokens.
"""

import argparse
import asyncio
import json
import textwrap
from datetime import date

from wellpilot.agent.loop import SUBMIT, AgentRun, Step, run_agent
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.llm import make_llm

DEFAULT_QUESTION = "Why did well F-12 decline faster after 2010? Show the evidence."
RESULT_PREVIEW_CHARS = 400


def _indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text.strip(), prefix)


def _print_call(call: dict) -> None:
    result = call["result"]
    if call["name"] == SUBMIT:
        if result["accepted"]:
            print("  -> submit_findings: claim check PASSED")
        else:
            print(f"  -> submit_findings: claim check REJECTED ({len(result['problems'])})")
            for problem in result["problems"]:
                print(_indent(f"- {problem}", "     "))
        return
    tag = f"[{call['evidence_id']}] " if "evidence_id" in call else "[error] "
    print(f"  -> {tag}{call['name']}({call['arguments']})")
    inner = result.get("result")
    if isinstance(inner, dict):
        for warning in inner.get("warnings", []):
            print(_indent(f"! {warning}", "     "))
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > RESULT_PREVIEW_CHARS:
        text = text[:RESULT_PREVIEW_CHARS] + f"... [{len(text)} chars]"
    print(_indent(text, "     "))


def print_step(step: Step) -> None:
    print(
        f"\n=== step {step.index}  ({step.seconds:.1f}s, "
        f"{step.prompt_tokens} in / {step.completion_tokens} out tokens)"
    )
    if step.reasoning:
        print("  reasoning:")
        print(_indent(step.reasoning))
    if step.content:
        print("  model:")
        print(_indent(step.content))
    for call in step.tool_calls:
        _print_call(call)


def print_run(run: AgentRun) -> None:
    if not run.finished:
        status = "STOPPED: step limit reached"
    elif run.verified:
        status = "VERIFIED: every finding passed the claim check"
    else:
        status = "UNVERIFIED: the claim check did not pass"
    print(f"\n=== answer ({status})")
    if run.summary:
        print(run.summary)
    for n, finding in enumerate(run.findings, start=1):
        print(f"\n{n}. [{finding.confidence}] {finding.claim}")
        print(f"   evidence: {', '.join(finding.evidence_ids)}")
    tokens_in, tokens_out = run.total_tokens
    print(
        f"\n{len(run.steps)} steps, {len(run.evidence)} evidence items, "
        f"{len(run.check_rounds)} submission(s), {tokens_in} input + {tokens_out} output tokens"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask WellPilot a question.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--as-of", type=date.fromisoformat, help="hide data after this day (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    settings = get_settings()
    llm = make_llm(settings)
    store = ProductionStore(settings.processed_dir, as_of=args.as_of)
    print(f"provider: {llm.provider}, model: {llm.model}, data as of: {args.as_of or 'all'}")
    print(f"question: {args.question}")

    run = await run_agent(args.question, llm, store, max_steps=args.max_steps, on_step=print_step)
    print_run(run)


if __name__ == "__main__":
    asyncio.run(main())
