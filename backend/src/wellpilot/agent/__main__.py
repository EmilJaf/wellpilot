"""Ask the agent a question from the terminal and watch every step.

Run: uv run python -m wellpilot.agent "Why did F-12 decline faster after 2010?"
This calls the LLM in .env (LLM_PROVIDER) and spends tokens.
"""

import argparse
import asyncio
import json
import textwrap
from datetime import date

from wellpilot.agent.loop import Step, run_agent
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.llm import make_llm

DEFAULT_QUESTION = "Why did well F-12 decline faster after 2010? Show the evidence."
RESULT_PREVIEW_CHARS = 600


def _indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text.strip(), prefix)


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
        print(f"  -> {call['name']}({call['arguments']})")
        result = json.dumps(call["result"], ensure_ascii=False)
        if len(result) > RESULT_PREVIEW_CHARS:
            result = result[:RESULT_PREVIEW_CHARS] + f"... [{len(result)} chars]"
        print(_indent(result, "     "))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask WellPilot a question.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--max-steps", type=int, default=10)
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

    tokens_in, tokens_out = run.total_tokens
    print(f"\n=== {'answer' if run.finished else 'stopped: step limit reached'}")
    if run.answer:
        print(run.answer)
    print(f"\n{len(run.steps)} steps, {tokens_in} input + {tokens_out} output tokens")


if __name__ == "__main__":
    asyncio.run(main())
