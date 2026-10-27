"""What the model knows about Volve without the data: the time machine's honesty check.

Volve is a public dataset; a model may have read about it, F-12's later years included. A
forecast made "as of 2010" is only blind if the model cannot recall what came after. This
asks the agent's own model, with no tools and no data, about F-12's years after 2010, and
sets each answer beside what the data says. A model that recalls those facts makes the
Volve time machine weaker evidence; the synthetic field (wellpilot.synthetic), which no
model can have read about, is the test that does not depend on it.

    uv run python -m wellpilot.memory_probe           # spends tokens (a few cents)
    uv run python -m wellpilot.memory_probe --guess   # the model must name a number

Told it may say it does not know, a model that declines could still hold a weak memory;
--guess makes it estimate anyway, so its numbers can be set against the data's.
"""

import asyncio
import json
import sys
from datetime import UTC, datetime

from wellpilot.config import get_settings
from wellpilot.llm import make_llm

# What the data says, for comparison (production data, daily drilling reports).
QUESTIONS = [
    (
        "How much oil did well F-12 (15/9-F-12) of the Volve field in the Norwegian North Sea "
        "produce in 2011, in standard cubic metres? Give your best number.",
        "about 340,000 Sm3 in 2011",
    ),
    (
        "What happened to the downhole pressure gauge of Volve well F-12 in 2010?",
        "it read 0 from 2010-10-12 to the end of the data: a dead gauge",
    ),
    (
        "What work was done on Volve well F-12 in November 2014, and what did the well do after?",
        "a plug at 3150 m and a new zone perforated (3126.2-3130.7 m MD); water cut 0.97 to "
        "about 0.3, oil about 180 to 860 Sm3/d",
    ),
    (
        "What were Volve well F-12's oil rate and water cut at the end of 2012?",
        "about 610 Sm3/d of oil per producing day at a water cut of 0.88 (Nov-Dec 2012)",
    ),
    (
        "When did the Volve field stop producing, and how much oil did it produce in total?",
        "shut down in September 2016; about 10 million Sm3 (63 million barrels)",
    ),
]
SYSTEM = (
    "Answer from what you know. If you do not know, say so; do not guess numbers you do not "
    "remember."
)
GUESS = (
    "Answer every question with your best estimate, as specific numbers and dates where they "
    "are asked for, even if you are unsure. Then say in a few words how sure you are."
)


async def ask(question: str, system: str = SYSTEM) -> dict:
    llm = make_llm(get_settings())
    response = await llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        reasoning_effort="low",
    )
    usage = response.usage
    return {
        "answer": response.choices[0].message.content,
        "tokens": (usage.prompt_tokens + usage.completion_tokens) if usage else 0,
    }


async def probe(system: str = SYSTEM) -> list[dict]:
    answers = await asyncio.gather(*(ask(q, system) for q, _ in QUESTIONS))
    return [
        {"question": q, "data_says": truth, **a}
        for (q, truth), a in zip(QUESTIONS, answers, strict=True)
    ]


def main() -> None:
    guess = "--guess" in sys.argv[1:]
    settings = get_settings()
    results = asyncio.run(probe(GUESS if guess else SYSTEM))
    out = settings.data_dir / "evals"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"memory_probe{'-guess' if guess else ''}-{stamp}.json"
    model = make_llm(settings).model
    path.write_text(
        json.dumps(
            {"model": model, "mode": "guess" if guess else "may decline", "results": results},
            indent=1,
            ensure_ascii=False,
        )
    )
    for r in results:
        print(f"\nQ: {r['question']}\nModel: {r['answer']}\nData: {r['data_says']}")
    print(f"\n{sum(r['tokens'] for r in results)} tokens; saved to {path}")


if __name__ == "__main__":
    main()
