"""Extract facts from report pages with Muse Spark vision.

Muse Spark reads each page image (tables and figures included) together with the page's
text layer and returns structured facts: an English statement, the verbatim quote in the
original language, topic and wells. Each quote is then checked against the page's text
layer, so a fact the model invented shows up as unverified.

Results are cached per page, so a rerun only pays for pages not done yet.

Run (spends tokens): uv run python -m wellpilot.ingest.report_facts --pages 23,25
"""

import argparse
import asyncio
import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from wellpilot.config import get_settings
from wellpilot.ingest.reports import PAGES_FILE, REPORTS_BY_ID, normalize_text, page_dir
from wellpilot.llm import LLM, json_schema, make_llm

FACTS_FILE = "report_facts.parquet"
SUMMARIES_FILE = "report_summaries.parquet"
TEXT_LAYER_CHARS = 8000
CONCURRENCY = 4

Topic = Literal[
    "production_forecast",
    "water",
    "injection",
    "pressure",
    "reserves",
    "facility_limits",
    "wells",
    "fluids",
    "schedule",
    "economics",
    "other",
]

SYSTEM_PROMPT = """\
You read pages of oil field reports about the Volve field (Norwegian North Sea, wells \
named like F-12 or 15/9-F-12). You get the page image and, when available, its text \
layer. Tables and figures are only fully visible in the image.

Extract the facts a petroleum engineer analysing well performance would cite: production \
forecasts and plateau rates, water breakthrough and water production, injection strategy, \
pressures, reserves, facility limits (liquid, oil, gas, injection capacity, maximum water \
cut), well design and completion, schedules.

For each fact:
- statement_en: one English sentence with the numbers and units exactly as on the page.
- quote: the verbatim excerpt in the page's original language that supports it (at most \
300 characters). Copy it character for character from the text layer when the fact is in \
the text; for a table, copy the row label and value; for a figure, its caption.
- source: text, table or figure. For a figure, describe what it shows in statement_en.
- wells: the wells the fact is about, written like F-12; empty for the whole field.

Never invent facts or numbers. Pages with nothing relevant (cover, contents, legal text) \
get an empty fact list."""


class PageFact(BaseModel):
    topic: Topic
    statement_en: str = Field(description="One English sentence with the page's numbers.")
    quote: str = Field(description="Verbatim excerpt in the original language.")
    source: Literal["text", "table", "figure"]
    wells: list[str] = Field(description="Wells the fact is about; empty for the field.")


class PageExtraction(BaseModel):
    page_summary_en: str = Field(description="One or two English sentences on the page.")
    facts: list[PageFact]


def parse_pages(spec: str, total: int) -> list[int]:
    """'23,25-27' -> [23, 25, 26, 27]; 'all' -> every page."""
    if spec == "all":
        return list(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        pages.update(range(int(first), int(last or first) + 1))
    if not pages or min(pages) < 1 or max(pages) > total:
        raise ValueError(f"pages must be within 1-{total}")
    return sorted(pages)


QuoteCheck = Literal["exact", "tokens", "unverified"]
# Glyphs the PDF text layer drops: vulgar fractions (8 ½", 10 ¾") and the ³ in Sm³.
DROPPED_GLYPHS = re.compile(r"[¼-¾⅐-⅞³]")


def _clean(text: str) -> str:
    """Normalized, lowercase text without the glyphs the text layer loses.

    The superscript in "Sm³" comes out of the text layer as "Sm3", "Sm 3" or nothing at
    all, so it is dropped on both sides.
    """
    text = re.sub(r"(?i)(?<=sm)\s*3(?!\d)", "", normalize_text(text))  # not "Sm 30"
    return DROPPED_GLYPHS.sub("", text).casefold()


def squash(text: str) -> str:
    """Only letters and digits, for spacing-proof matching."""
    return re.sub(r"\W+", "", _clean(text))


def quote_in_text(quote: str, page_text: str) -> bool:
    """True if the quote appears in the page text; quotes under 6 letters or digits (such
    as "2 år") are too short to prove anything."""
    squashed = squash(quote)
    return len(squashed) >= 6 and squashed in squash(page_text)


# Words, and numbers kept whole with their separators: "0,80" is one token, not 0 and 80.
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")


def quote_check(quote: str, page_text: str) -> QuoteCheck:
    """'exact': the quote is on the page as written. 'tokens': every word and whole number
    of the quote is on the page, but not in that order, as when a table row is read across
    columns. 'unverified': neither, e.g. text inside a figure or a misquote. Numbers count
    whole, so a quote whose number was swapped ("vannkutt 0,40" for 0,80) is unverified
    even when its digits occur elsewhere on the page."""
    if quote_in_text(quote, page_text):
        return "exact"
    words = set(TOKEN.findall(_clean(quote)))
    if len(words) >= 3 and words <= set(TOKEN.findall(_clean(page_text))):
        return "tokens"
    return "unverified"


def _fingerprint() -> str:
    """Of the prompt and output schema: a cached page made with others is stale."""
    spec = SYSTEM_PROMPT + json.dumps(json_schema(PageExtraction), sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:12]


def cache_path(processed_dir: Path, doc_id: str, page: int) -> Path:
    return page_dir(processed_dir, doc_id) / "facts" / f"p{page:03d}.json"


async def extract_page(
    llm: LLM,
    doc_id: str,
    page: pd.Series,
    processed_dir: Path,
    reasoning_effort: str = "",
) -> dict:
    """Ask the model for one page's facts; return the cache record (not yet written)."""
    report = REPORTS_BY_ID[doc_id]
    image = base64.b64encode((processed_dir / page["image"]).read_bytes()).decode()
    text_layer = page["text"][:TEXT_LAYER_CHARS] if page["has_text"] else "(no text layer)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Document: {report.title}, issued {report.doc_date[:7]}. "
                    f"Page {page['page']} of {page['pages']}.\n\nText layer:\n{text_layer}",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        },
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "page_extraction", "schema": json_schema(PageExtraction)},
    }
    started = time.monotonic()
    errors = []
    for _ in range(2):  # one retry if the JSON does not validate
        response = await llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            response_format=response_format,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        try:
            extraction = PageExtraction.model_validate_json(
                response.choices[0].message.content or ""
            )
        except ValidationError as e:
            errors.append(str(e)[:500])
            continue
        return {
            "doc_id": doc_id,
            "page": int(page["page"]),
            "model": llm.model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "reasoning_tokens": (details.reasoning_tokens or 0) if details else 0,
            "reasoning_effort": reasoning_effort or "default",
            "seconds": round(time.monotonic() - started, 1),
            "fingerprint": _fingerprint(),
            "extraction": extraction.model_dump(),
        }
    raise ValueError(f"page {page['page']}: no valid extraction: {errors}")


def combine(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge every cached page of every document into facts and summaries tables, checking
    each quote against its page's text layer."""
    pages = pd.read_parquet(processed_dir / PAGES_FILE).set_index(["doc_id", "page"])
    facts, summaries = [], []
    for path in sorted((processed_dir / "reports").glob("*/facts/p*.json")):
        record = json.loads(path.read_text())
        key = (record["doc_id"], record["page"])
        text = pages.loc[key, "text"] if key in pages.index else ""
        extraction = record["extraction"]
        summaries.append(
            {
                "doc_id": record["doc_id"],
                "page": record["page"],
                "page_summary_en": extraction["page_summary_en"],
                "facts": len(extraction["facts"]),
                "model": record["model"],
            }
        )
        for n, fact in enumerate(extraction["facts"], start=1):
            facts.append(
                {
                    "doc_id": record["doc_id"],
                    "page": record["page"],
                    "fact_no": n,
                    "topic": fact["topic"],
                    "statement_en": fact["statement_en"],
                    "quote": fact["quote"],
                    "quote_check": (check := quote_check(fact["quote"], text)),
                    "quote_verified": check != "unverified",
                    "source": fact["source"],
                    "wells": ", ".join(fact["wells"]),
                }
            )
    return pd.DataFrame(facts), pd.DataFrame(summaries)


async def run(doc_id: str, page_numbers: list[int], force: bool) -> None:
    settings = get_settings()
    processed = settings.processed_dir
    pages = pd.read_parquet(processed / PAGES_FILE)
    pages = pages[pages["doc_id"] == doc_id].set_index("page")

    def stale(page: int) -> bool:
        path = cache_path(processed, doc_id, page)
        return (
            not path.exists() or json.loads(path.read_text()).get("fingerprint") != _fingerprint()
        )

    todo = [p for p in page_numbers if force or stale(p)]
    print(
        f"{doc_id}: {len(page_numbers)} pages requested, {len(todo)} to extract "
        "(missing, made with another prompt or schema, or --force)"
    )

    if todo:
        llm = make_llm(settings, timeout=180.0)  # a dense page can take minutes to transcribe
        effort = settings.extract_reasoning_effort
        print(f"provider: {llm.provider}, model: {llm.model}, reasoning: {effort or 'default'}")
        semaphore = asyncio.Semaphore(CONCURRENCY)
        totals = [0, 0, 0]

        async def one(number: int) -> None:
            async with semaphore:
                row = pages.loc[number].copy()
                row["page"] = number
                try:
                    record = await extract_page(llm, doc_id, row, processed, effort)
                except Exception as e:  # report and keep going with other pages
                    print(f"  p{number}: FAILED {e}")
                    return
                path = cache_path(processed, doc_id, number)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                totals[0] += record["prompt_tokens"]
                totals[1] += record["completion_tokens"]
                totals[2] += record["reasoning_tokens"]
                n = len(record["extraction"]["facts"])
                print(
                    f"  p{number}: {n} facts, {record['prompt_tokens']} in / "
                    f"{record['completion_tokens']} out tokens "
                    f"({record['reasoning_tokens']} reasoning), {record['seconds']}s"
                )

        await asyncio.gather(*(one(p) for p in todo))
        print(f"tokens: {totals[0]} input + {totals[1]} output (of which {totals[2]} reasoning)")

    facts, summaries = combine(processed)
    facts.to_parquet(processed / FACTS_FILE, index=False)
    summaries.to_parquet(processed / SUMMARIES_FILE, index=False)
    checks = facts["quote_check"].value_counts().to_dict() if len(facts) else {}
    print(
        f"wrote {len(facts)} facts from {len(summaries)} pages; quotes against the text "
        f"layer: {checks.get('exact', 0)} exact, {checks.get('tokens', 0)} tokens, "
        f"{checks.get('unverified', 0)} unverified"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract report facts with Muse Spark.")
    parser.add_argument("--doc", default="volve_pud", choices=sorted(REPORTS_BY_ID))
    parser.add_argument("--pages", default="all", help="e.g. 23,25-28 or all")
    parser.add_argument("--force", action="store_true", help="re-extract cached pages")
    args = parser.parse_args(argv)

    settings = get_settings()
    pages = pd.read_parquet(settings.processed_dir / PAGES_FILE)
    total = int(pages.loc[pages["doc_id"] == args.doc, "pages"].max())
    asyncio.run(run(args.doc, parse_pages(args.pages, total), args.force))


if __name__ == "__main__":
    main()
