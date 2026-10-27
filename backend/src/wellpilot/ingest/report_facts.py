"""Extract facts from report pages with Muse Spark vision.

Muse Spark reads each page image (tables and figures included) together with the page's
text layer and returns structured facts: an English statement, the verbatim quote in the
original language, topic and wells. Each quote is then checked against the page's text
layer, so a fact the model invented shows up as unverified.

Each page's extraction is then reviewed against the page, deterministically, the way the
agent's answers are checked: quotes that are not on the page word for word, numbers on the
page that no fact covers (a table row left out), a page of text with no facts. Problems go
back to the model, which returns a corrected extraction; up to MAX_REVIEW_ROUNDS rounds,
keeping the extraction with the fewest problems.

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
from contextlib import suppress
from pathlib import Path
from typing import Literal

import pandas as pd
import pypdfium2 as pdfium
from pydantic import BaseModel, Field, ValidationError

from wellpilot.config import get_settings
from wellpilot.ingest.reports import PAGES_FILE, REPORTS_BY_ID, normalize_text, page_dir
from wellpilot.llm import LLM, json_schema, make_llm

FACTS_FILE = "report_facts.parquet"
SUMMARIES_FILE = "report_summaries.parquet"
TEXT_LAYER_CHARS = 8000
CONCURRENCY = 4
MAX_REVIEW_ROUNDS = 2  # corrections after the first extraction
MIN_UNCOVERED = 3  # fewer uncovered numbers than this are prose asides, not a missed table
EMPTY_PAGE_CHARS = 1000  # a page with this much text and no facts is asked about once

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

Extract the facts a petroleum engineer or a project reviewer would cite: production \
forecasts and plateau rates, water breakthrough and water production, injection strategy, \
pressures, reserves, facility limits (liquid, oil, gas, injection capacity, maximum water \
cut), well design and completion, drilling, schedules, costs and economics. For a table, \
extract every row as one fact whose statement gives all the row's values, with the row as \
its quote.

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


def _readings(token: str) -> set[float]:
    """A number token read both ways: "1,25" is 1.25 (Norwegian) or 125 (English
    thousands); "9.000" is 9.0 or 9000."""
    candidates = {
        token.replace(",", "."),  # one decimal comma
        token.replace(".", "").replace(",", "."),  # Norwegian thousands and decimals
        token.replace(",", ""),  # English thousands
        token.replace(",", "").replace(".", ""),  # every separator a thousands group
    }
    values = set()
    for c in candidates:
        with suppress(ValueError):
            values.add(round(float(c), 6))
    return values


def _is_year(token: str) -> bool:
    return token.isdigit() and 1900 <= int(token) <= 2100


# Numbers that are names, not values: section numbers ("4.5.1", "12.1"; the PUD writes
# decimals with a comma), and numbers after a word that makes them a reference or a label.
SECTION = re.compile(r"^\d{1,2}(?:\.\d{1,2})+$")
DOT_LEADER = re.compile(r"(?:\.\s?){4,}\d+")
REFERENCE_WORDS = {"tabell", "tab", "figur", "fig", "kapittel", "kapitel", "kap", "side"}
REFERENCE_WORDS |= {"vedlegg", "table", "figure", "chapter", "section", "appendix", "page"}
LABEL_WORDS = {"p", "pl", "f", "st"}  # P90, licence PL 046, well F-12, survey ST 0202
NAMING_WORDS = REFERENCE_WORDS | LABEL_WORDS


def _a_name(tokens: list[str], i: int) -> bool:
    token = tokens[i]
    before = tokens[i - 1] if i else ""
    after = tokens[i + 1 : i + 3]
    return (
        bool(SECTION.match(token))
        or before in NAMING_WORDS
        or (token == "15" and after[:1] == ["9"])  # 15/9-F-12
        or (token == "9" and before == "15")
    )


def uncovered_numbers(text: str, facts: list[dict], page: int) -> list[str]:
    """Numbers on the page that no fact's quote or statement holds, each shown in a few
    words of context. Years, small integers, the page number and chart axes (four or more
    evenly spaced numbers in a row) are not counted."""
    # A contents page's page numbers stand after dot leaders ("Hydrater......28").
    tokens = TOKEN.findall(_clean(DOT_LEADER.sub(" ", text)))
    covered: set[float] = set()
    for fact in facts:
        for t in TOKEN.findall(_clean(f"{fact['quote']} {fact['statement_en']}")):
            if t[0].isdigit():
                covered |= _readings(t)
    numeric = [i for i, t in enumerate(tokens) if t[0].isdigit()]
    axis: set[int] = set()
    run: list[int] = []
    for i in [*numeric, -2]:  # -2 closes the last run
        if run and i == run[-1] + 1:
            run.append(i)
            continue
        values = [min(_readings(tokens[j]) or {0.0}) for j in run]
        steps = {round(b - a, 6) for a, b in zip(values, values[1:], strict=False)}
        if len(run) >= 4 and len(steps) == 1 and steps != {0.0}:
            axis.update(run)
        run = [i]
    missed = [
        i
        for i in numeric
        if i not in axis
        and not _is_year(tokens[i])
        and not (tokens[i].isdigit() and (int(tokens[i]) <= 10 or int(tokens[i]) == page))
        and not _a_name(tokens, i)
        and not (_readings(tokens[i]) & covered)
    ]
    snippets: list[tuple[int, int]] = []
    for i in missed:
        start, end = max(0, i - 4), min(len(tokens), i + 5)
        if snippets and start <= snippets[-1][1]:
            snippets[-1] = (snippets[-1][0], end)
        else:
            snippets.append((start, end))
    return [" ".join(tokens[a:b]) for a, b in snippets] if len(missed) >= MIN_UNCOVERED else []


def review(extraction: "PageExtraction", text: str, page: int) -> list[str]:
    """Problems with one page's extraction, as sentences for the model; [] means none."""
    facts = [f.model_dump() for f in extraction.facts]
    problems = [
        f"Fact {n}: the quote is not on the page word for word: '{f['quote'][:160]}'. "
        "Copy it character for character from the text layer, or drop the fact."
        for n, f in enumerate(facts, start=1)
        if quote_check(f["quote"], text) == "unverified"
    ]
    missed = uncovered_numbers(text, facts, page)
    if missed:
        shown = "; ".join(f"«{m}»" for m in missed[:12])
        problems.append(
            f"These numbers on the page are in no fact (shown in context): {shown}. Add a "
            "fact for each one an engineer or reviewer would cite: every row of a table, "
            "every rate, volume, pressure, date or cost. Leave out chart axis labels."
        )
    elif not facts and len(text) >= EMPTY_PAGE_CHARS:
        problems.append(
            f"The page has {len(text)} characters of text and no facts. If any of it would "
            "be cited (decisions, methods, dates, numbers), extract it; if nothing would, "
            "return an empty fact list again."
        )
    return problems


def _fingerprint() -> str:
    """Of the prompt and output schema: a cached page made with others is stale."""
    spec = SYSTEM_PROMPT + json.dumps(json_schema(PageExtraction), sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:12]


def locate_quote(page: "pdfium.PdfPage", quote: str) -> list[list[float]]:
    """Where a quote stands on a PDF page, one rectangle per line: [x, y, width, height] as
    fractions of the page, top-left origin. Matched like quote_in_text (letters and digits
    only, case and the dropped glyphs ignored), so a quote found "exact" can be shown.
    Empty when the quote is not on the page as written."""
    target = squash(quote)
    if len(target) < 6:
        return []
    textpage = page.get_textpage()
    kept: list[tuple[str, int]] = []  # each squashed character and its index on the page
    for i in range(textpage.count_chars()):
        piece = re.sub(r"\W+", "", DROPPED_GLYPHS.sub("", textpage.get_text_range(i, 1)).casefold())
        kept.extend((ch, i) for ch in piece)
    # The 3 of "Sm3" is dropped on both sides (it is a superscript in the PDF).
    kept = [
        (ch, i)
        for k, (ch, i) in enumerate(kept)
        if not (
            ch == "3"
            and k >= 2
            and kept[k - 2][0] + kept[k - 1][0] == "sm"
            and not (k + 1 < len(kept) and kept[k + 1][0].isdigit())
        )
    ]
    at = "".join(ch for ch, _ in kept).find(target)
    if at < 0:
        return []
    width, height = page.get_size()
    lines: list[list[float]] = []  # left, bottom, right, top in PDF points
    for i in sorted({i for _, i in kept[at : at + len(target)]}):
        left, bottom, right, top = textpage.get_charbox(i, loose=True)
        if right <= left or top <= bottom:
            continue
        line = next((b for b in lines if abs(b[1] - bottom) < (top - bottom) * 0.5), None)
        if line is None:
            lines.append([left, bottom, right, top])
        else:
            line[:] = [
                min(line[0], left),
                min(line[1], bottom),
                max(line[2], right),
                max(line[3], top),
            ]
    return [
        [
            round(left / width, 4),
            round(1 - top / height, 4),
            round((right - left) / width, 4),
            round((top - bottom) / height, 4),
        ]
        for left, bottom, right, top in lines
    ]


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
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}

    async def ask() -> tuple[PageExtraction, str]:
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
            tokens["prompt"] += usage.prompt_tokens if usage else 0
            tokens["completion"] += usage.completion_tokens if usage else 0
            tokens["reasoning"] += (details.reasoning_tokens or 0) if details else 0
            content = response.choices[0].message.content or ""
            try:
                return PageExtraction.model_validate_json(content), content
            except ValidationError as e:
                errors.append(str(e)[:500])
        raise ValueError(f"page {page['page']}: no valid extraction: {errors}")

    # Extract, review against the page, send problems back; keep the best extraction.
    text = page["text"] if page["has_text"] else ""
    rounds: list[list[str]] = []
    best: tuple[int, PageExtraction] | None = None
    for attempt in range(1 + MAX_REVIEW_ROUNDS):
        extraction, content = await ask()
        problems = review(extraction, text, int(page["page"])) if text else []
        rounds.append(problems)
        weight = len(problems)
        if best is None or weight <= best[0]:
            best = (weight, extraction)
        if not problems or attempt == MAX_REVIEW_ROUNDS:
            break
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "A check of your extraction against the page found:\n- "
                + "\n- ".join(problems)
                + "\nReturn the whole corrected extraction: every fact, fixed and added.",
            }
        )
    return {
        "doc_id": doc_id,
        "page": int(page["page"]),
        "model": llm.model,
        "prompt_tokens": tokens["prompt"],
        "completion_tokens": tokens["completion"],
        "reasoning_tokens": tokens["reasoning"],
        "reasoning_effort": reasoning_effort or "default",
        "seconds": round(time.monotonic() - started, 1),
        "fingerprint": _fingerprint(),
        "review_rounds": rounds,
        "extraction": best[1].model_dump(),
    }


def combine(processed_dir: Path, raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge every cached page of every document into facts and summaries tables, checking
    each quote against its page's text layer and, when the PDF is at hand, locating it on
    the page so the UI can mark it on the page image."""
    pages = pd.read_parquet(processed_dir / PAGES_FILE).set_index(["doc_id", "page"])
    pdfs: dict[str, pdfium.PdfDocument] = {}
    if raw_dir is not None:
        for report in REPORTS_BY_ID.values():
            if (raw_dir / "reports" / report.file).exists():
                pdfs[report.doc_id] = pdfium.PdfDocument(raw_dir / "reports" / report.file)
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
                    "quote_boxes": json.dumps(
                        locate_quote(pdfs[record["doc_id"]][record["page"] - 1], fact["quote"])
                        if check == "exact" and record["doc_id"] in pdfs
                        else []
                    ),
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
                rounds = record["review_rounds"]
                review_note = (
                    f", reviewed {len(rounds)}x, {len(rounds[-1])} problems left"
                    if len(rounds) > 1
                    else ""
                )
                print(
                    f"  p{number}: {n} facts, {record['prompt_tokens']} in / "
                    f"{record['completion_tokens']} out tokens "
                    f"({record['reasoning_tokens']} reasoning), {record['seconds']}s{review_note}"
                )

        await asyncio.gather(*(one(p) for p in todo))
        print(f"tokens: {totals[0]} input + {totals[1]} output (of which {totals[2]} reasoning)")

    facts, summaries = combine(processed, settings.raw_dir)
    facts.to_parquet(processed / FACTS_FILE, index=False)
    summaries.to_parquet(processed / SUMMARIES_FILE, index=False)
    checks = facts["quote_check"].value_counts().to_dict() if len(facts) else {}
    print(
        f"wrote {len(facts)} facts from {len(summaries)} pages; quotes against the text "
        f"layer: {checks.get('exact', 0)} exact, {checks.get('tokens', 0)} tokens, "
        f"{checks.get('unverified', 0)} unverified"
    )
    from wellpilot.ingest.facts_gate import check  # the gate reads what was just written

    problems = check(processed)
    print("gate passed" if not problems else "GATE FAILED:\n- " + "\n- ".join(problems))


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
