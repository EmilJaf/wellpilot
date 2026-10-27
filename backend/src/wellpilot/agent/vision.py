"""Look at a report page during a run: the agent shows a page image to the model.

Report facts are extracted once, at ingest. Some things live only in the picture: a
forecast curve, a log track, a scanned table. look_at_page sends the page image and the
agent's question to the same model (Muse Spark Open on Baseten) and returns its reading.
A reading off a picture is an estimate, not data, so a result carries a warning: the
checker then lets a finding cite it only as an inference, and the summary only with a
hedge. Where SAM 3.1 measured the figure, each reading is set against that measurement
(crosscheck.py); a page whose readings all agree with it carries none.
"""

import asyncio
import base64
import json

from pydantic import BaseModel, Field, ValidationError

from wellpilot.agent.crosscheck import MONTHS_TOLERANCE, RATE_TOLERANCE, cross_check, measured
from wellpilot.agent.tools import MAX_RESULT_CHARS, Args, Tool, to_jsonable
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.reports import REPORTS_BY_ID
from wellpilot.llm import LLM, json_schema

LOOK = "look_at_page"
TIMEOUT_S = 120.0
WARNING = (
    "read from the page image by the model, not from data: values are approximate readings "
    "(a figure's scale, a scan); treat conclusions from them as inferences"
)
SYSTEM = (
    "You read one page of an oil field report for an engineer. Answer the question from what "
    "the page shows, the picture included. Give each value you read with its label and unit, "
    "and say where it is: a figure, a table or the text. Read figures carefully off their axes "
    "and give only values you can see; if the page does not show it, say so and give no value."
)


class LookArgs(Args):
    doc_id: str = Field(description="Report id, e.g. 'volve_pud'.")
    page: int = Field(description="Page number.")
    question: str = Field(
        description="What to read off the page, e.g. 'the oil rate in Sm3/d the forecast "
        "figure shows at the start of each year'."
    )


class Reading(BaseModel):
    label: str = Field(description="What the value is, e.g. 'oil rate, start of 2010'.")
    value: float
    unit: str
    where: str = Field(description="'figure', 'table' or 'text'.")


class PageReading(BaseModel):
    answer: str = Field(description="The answer in one to three sentences, in English.")
    readings: list[Reading] = Field(description="Every value the answer uses.")


async def look_at_page(store: ProductionStore, llm: LLM, args: LookArgs) -> dict:
    """Show the model a report page image with the agent's question; return its reading."""
    if args.doc_id not in REPORTS_BY_ID:
        raise ValueError(f"unknown document {args.doc_id!r}; known: {', '.join(REPORTS_BY_ID)}")
    if "report_pages" not in store.tables:
        raise ValueError("no report pages are loaded")
    page = store.query(
        "SELECT pages, image FROM report_pages WHERE doc_id = ? AND page = ?",
        [args.doc_id, args.page],
    ).rows
    if page.empty:
        raise ValueError(f"{args.doc_id} has no page {args.page}")
    report = REPORTS_BY_ID[args.doc_id]
    image = base64.b64encode((store.processed_dir / page.loc[0, "image"]).read_bytes()).decode()
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{report.title}, issued {report.doc_date[:7]}, page {args.page} of "
                    f"{int(page.loc[0, 'pages'])}.\n\nQuestion: {args.question}",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        },
    ]
    response = await asyncio.wait_for(
        llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "page_reading", "schema": json_schema(PageReading)},
            },
            reasoning_effort="low",
        ),
        TIMEOUT_S,
    )
    try:
        reading = PageReading.model_validate_json(response.choices[0].message.content or "")
    except ValidationError as e:
        raise ValueError(f"the page reading was not valid: {str(e)[:300]}") from None
    readings = [r.model_dump() for r in reading.readings]
    result = {
        "document": report.title,
        "doc_id": args.doc_id,
        "page": args.page,
        "question": args.question,
        "answer": reading.answer,
        "readings": readings,
        "read_by": llm.model,
        "warnings": [WARNING],
    }
    # A figure SAM 3.1 measured: each reading set against the measurement of the same
    # pixels, within tolerances fixed by the figure's grid. Agreed on all, no warning.
    sam = measured(store.processed_dir, args.doc_id, args.page)
    if sam is not None:
        checks = cross_check(readings, sam)
        result["measured_by_sam"] = {
            "what": sam["what"],
            "plateau": sam["plateau"],
            "colour_check": sam["colour_check"],
            "tolerances": {"rate_sm3_per_day": RATE_TOLERANCE, "months": MONTHS_TOLERANCE},
        }
        result["cross_check"] = checks
        if checks and all(c["agree"] for c in checks):
            result["warnings"] = []
        else:
            off = [c["label"] for c in checks if c["agree"] is False]
            if off:
                result["warnings"] = [
                    WARNING,
                    f"SAM 3.1's measurement of the same figure disagrees on: {', '.join(off)}",
                ]
    return to_jsonable(result)


async def look(store: ProductionStore, llm: LLM, arguments: str) -> dict:
    """Run a look_at_page call from the model: {"result": ...} or {"error": ...}, like
    tools.execute."""
    try:
        args = LookArgs.model_validate_json(arguments or "{}")
    except ValidationError as e:
        return {"error": f"invalid arguments for {LOOK}: {e}"}
    try:
        result = await look_at_page(store, llm, args)
    except (ValueError, TimeoutError) as e:
        return {"error": str(e) or f"{LOOK} timed out"}
    except Exception as e:  # a provider error must cost the call, not the run
        return {"error": f"{LOOK} failed: {type(e).__name__}"}
    if len(json.dumps(result, ensure_ascii=False)) > MAX_RESULT_CHARS:
        return {"error": f"{LOOK} returned too much; ask a narrower question"}
    return {"result": result}


VISION_TOOL = Tool(
    LOOK,
    "Look at a report page yourself: the page image is shown to you with your question, for "
    "what only the picture holds, such as a figure's curve, a chart or a scanned table (the "
    "plan's oil forecast profile is a figure on PUD page 23; completion reports hold survey "
    "tables, time-depth charts and schematics as pictures). Values read off a picture are "
    "approximate and carry a warning. Where SAM 3.1 measured the figure (the plan's forecast "
    "curve), each reading is set against that measurement in cross_check: readings both "
    "agree on lose the warning; where they disagree, report SAM's measured value and say "
    "they differ.",
    LookArgs,
    look_at_page,
)
