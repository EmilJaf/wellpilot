"""The agent loop: plan, call tools, submit findings, pass the claim checker.

Every successful tool result is logged as evidence with an id. The model finishes by
calling submit_findings; the checker verifies each finding's numbers and dates against
the evidence it cites, and rejected submissions go back to the model to fix.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
import pandas as pd
from pydantic import ValidationError

from wellpilot.agent.checker import (
    DeclaredNumber,
    Finding,
    Submission,
    check_submission,
    display_spans,
    lookup_spans,
    readability,
)
from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.agent.review import Review, revise_message
from wellpilot.agent.sandbox import (
    MAX_CALLS_PER_RUN,
    PYTHON_TOOL,
    RUN_PYTHON,
    PythonFn,
    as_inference,
)
from wellpilot.agent.tools import TOOLS_BY_NAME, execute, tool_schemas
from wellpilot.agent.vision import LOOK, VISION_TOOL, look
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.reports import REPORTS_BY_ID
from wellpilot.llm import LLM, json_schema

logger = logging.getLogger(__name__)

# Reviews an accepted answer: (llm, question, answer, evidence) -> (review, tokens).
ReviewFn = Callable[[LLM, str, Submission, EvidenceLog], Awaitable[tuple[Review, int]]]

SUBMIT = "submit_findings"
MAX_STEPS = 14  # real runs use 5-9; the model is warned two steps before the end
MAX_CHECK_ROUNDS = 3
MAX_CALLS_PER_STEP = 8  # tool calls past this in one step get an error, not a result
PREVIEW_CHARS = 300
MAX_REASONING_CHARS = 6000  # Muse Spark Open's visible reasoning, per step, in the stream

SYSTEM_PROMPT = """\
You are WellPilot, an analysis agent for petroleum engineers working on the {field}. \
{wells} {reports}

How you work:
- Start with a short numbered plan, then call tools. Get every number from a tool; never \
invent or estimate a number you have not computed.
- Every successful tool result carries an evidence_id (E1, E2, ...).
- Check results before concluding. Tool warnings mean a result is not trustworthy as is: \
for a poor decline fit, refit on a window that starts after the last regime change. Before \
reading a change as reservoir behaviour, rule out data and operations (events show shut-ins, \
gauge failures and regime changes; well_operations shows work done on the well, such as a \
plug set to shut off water or a new zone perforated, which can change a rate overnight). \
When a well has a production log, production_log shows where along the well its oil and \
water came in: pass a plug's depth to see what the plug could shut off, and compare that \
with what the well did after. {drive} When a waterflooded well's water cut rises, check with \
waterflood whether liquid held while the choke opened (water taking the well's capacity) and \
how much oil the WOR trend leaves before the economic water cut; its field voidage \
replacement_ratio is at reservoir conditions (near 1 means injection kept pace). For a \
forecast's uncertainty, report the calibrated band (calibrated_band_sm3, from the errors of \
earlier forecasts made the same way), not the bootstrap band, which only shows the fit's \
scatter, and say so when the Arps models disagree (p50_by_model_sm3). A fit with a warning \
has no decline rate worth comparing: never divide one decline rate by another unless both \
fits are sound.
- Compute every derived number (ratios, differences, percent changes) with the calculate \
tool, so it has evidence too, all of them in one call ("expressions": [...]). A share needs \
no calculation to be written as a percent: declare 51.3% with the evidence's value 0.513. \
Call tools that do not depend on each other in the same step.
- Finish by calling submit_findings: a short summary and findings. Each finding cites the \
evidence ids its numbers and dates come from and is marked "fact" (shown directly by the \
evidence) or "inference" (an interpretation, such as a likely cause). A finding that \
rests on evidence with warnings must be an inference.
- Declare every number you write: in each finding's "numbers" (and the summary's \
"summary_numbers") list it with its text exactly as written, its full value, and the \
evidence id it comes from, e.g. {{"text": "4 050", "value": 4050, "evidence_id": "E3"}} \
or {{"text": "72%", "value": 0.72, "evidence_id": "E9"}}. Write dates as YYYY-MM-DD. \
Dates, years, well names, integers up to 10, window lengths ("30-day median", "the next 12 \
months") and numbers from the question need no declaration. Write amounts in digits, never \
in words ("55% lower", not "half"; "67%", not "two thirds"): words cannot be checked. A \
level against a limit is written as the level ("a water cut of 67%, under the 80% limit"), \
not as a difference ("67% below the limit"). A range runs low to high.
- Write for an engineer and their manager. The summary's first sentence answers the \
question in plain words a manager follows: what is wrong with what ("its downhole pressure \
gauge is dead; water is now most of what the well makes"), not the tools' labels (alarm, \
caution, step down, regime change) or engineering shorthand. A forecast in the summary \
gives its central estimate with the calibrated range next to it ("about 359,000 Sm3, range \
324,000 to 470,000"): the range says how sure it is better than words do. Round numbers to \
what a reader needs (0.45 per year, 72%, 1.13 million Sm3) and give their units; in the \
summary and the claims, at most 3 significant digits (0.45, not 0.4461; 359,000, not 359 \
007; 1.13 million, not 1,133,053), and say "central estimate" and "range", not P50, P90 \
and P10. Each finding's claim is one or two plain sentences with at most three numbers, \
all declared; the fit behind it (model, window, r2_log, di, b) and exact values go in its \
detail, copied as the tools give them: the detail's numbers need no declaring, but each \
must be in the evidence the finding cites. Keep fit statistics out of claims and the \
summary, and never write internal scores or field names. For a share such as water cut, \
give the values before and after, not a percent change. Translate every quote that is not \
in English, in brackets right after it. A summary number from evidence with warnings needs \
a hedge in its own sentence ("the fit suggests").
- A checker verifies every declared number, date and quote against the evidence, and that \
each number is the well, quantity, direction and event the text says. If it rejects your \
submission, fix the listed problems and submit again; when it offers a declaration for a \
true number, declare it rather than delete it.{as_of_note}"""

# What a text-only reply gets back depends on where the run is.
NUDGE_START = (
    "Continue by calling the tools in your plan; call submit_findings once every number you "
    "write has evidence."
)
NUDGE = (
    "Call a tool to continue your analysis, or submit your answer with the submit_findings "
    "tool, citing the evidence ids of the tool results each finding rests on."
)
NUDGE_FIX = "Fix the problems the checker listed and call submit_findings again."
SUBMIT_BY = 8  # a run with no submission by this step is told to submit, keeping steps to fix
SUBMIT_ONLY = 10  # still none: only submit_findings is offered until the checker answers
SUBMIT_ONLY_NOTE = (
    "Only submit_findings is available now: submit your answer with the evidence you have. "
    "When the checker replies, every tool is available again to fix what it lists."
)
SUBMIT_NOW = (
    "You have gathered evidence for several steps. Submit your findings now with "
    "submit_findings: the checker may send them back, and fixing takes steps too."
)
TWO_STEPS_LEFT = (
    "Two steps left, this one included: finish what you need and call submit_findings with "
    "the evidence you have."
)
LAST_STEP = "This is your last step: call submit_findings now with the evidence you have."
CUT_OFF = (
    "Your last reply was cut off by the output limit. Keep replies short: call tools, or "
    "submit your answer."
)
TEXT_SUBMISSION = (
    "Your reply held a submission as text, so it was checked as a submit_findings call. "
    "The checker's reply: {payload}. Send answers with the submit_findings tool."
)
FIELD_FILE = "field.json"  # optional, beside the processed data: {"name", "drive"}
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def submit_tool_schema() -> dict:
    parameters = json_schema(Submission)
    # Declared numbers are optional in the model only so old answers still load; the agent
    # must always send them.
    parameters["required"] = [*parameters["required"], "summary_numbers"]
    finding = parameters["properties"]["findings"]["items"]
    finding["required"] = [*finding["required"], "numbers"]
    return {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": "Submit the final answer: a summary and findings that cite "
            "evidence ids, with every number of the summary and the claims declared. A checker "
            "verifies them; fix any problems it returns. A finding looks like: "
            '{"claim": "F-12\'s oil rate fell 72% around 2010-07-18, to 1 146 Sm3/d.", '
            '"detail": "Weekly mean 4050.37 -> 1145.72 Sm3/d; exponential fit after it, '
            'r2_log 0.8732.", "evidence_ids": ["E3", "E7"], "confidence": "fact", "numbers": '
            '[{"text": "72%", "value": 0.7171, "evidence_id": "E3"}, {"text": "1 146", '
            '"value": 1146, "evidence_id": "E3"}]}. The detail\'s numbers are looked up in the '
            "cited evidence: copy them as the tools give them.",
            "parameters": parameters,
        },
    }


@dataclass
class Step:
    index: int
    content: str | None
    reasoning: str | None
    tool_calls: list[dict] = field(default_factory=list)  # name, arguments, result, evidence_id
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def to_event(self) -> dict:
        """A compact, JSON-ready view for streaming: results become short previews; full
        results stay in the evidence log."""
        calls = []
        for call in self.tool_calls:
            result = call["result"]
            view: dict = {"name": call["name"], "arguments": _parse_arguments(call["arguments"])}
            if call["name"] == SUBMIT:
                view["accepted"] = result.get("accepted", False)
                view["problems"] = result.get("problems", [])
                if call.get("as_text"):  # written as the reply, not called: replay skips it
                    view["as_text"] = True
            elif "error" in result:
                view["error"] = result["error"]
            else:
                inner = result.get("result")
                text = json.dumps(inner, ensure_ascii=False)
                view["evidence_id"] = call.get("evidence_id")
                view["warnings"] = inner.get("warnings", []) if isinstance(inner, dict) else []
                view["preview"] = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
            calls.append(view)
        reasoning = self.reasoning or None
        if reasoning and len(reasoning) > MAX_REASONING_CHARS:
            reasoning = reasoning[:MAX_REASONING_CHARS] + "…"
        return {
            "index": self.index,
            "content": self.content,
            "reasoning": reasoning,
            "tool_calls": calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class AgentRun:
    question: str
    summary: str | None
    findings: list[Finding]
    evidence: list[Evidence]
    check_rounds: list[list[str]]  # problems found at each submission; [] = accepted
    verified: bool
    finished: bool
    steps: list[Step]
    summary_numbers: list[DeclaredNumber] = field(default_factory=list)
    stopped: str | None = None  # why the run ended early: step or time limit, provider error
    # The second opinion: the reviewer's points, and whether the answer was revised after
    # them ("revised"), kept because the revision did not pass ("kept"), or not needed.
    review: dict | None = None

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.prompt_tokens for s in self.steps),
            sum(s.completion_tokens for s in self.steps),
        )

    def to_dict(self) -> dict:
        """The answer and an index of its evidence (results are served separately)."""
        tokens_in, tokens_out = self.total_tokens
        return with_spans(
            {
                "question": self.question,
                "summary": self.summary,
                "findings": [f.model_dump() for f in self.findings],
                "summary_numbers": [n.model_dump() for n in self.summary_numbers],
                "verified": self.verified,
                "finished": self.finished,
                "stopped": self.stopped,
                "check_rounds": self.check_rounds,
                "review": self.review,
                "steps": len(self.steps),
                "tokens": {"input": tokens_in, "output": tokens_out},
                "evidence": [
                    {k: v for k, v in asdict(e).items() if k != "result"} for e in self.evidence
                ],
            },
            EvidenceLog.restore(asdict(e) for e in self.evidence),
        )


def with_spans(result: dict, log: EvidenceLog | None = None) -> dict:
    """Add the display spans of the summary and each finding to a result (also to one saved
    before spans existed, so replays mark numbers by the same rules). With the evidence log,
    each quote also names the evidence it was found in."""

    def numbers(raw: list | None) -> list[DeclaredNumber]:
        return [DeclaredNumber.model_validate(n) for n in raw or []]

    result["summary_spans"] = display_spans(
        result.get("summary"), numbers(result.get("summary_numbers")), log
    )
    for finding in result.get("findings", []):
        cited = None
        if log is not None:  # a finding's quote is looked for in what it cites
            ids = set(finding.get("evidence_ids", []))
            cited = EvidenceLog.restore(asdict(e) for e in log if e.id in ids)
        declared = numbers(finding.get("numbers"))
        finding["spans"] = display_spans(finding.get("claim"), declared, cited)
        finding["detail_spans"] = lookup_spans(finding.get("detail"), cited)
    return result


def _reasoning(message: Any) -> str | None:
    """Providers return reasoning under different non-standard fields, if at all."""
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if extra.get(key):
            return str(extra[key])
    return None


def describe_wells(store: ProductionStore) -> str:
    """The wells as the data shows them, so a run with an as-of date learns nothing about
    later years (not which wells come later, nor when the field stops)."""
    rows = store.query(
        """
        SELECT well,
               min(date) FILTER (WHERE oil_sm3 > 0) AS first_oil,
               bool_or(water_inj_sm3 > 0) AS injects,
               max(date) AS last_date
        FROM production_daily GROUP BY well ORDER BY well
        """
    ).rows
    if rows.empty:
        return "No production data is available."
    producers = [
        f"{r.well} (first oil {r.first_oil.date().isoformat()})"
        for r in rows.itertuples()
        if r.first_oil is not None and not pd.isna(r.first_oil)
    ]
    injectors = [r.well for r in rows.itertuples() if r.injects]
    last = rows["last_date"].max().date().isoformat()
    return (
        f"Producers in the data: {', '.join(producers) or 'none yet'}. "
        f"Water injectors: {', '.join(injectors) or 'none yet'}. The data ends on {last}."
    )


def describe_reports(store: ProductionStore, note: str = "", unnamed: bool = False) -> str:
    """The reports the store holds (an as-of store hides those issued later), and how to
    use them; a field without reports is told so. unnamed: titles without the field's name."""
    if "report_pages" not in store.tables:
        return "No reports are indexed: the report and plan tools have nothing to read."
    ids = store.query("SELECT DISTINCT doc_id FROM report_pages ORDER BY doc_id DESC").rows
    if ids.empty:
        return "No reports are indexed yet."
    language = {"no": "Norwegian", "en": "English"}
    listed = "; ".join(
        f"{r.title.replace('Volve', 'the field') if unnamed else r.title} (doc_id {r.doc_id}, "
        f"{r.doc_date[:7]}, {language.get(r.language, r.language)})"
        for r in (REPORTS_BY_ID[d] for d in ids["doc_id"])
    )
    return f"Reports are indexed too: {listed}. {note}".strip()


VOLVE = {
    "name": "Volve oil field (Norwegian North Sea, operated by Equinor)",
    "drive": "Volve is a waterflood.",
    "reports_note": (
        "Use search_reports to compare what happened with what was planned (plan_vs_actual "
        "gives the field's production against the plan's forecast, by calendar year and by "
        "year of production, since the plan assumed first oil a year before the field "
        "started; field_vs_design gives the field against the plan's design limits), and "
        "quote the original wording with an English translation. Prefer facts whose "
        'quote_check is "exact". Facts were extracted from the text and tables; to read a '
        "figure, such as the plan's forecast curve, look at the page yourself with "
        "look_at_page. The plan's limits and capacities are for the whole field and "
        "platform: compare them with field totals from field_vs_design, never with one "
        "well's numbers. The maximum water cut is a share at full liquid rate: the field "
        "reaching it means it ran at the design water cut; only a rate over its capacity "
        "(water handling, liquid, oil, injection) means it exceeded the design."
    ),
}


def describe_field(store: ProductionStore) -> dict:
    """What the prompt says about the field: field.json beside the processed data (a
    synthetic field, another operator's), else Volve."""
    path = store.processed_dir / FIELD_FILE
    return {**VOLVE, **json.loads(path.read_text())} if path.exists() else VOLVE


# A question asked as of a past date names no field and no operator: the field is a public
# dataset a model may have read about, and what it read about the years after the date
# must not stand in for the data. (Well names and the plan still identify it to a model that
# knows it; memory_probe measures what the model knows without tools, and the synthetic
# field is a test it cannot have read about.)
UNNAMED = {"name": "field in this data", "drive": "The field is a waterflood."}


def system_prompt(store: ProductionStore) -> str:
    as_of = store.as_of
    note = f"\n\nData is available only up to {as_of.isoformat()}." if as_of else ""
    field = describe_field(store)
    unnamed = as_of is not None and field["name"] == VOLVE["name"]
    shown = {**field, **UNNAMED} if unnamed else field
    return SYSTEM_PROMPT.format(
        field=shown["name"],
        drive=shown["drive"],
        wells=describe_wells(store),
        reports=describe_reports(store, field.get("reports_note", ""), unnamed),
        as_of_note=note,
    )


def _parse_arguments(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def _text_submission(content: str | None) -> str | None:
    """The JSON of a submission the model wrote as its reply instead of calling
    submit_findings (Muse Spark Open does this now and then late in a run), or None."""
    text = FENCE.sub("", (content or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and {"summary", "findings"} <= parsed.keys():
        return text[start : end + 1]
    return None


def _nudge(log: EvidenceLog, last_submission: Submission | None) -> str:
    if last_submission:
        return NUDGE_FIX
    return NUDGE if len(log) else NUDGE_START


async def run_agent(
    question: str,
    llm: LLM,
    store: ProductionStore,
    *,
    max_steps: int = MAX_STEPS,
    reasoning_effort: str = "",
    timeout_s: float | None = None,
    on_step: Callable[[Step], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    evidence_log: EvidenceLog | None = None,
    look_fn: Callable[[ProductionStore, LLM, str], Awaitable[dict]] = look,
    review_fn: ReviewFn | None = None,
    python_fn: PythonFn | None = None,
) -> AgentRun:
    """Run the agent on one question.

    on_step gets each finished Step; on_event gets JSON-ready events for streaming:
    {"type": "llm_call", "step": n} before each model call, {"type": "step", ...} after.
    Pass evidence_log to read evidence while the run is still going. timeout_s bounds the
    whole run, including a model call that hangs. look_fn runs look_at_page calls (the
    replay passes recorded readings instead of asking the model again). review_fn, when
    given, reviews the first accepted answer; with points raised, the model gets one chance
    to revise, and the checked answer stands if the revision does not pass. python_fn, when
    given, runs run_python calls in the run's sandbox (the replay passes the recorded
    results); without it the tool is not offered.

    The run ends with an answer whatever happens to it: out of steps or time, or a provider
    error after the client's retries, it returns the last submission (unverified) with the
    reason in `stopped`, and its steps and evidence intact.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(store)},
        {"role": "user", "content": question},
    ]
    tools = [
        *tool_schemas(),
        VISION_TOOL.schema(),
        *([PYTHON_TOOL.schema()] if python_fn else []),
        submit_tool_schema(),
    ]
    python_calls = 0
    log = evidence_log if evidence_log is not None else EvidenceLog()
    steps: list[Step] = []
    check_rounds: list[list[str]] = []
    last_submission: Submission | None = None
    asked_readability = False  # rounded numbers and plain labels, asked once per run
    reviewed: dict | None = None  # the second opinion, once given
    checked: Submission | None = None  # the accepted answer the review was given on
    text_replies = 0  # consecutive replies without a tool call
    deadline = time.monotonic() + timeout_s if timeout_s else None

    def finish(
        summary, findings, verified, finished=True, submission=None, stopped=None
    ) -> AgentRun:
        numbers = (submission.summary_numbers or []) if submission else []
        return AgentRun(
            question,
            summary,
            findings,
            list(log),
            check_rounds,
            verified,
            finished,
            steps,
            numbers,
            stopped,
            reviewed,
        )

    def unrevised(**kwargs) -> AgentRun | None:
        """After a review, any ending without a new accepted answer keeps the checked one."""
        if checked is None:
            return None
        reviewed["outcome"] = "kept"
        return finish(checked.summary, checked.findings, True, submission=checked, **kwargs)

    def stop(reason: str) -> AgentRun:
        """End early, keeping the last answer the model submitted, marked unverified."""
        if kept := unrevised():
            return kept
        if last_submission:
            return finish(
                last_submission.summary,
                last_submission.findings,
                False,
                finished=False,
                submission=last_submission,
                stopped=reason,
            )
        return finish(None, [], False, finished=False, stopped=reason)

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    effort = reasoning_effort or llm.default_reasoning

    for index in range(1, max_steps + 1):
        if index == max_steps:
            messages.append({"role": "user", "content": LAST_STEP})
        elif index == max_steps - 1:
            messages.append({"role": "user", "content": TWO_STEPS_LEFT})
        elif index == SUBMIT_BY and not check_rounds and max_steps > SUBMIT_BY + 2:
            messages.append({"role": "user", "content": SUBMIT_NOW})
        # Calculating one number per step can use every step; past SUBMIT_ONLY without an
        # answer the model may only submit, which leaves steps for the checker's corrections.
        forced = index >= SUBMIT_ONLY and not check_rounds and max_steps > SUBMIT_ONLY + 1
        if forced and index == SUBMIT_ONLY:
            messages.append({"role": "user", "content": SUBMIT_ONLY_NOTE})
        emit({"type": "llm_call", "step": index})
        started = time.monotonic()
        try:
            remaining = deadline - started if deadline else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            response = await asyncio.wait_for(
                llm.client.chat.completions.create(
                    model=llm.model,
                    messages=messages,
                    tools=[submit_tool_schema()] if forced else tools,
                    **({"reasoning_effort": effort} if effort else {}),
                ),
                remaining,
            )
        except TimeoutError:
            return stop(f"time limit of {timeout_s:.0f} s")
        except openai.APIError as e:  # after the client's own retries
            logger.warning("model call failed at step %d: %s", index, e)
            status = getattr(e, "status_code", None)
            return stop(
                f"model provider error ({type(e).__name__}{f' {status}' if status else ''})"
            )
        if not response.choices:
            return stop("model provider returned no reply")
        choice = response.choices[0]
        message = choice.message
        usage = response.usage
        step = Step(
            index=index,
            content=message.content,
            reasoning=_reasoning(message),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=time.monotonic() - started,
        )

        # Content must be a string: providers reject an assistant message with null content
        # and no tool calls.
        assistant: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        messages.append(assistant)

        accepted: Submission | None = None

        def check(arguments: str) -> dict:
            nonlocal accepted, last_submission, asked_readability
            try:
                submission = Submission.model_validate_json(arguments or "{}")
            except ValidationError as e:
                return {"accepted": False, "problems": [f"invalid submission: {e}"]}
            last_submission = submission
            try:
                problems = check_submission(submission, log, question)
            except Exception as e:  # a checker bug must cost a round, not the run
                logger.exception("checker failed on a submission")
                problems = [
                    f"The checker could not read this submission ({type(e).__name__}). "
                    "Write every number in plain decimal form and submit again."
                ]
            # Readability is asked once: a second request could cost the answer its rounds.
            if not asked_readability and (style := readability(submission)):
                asked_readability = True
                problems = [*problems, *style]
            check_rounds.append(problems)
            if not problems:
                accepted = submission
            return {"accepted": not problems, "problems": problems}

        calls = 0
        for call in message.tool_calls or []:
            name, arguments = call.function.name, call.function.arguments
            record: dict = {"name": name, "arguments": arguments}
            if name == SUBMIT:
                payload = check(arguments)
            elif (calls := calls + 1) > MAX_CALLS_PER_STEP:
                payload = {
                    "error": f"too many tool calls in one step; call at most {MAX_CALLS_PER_STEP}"
                }
            else:
                if name == LOOK:  # asks the model about a page image: async, not a thread
                    payload = await look_fn(store, llm, arguments)
                elif name == RUN_PYTHON and python_fn:
                    python_calls += 1
                    payload = (
                        await python_fn(arguments)
                        if python_calls <= MAX_CALLS_PER_RUN
                        else {
                            "error": f"{RUN_PYTHON} is limited to {MAX_CALLS_PER_RUN} calls per "
                            "question; answer with the evidence you have"
                        }
                    )
                    if isinstance(payload.get("result"), dict):  # recorded ones too
                        payload = {**payload, "result": as_inference(payload["result"])}
                else:
                    # Tools are synchronous (DuckDB, scipy); keep the event loop free.
                    payload = await asyncio.to_thread(execute, store, name, arguments)
                if "result" in payload:
                    evidence = log.add(
                        name,
                        _parse_arguments(arguments),
                        payload["result"],
                        VISION_TOOL.run
                        if name == LOOK
                        else PYTHON_TOOL.run
                        if name == RUN_PYTHON
                        else TOOLS_BY_NAME[name].run,
                    )
                    payload = {"evidence_id": evidence.id, **payload}
                    record["evidence_id"] = evidence.id
            record["result"] = payload
            step.tool_calls.append(record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                }
            )
        # A submission written as the reply instead of a call: checked the same way, so a
        # model that forgets the tool still gets an answer checked and its problems back.
        written = None if message.tool_calls else _text_submission(message.content)
        if written:
            payload = check(written)
            step.tool_calls.append(
                {"name": SUBMIT, "arguments": written, "result": payload, "as_text": True}
            )
            if not accepted:
                reply = json.dumps(payload, ensure_ascii=False, default=str)
                messages.append({"role": "user", "content": TEXT_SUBMISSION.format(payload=reply)})

        steps.append(step)
        if on_step:
            on_step(step)
        emit({"type": "step", **step.to_event()})

        if accepted:
            # A second opinion on the first accepted answer, while steps remain to act on
            # it; points raised give the model one chance to revise.
            if review_fn is not None and reviewed is None and index <= max_steps - 3:
                try:
                    result, tokens = await review_fn(llm, question, accepted, log)
                except Exception as e:  # a failed review must not cost the checked answer
                    logger.warning("review failed: %s", type(e).__name__)
                    result, tokens = Review(verdict="sound", points=[]), 0
                reviewed = {**result.model_dump(), "tokens": tokens, "after_step": index}
                emit({"type": "review", **reviewed})
                if result.points:
                    checked, accepted = accepted, None
                    reviewed["outcome"] = "asked"
                    messages.append({"role": "user", "content": revise_message(result)})
                    continue
                reviewed["outcome"] = "sound"
            elif reviewed is not None and checked is not None:
                reviewed["outcome"] = "revised"
            return finish(accepted.summary, accepted.findings, True, submission=accepted)
        if last_submission and len(check_rounds) >= MAX_CHECK_ROUNDS + (checked is not None):
            # Out of correction rounds: return the last answer, marked unverified (or the
            # checked one, if this was a revision after a review).
            if kept := unrevised():
                return kept
            return finish(
                last_submission.summary,
                last_submission.findings,
                False,
                submission=last_submission,
            )
        if message.tool_calls or written:
            text_replies = 0
        else:
            text_replies += 1
            if text_replies >= 2:  # two in a row: the model will not call tools again
                if kept := unrevised():
                    return kept
                if last_submission:
                    return finish(
                        last_submission.summary,
                        last_submission.findings,
                        False,
                        submission=last_submission,
                    )
                return finish(message.content, [], verified=False)
            messages.append({"role": "user", "content": _nudge(log, last_submission)})
        if choice.finish_reason == "length":
            messages.append({"role": "user", "content": CUT_OFF})

    return stop("step limit")
