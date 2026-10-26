"""Deterministic claim checker for the agent's final findings.

Before an answer is accepted:
- every number, ISO date and quoted passage in a finding must be found in the evidence
  that finding cites, so a quote from a report traces back through the evidence to the
  page it was read from;
- evidence counts only for what the data returned, never for what the model wrote: tool
  arguments, literals typed into a SQL query, and calculations on numbers that are not
  themselves in evidence are all excluded;
- a finding marked "fact" may not state a cause, nor rest on evidence that carries
  warnings (e.g. a poor decline fit), including warnings carried into a calculation;
- the summary's numbers, dates and quotes must exist in some evidence, and it may name a
  cause only with a hedge ("likely", "consistent with").
Problems go back to the model as plain sentences so it can fix them and resubmit.

Known limit: the sign of a change is not checked ("rose 45%" passes against -0.45).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.ingest.report_facts import squash

WELL = re.compile(r"\b(?:15/9-)?F-\d+(?:\s?[A-Z]\b)?")
DATE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?)\b")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
SPACED_THOUSANDS = re.compile(r"\b\d{1,3}(?:[ \u00a0\u202f]\d{3})+\b")
# A number may run straight into a unit ("98765Sm3") or end a sentence, but not run into
# more digits.
NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*(%|x\b|×|million\b|k\b))?(?!\d|\.\d)"
)
SQL_STRING = re.compile(r"'((?:[^']|'')*)'")
SCALES = {"%": (0.01, 1.0), "million": (1e6,), "k": (1e3,)}
SMALL_INTEGER = 10  # "3 wells", "2 years": too generic to demand evidence for
# Unit conversions a calculation may use without citing them.
CONSTANTS = {12.0, 24.0, 30.0, 30.44, 100.0, 1000.0, 365.0, 365.25}
# Causes are interpretations: the data shows timing and size, not why.
CAUSAL = re.compile(
    r"\b(because|caus(?:e|ed|es|ing)|dr(?:ove|ives?|iven|iving)|due to|led to|leads? to|"
    r"result(?:s|ed)? (?:from|in)|responsible for)\b",
    re.IGNORECASE,
)
# A quoted passage: opened after a space or "(", closed before a space or punctuation, so
# apostrophes inside words ("oil's") do not end it.
QUOTED = re.compile(r"""(?:(?<=[\s(])|^)['"‘“«](.{15,}?)['"’”»](?=[\s.,;:)/]|$)""")
MIN_QUOTE_CHARS = 12  # letters and digits; shorter quotes are phrases, not citations
HEDGE = re.compile(
    r"\b(likely|probabl[ey]|possibly|consistent with|suggests?|appears?|may|might|"
    r"hypothes\w*|indicat\w*)\b",
    re.IGNORECASE,
)


class Finding(BaseModel):
    claim: str = Field(description="One finding in one or two sentences, with its numbers.")
    evidence_ids: list[str] = Field(
        description="Ids (E1, E2, ...) of the evidence every number and date in the claim "
        "comes from."
    )
    confidence: Literal["fact", "inference"] = Field(
        description="'fact' if the cited evidence shows it directly; 'inference' for "
        "interpretations such as likely causes."
    )


class Submission(BaseModel):
    summary: str = Field(description="Two or three sentences answering the question.")
    findings: list[Finding] = Field(min_length=1)


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _flatten(v)
    else:
        yield value


def _numbers_in_text(text: str, european: bool = False) -> list[tuple[float, int, str | None]]:
    """(value, decimals, suffix) for each number, ignoring well names and ISO dates.

    european=True reads Norwegian report text: "9.000" is nine thousand, "0,80" is 0.8.
    """
    text = DATE.sub(" ", WELL.sub(" ", text))
    # "4 050" and "1 133 053": thousands grouped with a space (or a narrow no-break space).
    text = SPACED_THOUSANDS.sub(lambda m: re.sub(r"\s", "", m.group(0)), text)
    if european:
        text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
        text = re.sub(r"(?<=\d),(?=\d{1,2}\b)", ".", text)
    text = THOUSANDS.sub("", text)
    found = []
    for match in NUMBER.finditer(text):
        digits, suffix = match.group(1), match.group(2)
        mantissa = re.split(r"[eE]", digits)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        found.append((float(digits), decimals, suffix))
    return found


def _matches(candidates: Iterable[float], value: float, decimals: int, suffix: str | None) -> bool:
    """Does any candidate round to the claimed value at the precision it was written?"""
    pool = list(candidates)
    for scale in SCALES.get(suffix or "", (1.0,)):
        target = value * scale
        tolerance = 0.5 * 10**-decimals * scale + 1e-9
        if any(abs(n - target) <= tolerance for n in pool):
            return True
    return False


def _typed_by_model(item: Evidence) -> tuple[set[float], set[str]]:
    """Numbers and strings the model wrote into a SQL query: echoes, not data."""
    if item.tool != "run_sql":
        return set(), set()
    sql = str(item.arguments.get("sql", ""))
    strings = {s.replace("''", "'") for s in SQL_STRING.findall(sql)}
    numbers = {v for v, _, _ in _numbers_in_text(SQL_STRING.sub(" ", sql))}
    return numbers, strings


class _Pool:
    """Every number, date and text that a set of evidence results returned from the data."""

    def __init__(self, evidence: Iterable[Evidence], extra_numbers: Iterable[float] = ()) -> None:
        self.numbers: list[float] = list(extra_numbers)
        self.dates: set[str] = set()
        texts: list[str] = []
        for item in evidence:
            typed_numbers, typed_strings = _typed_by_model(item)
            for leaf in _flatten(item.result):
                if isinstance(leaf, bool) or leaf is None:
                    continue
                if isinstance(leaf, int | float):
                    if abs(float(leaf)) not in typed_numbers:
                        self.numbers.append(abs(float(leaf)))
                elif isinstance(leaf, str) and leaf not in typed_strings:
                    texts.append(squash(leaf))
                    self.dates.update(DATE.findall(leaf))
                    # Report text may use either convention, so keep both readings.
                    for european in (False, True):
                        self.numbers.extend(v for v, _, _ in _numbers_in_text(leaf, european))
        self.years = {d[:4] for d in self.dates}
        self.text = "|".join(texts)

    def has_number(self, value: float, decimals: int, suffix: str | None) -> bool:
        return _matches(self.numbers, value, decimals, suffix)

    def has_date(self, day: str) -> bool:
        """A day must appear as that day; a month matches any day of it, never the reverse."""
        if len(day) == 10:
            return day in self.dates
        return any(d.startswith(day) for d in self.dates)


@dataclass(frozen=True)
class _Calc:
    unsupported: tuple[str, ...]  # inputs found in no evidence
    warnings: tuple[str, ...]  # warnings of the evidence its inputs came from


def _calculations(log: EvidenceLog) -> dict[str, _Calc]:
    """Check each calculate call's inputs against the evidence before it.

    A calculation proves nothing about numbers the model typed in: every input must
    appear in earlier evidence (or an earlier sound calculation). If an input appears only
    in evidence with warnings, the calculation carries those warnings.
    """
    calcs: dict[str, _Calc] = {}
    earlier: list[Evidence] = []
    for item in log:
        if item.tool != "calculate":
            earlier.append(item)
            continue
        missing, warnings = [], []
        expression = str(item.arguments.get("expression", ""))
        for value, decimals, _ in _numbers_in_text(expression):
            if value in CONSTANTS or (decimals == 0 and value <= SMALL_INTEGER):
                continue
            holders = [e for e in earlier if _Pool([e]).has_number(value, decimals, None)]
            if not holders:
                missing.append(f"{value:g}")
            elif all(e.warnings or calcs.get(e.id, _Calc((), ())).warnings for e in holders):
                for e in holders:
                    warnings.extend(e.warnings or calcs[e.id].warnings)
        calcs[item.id] = _Calc(tuple(missing), tuple(dict.fromkeys(warnings)))
        if not missing:
            earlier.append(item)  # a sound result can feed later calculations
    return calcs


def _unsupported(text: str, pool: _Pool) -> tuple[list[str], list[str]]:
    numbers = []
    for value, decimals, suffix in _numbers_in_text(text):
        bare_integer = decimals == 0 and not suffix and value.is_integer()
        if bare_integer and 1900 <= value <= 2100:
            # A year, or a plain number such as 2095 days or 2000 Sm3/d.
            if str(int(value)) not in pool.years and not pool.has_number(value, 0, None):
                numbers.append(str(int(value)))
        elif decimals == 0 and not suffix and value <= SMALL_INTEGER:
            continue
        elif not pool.has_number(value, decimals, suffix):
            numbers.append(f"{value:g}{suffix or ''}")
    dates = [d for d in DATE.findall(text) if not pool.has_date(d)]
    return numbers, dates


def _missing_quotes(text: str, pool: _Pool) -> list[str]:
    quotes = [q for q in QUOTED.findall(text) if len(squash(q)) >= MIN_QUOTE_CHARS]
    return [q for q in quotes if squash(q) not in pool.text]


def check_submission(submission: Submission, log: EvidenceLog) -> list[str]:
    """Return problems as sentences for the model; an empty list means accepted."""
    calcs = _calculations(log)
    sound = [e for e in log if not (e.id in calcs and calcs[e.id].unsupported)]
    problems = []
    for n, finding in enumerate(submission.findings, start=1):
        if not finding.evidence_ids:
            problems.append(f"Finding {n} cites no evidence; cite the evidence ids it uses.")
            continue
        unknown = [e for e in finding.evidence_ids if log.get(e) is None]
        if unknown:
            problems.append(f"Finding {n} cites unknown evidence {', '.join(unknown)}.")
            continue
        cited = [log.get(e) for e in finding.evidence_ids]
        ids = ", ".join(finding.evidence_ids)
        for item in cited:
            if item.id in calcs and calcs[item.id].unsupported:
                problems.append(
                    f"Finding {n}: {item.id} calculates from "
                    f"{', '.join(calcs[item.id].unsupported)}, which no evidence contains. "
                    "Calculate only with numbers that tool results returned."
                )
        pool = _Pool(e for e in cited if not (e.id in calcs and calcs[e.id].unsupported))
        numbers, dates = _unsupported(finding.claim, pool)
        if numbers:
            problems.append(
                f"Finding {n}: {', '.join(numbers)} not found in {ids}. Cite the evidence it "
                "comes from; compute derived numbers with calculate; or remove them."
            )
        if dates:
            problems.append(f"Finding {n}: date(s) {', '.join(dates)} not found in {ids}.")
        missing_quotes = _missing_quotes(finding.claim, pool)
        if missing_quotes:
            listed = "; ".join(f"'{q[:80]}'" for q in missing_quotes)
            problems.append(
                f"Finding {n}: quoted text not found in {ids}: {listed}. Quote the evidence "
                "word for word (the original quote or its statement_en) or drop the quotes."
            )
        # A cause inside a quote is the report's claim, which the finding only reports.
        cause = CAUSAL.search(QUOTED.sub(" ", finding.claim))
        if finding.confidence == "fact" and cause:
            problems.append(
                f"Finding {n} states a cause ('{cause.group(0)}') but is marked fact. The data "
                "shows timing and size, not causes: mark it inference."
            )
        warned = {
            e.id: e.warnings or (calcs[e.id].warnings if e.id in calcs else ()) for e in cited
        }
        flagged = {k: v for k, v in warned.items() if v}
        if finding.confidence == "fact" and flagged:
            reasons = "; ".join(f"{k}: {w}" for k, ws in flagged.items() for w in ws)
            problems.append(
                f"Finding {n} is marked fact but rests on evidence with warnings ({reasons}). "
                "Mark it inference, or refit and cite the new evidence."
            )

    everything = _Pool(sound)
    numbers, dates = _unsupported(submission.summary, everything)
    missing = (
        numbers + dates + [f"'{q[:80]}'" for q in _missing_quotes(submission.summary, everything)]
    )
    if missing:
        problems.append(f"Summary: {', '.join(missing)} not found in any evidence.")
    cause = CAUSAL.search(QUOTED.sub(" ", submission.summary))
    if cause and not HEDGE.search(submission.summary):
        problems.append(
            f"Summary states a cause as certain ('{cause.group(0)}'). The data shows timing "
            "and size, not causes: hedge it, e.g. 'most likely because' or 'consistent with'."
        )
    return problems
