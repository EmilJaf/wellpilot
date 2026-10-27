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

Numbers are declared, not parsed: each finding lists every number it writes with its
value and evidence id (DeclaredNumber); the written number must show that value and be
found in that evidence at the precision it was written, and any digits left in the text
undeclared are rejected. Answers from before declared
numbers (numbers=None) fall back to reading numbers out of the text.

Known limit: the sign of a change is not checked ("rose 45%" passes against -0.45).
"""

import json
import math
import re
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field

from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.ingest.report_facts import squash
from wellpilot.ingest.reports import REPORTS

WELL = re.compile(r"\b(?:15/9-)?F-\d+(?:\s?[A-Z]\b)?")
DATE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?)\b")
# English month names, capitalised ("May 2010" is a month, "may" is not), with a year.
_MONTH = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}\s+(\d{{4}})\b")
MONTH_DAY_YEAR = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
MONTH_YEAR = re.compile(rf"\b{_MONTH}\s+(\d{{4}})\b")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
SPACED_THOUSANDS = re.compile(r"\b\d{1,3}(?:[ \u00a0\u202f]\d{3})+\b")
# A number may run straight into a unit ("98765Sm3") or end a sentence, but not run into
# more digits.
# Suffixes that scale a number: "4.58 MSm3", "276 kSm3", "1.2 mill Sm3", "75.6 percent".
NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s*(%|percent\b|pct\b|x\b|×|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm)|k\b))?"
    r"(?!\d|\.\d)"
)
SUFFIX_AFTER = re.compile(
    r"\s*(?:%|percent\b|pct\b|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm))"
)
SQL_STRING = re.compile(r"'((?:[^']|'')*)'")
SCALES = {
    "%": (0.01, 1.0),
    "percent": (0.01, 1.0),
    "pct": (0.01, 1.0),
    "million": (1e6,),
    "mill": (1e6,),
    "mill.": (1e6,),
    "M": (1e6,),
    "thousand": (1e3,),
    "k": (1e3,),
}
SMALL_INTEGER = 10  # "3 wells", "2 years": too generic to demand evidence for
# Years and months of the indexed reports: the system prompt names "the February 2005
# development plan", so the model may too, citing data evidence only.
REPORT_YEARS = {float(r.doc_date[:4]) for r in REPORTS}
REPORT_DATES = {r.doc_date for r in REPORTS} | {r.doc_date[:7] for r in REPORTS}


def normalize_dates(text: str) -> str:
    """Dates written with month names as ISO dates, so they are checked as dates: "12
    October 2010" and "October 12, 2010" become 2010-10-12, "Oct 2010" becomes 2010-10.
    Impossible days ("31 June 2010") are left as they are. "Mid-2010" stays a year."""

    def iso(year: str, month: str, day: str | None) -> str | None:
        number = [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ].index(month[:3].lower()) + 1
        if day is None:
            return f"{year}-{number:02d}"
        try:
            return date(int(year), number, int(day)).isoformat()
        except ValueError:
            return None

    text = DAY_MONTH_YEAR.sub(lambda m: iso(m[3], m[2], m[1]) or m[0], text)
    text = MONTH_DAY_YEAR.sub(lambda m: iso(m[3], m[1], m[2]) or m[0], text)
    return MONTH_YEAR.sub(lambda m: iso(m[2], m[1], None) or m[0], text)


def _normalize(text: str) -> str:
    """The claim as the checker reads it: ISO dates, ASCII minus and hyphen."""
    return normalize_dates(text.replace("\u2212", "-").replace("\u2013", "-"))


# Unit conversions a calculation may use without citing them (besides powers of ten): days,
# hours, weeks and months; barrels per Sm3 and back; cubic feet per Sm3.
CONSTANTS = {7.0, 12.0, 24.0, 30.0, 30.44, 52.0, 60.0, 365.0, 365.25, 3600.0, 86400.0}
CONSTANTS |= {6.2898, 0.159, 0.158987, 35.3147}
# Causes are interpretations: the data shows timing and size, not why.
# "Water drive" and "gas drive" are recovery mechanisms, and "the fit results in E9" points
# at evidence: neither is a claim about why.
CAUSAL = re.compile(
    r"\b(because|caus(?:e|ed|es|ing)|"
    r"(?<!water )(?<!gas )(?<!aquifer )(?<!depletion )(?<!gravity )dr(?:ove|ives?|iven|iving)|"
    r"due to|led to|leads? to|result(?:s|ed)? (?:from|in)(?! E\d)|(?:as a )?result of|"
    r"responsible for|attribut(?:able|ed) to|triggered by|owing to|thanks to|"
    r"(?:a )?consequence of|contribut(?:ed|es|ing) to|stem(?:s|med)? from|induced by|"
    r"the reason|accounts? for|explain(?:s|ed by)?)\b",
    re.IGNORECASE,
)
# A quoted passage: opened after a space or "(", closed before a space or punctuation, so
# apostrophes inside words ("oil's") do not end it.
# Curly quotes and guillemets open after any non-letter ("—“..."), straight quotes only
# after a space, "(" or the start. A quote cannot run across another opening quote, so two
# scare-quoted words ('plateau' ... 'decline') are not read as one quotation.
QUOTED = re.compile(
    r"""(?:(?:(?<=[\s(])|^)['"]|(?<!\w)[‘“«])((?:(?!\s['"‘“«]).){15,}?)['"’”»]"""
    r"""(?=[\s.,;:)/\]?!—–-]|$)"""
)
# A quote with its translation after it, in brackets or parentheses: “Produksjonen
# avtar...” [Production declines due to...]. A translation found in the evidence (the
# fact's statement_en) reports the document's claim, so its causal words are not the
# finding's; one written freely is checked like the finding's own words.
QUOTED_CLAIM = re.compile(QUOTED.pattern + r"(\s*(?:\[[^\]]*\]|\([^)]*\)))?")
MIN_QUOTE_CHARS = 12  # letters and digits; shorter quotes are phrases, not citations
# Checked per sentence: a hedge in one sentence does not license a cause in another.
HEDGE = re.compile(
    r"\b(likely|probabl[ey]|possibly|perhaps|presumabl[ey]|plausibl[ey]|consistent with|"
    r"suggests?|seems?|appears? to|could|would|may|might|hypothes\w*|interpret\w*|"
    r"points? to)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"(?<=[.;!?])\s+")


class DeclaredNumber(BaseModel):
    text: str = Field(
        description="The number exactly as written in the text, e.g. '4 050', '72%', '0.4461', "
        "'1.13 million'."
    )
    value: float = Field(
        description="Its full value as the evidence has it: 4050 for '4 050' or '4.05k', "
        "1133053 for '1.13 million', 0.72 or 72 for '72%'."
    )
    evidence_id: str = Field(description="The evidence id (E1, E2, ...) that contains the value.")


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
    # None only for answers from before numbers were declared; the agent must declare them.
    numbers: list[DeclaredNumber] | None = Field(
        None,
        description="Every number written in the claim (not dates, years or well names), "
        "with its value and the cited evidence it comes from. Empty if there are none.",
    )


class Submission(BaseModel):
    summary: str = Field(description="Two or three sentences answering the question.")
    findings: list[Finding] = Field(min_length=1)
    summary_numbers: list[DeclaredNumber] | None = Field(
        None, description="Every number written in the summary, declared the same way."
    )


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
        mantissa, _, exponent = digits.lower().partition("e")
        # Decimals of the value as written: "1.200e+04" is good to tens, so -1. Clamped so
        # that "1e400" cannot overflow a tolerance.
        decimals = max(len(mantissa.partition(".")[2]) - int(exponent or 0), -30)
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


@lru_cache(maxsize=256)
def _echoes(sql: str) -> tuple[frozenset[float], frozenset[str]]:
    """Literals the model wrote into a SELECT list (SELECT 4321 AS n): the query only echoes
    them back. A literal in a filter (WHERE date = '2010-10-12') is not an echo: when the
    data returns that value, it is data. Unparseable SQL counts every literal as an echo."""
    tree = json.loads(duckdb.connect().execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    if tree.get("error"):
        strings = {s.replace("''", "'") for s in SQL_STRING.findall(sql)}
        numbers = {v for v, _, _ in _numbers_in_text(SQL_STRING.sub(" ", sql))}
        return frozenset(numbers), frozenset(strings)
    numbers, strings = set(), set()

    def walk(node: Any, projected: bool) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, projected)
        elif isinstance(node, dict):
            literal = node.get("literal") if node.get("class") == "CONSTANT" else None
            if projected and isinstance(literal, dict):
                if literal.get("kind") == "STRING":
                    strings.add(literal.get("text", ""))
                else:
                    with suppress(ValueError):
                        numbers.add(abs(float(literal.get("text", ""))))
            for key, child in node.items():
                walk(child, projected or key == "select_list")

    walk(tree, False)
    return frozenset(numbers), frozenset(strings)


def _typed_by_model(item: Evidence) -> tuple[frozenset[float], frozenset[str]]:
    """Numbers and strings the model wrote into a SQL query's SELECT list: echoes, not data."""
    if item.tool != "run_sql":
        return frozenset(), frozenset()
    return _echoes(str(item.arguments.get("sql", "")))


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
                    self.dates.update(DATE.findall(normalize_dates(leaf)))
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


def _is_constant(value: float) -> bool:
    """A unit constant: a listed conversion or a power of ten (/1000, /1e6, *100)."""
    if value in CONSTANTS:
        return True
    return value >= 10 and math.isclose(math.log10(value), round(math.log10(value)))


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
        years = _Pool(earlier).years
        for value, decimals, _ in _numbers_in_text(expression):
            if _is_constant(value) or (decimals <= 0 and value <= SMALL_INTEGER):
                continue
            if value.is_integer() and str(int(value)) in years:
                continue  # year arithmetic: "2010 - 2008"
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


def _free(value: float, decimals: int, suffix: str | None, pool: _Pool, free: set[float]) -> bool:
    """Whole numbers that need no evidence: integers up to 10, years (an evidence year, a
    year the data returns as a number, a report's year) and whole numbers the question
    itself names ("the next 12 months")."""
    if decimals != 0 or suffix or not value.is_integer():
        return False
    if value <= SMALL_INTEGER or value in free:
        return True
    return 1900 <= value <= 2100 and (
        str(int(value)) in pool.years or pool.has_number(value, 0, None)
    )


def _unsupported(
    text: str, pool: _Pool, free: set[float], free_dates: set[str]
) -> tuple[list[str], list[str]]:
    text = _normalize(text)
    numbers = []
    for value, decimals, suffix in _numbers_in_text(_mask(text)):
        if not _free(value, decimals, suffix, pool, free) and not pool.has_number(
            value, decimals, suffix
        ):
            numbers.append(_as_written(value, decimals, suffix))
    dates = [d for d in DATE.findall(text) if d not in free_dates and not pool.has_date(d)]
    return numbers, dates


def _quoted_in(quote: str, pool: _Pool) -> bool:
    """A quote is in the evidence word for word; one elided with "…" or "[...]" must have
    each part there, in order."""
    parts = [squash(p) for p in re.split(r"\[?(?:…|\.\.\.)\]?", quote)]
    at = 0
    for part in (p for p in parts if p):
        at = pool.text.find(part, at)
        if at < 0:
            return False
        at += len(part)
    return True


def _missing_quotes(text: str, pool: _Pool) -> list[str]:
    quotes = [q for q in QUOTED.findall(text) if len(squash(q)) >= MIN_QUOTE_CHARS]
    return [q for q in quotes if not _quoted_in(q, pool)]


def _own_words(text: str, pool: _Pool) -> str:
    """The text without quotes, and without translations of quotes found in the evidence:
    what the finding itself claims."""

    def strip(match: re.Match) -> str:
        translation = match.group(2)
        if translation and squash(translation.strip()[1:-1]) not in pool.text:
            return " " + translation
        return " "

    return QUOTED_CLAIM.sub(strip, text)


def _precision(value: float) -> float:
    """Half a unit in the last significant digit of value as written: 4050 -> 5, 0.72 ->
    0.005, 1146 -> 0.5."""
    exponent = Decimal(repr(float(value))).normalize().as_tuple().exponent
    return 0.5 * 10.0 ** int(exponent) + 1e-9


Reading = tuple[float, float, tuple[float, ...]]  # number, half a unit, allowed scales


def _readings(text: str) -> list[Reading]:
    """What a declared number's text can show: the number, half a unit in its last digit, and
    the scales its suffix allows ("72%" is 72 or 0.72, "1.13 million" is 1.13 or 1 130 000).

    Both conventions are tried, since answers quote Norwegian reports: "9.000" is 9 or 9000,
    "0,80" is 0.8. Trailing zeros of a whole number count as rounding: "4 050" is good to 5,
    "1150" to 5. Empty unless the text holds exactly one number.
    """
    readings: list[Reading] = []
    for european in (False, True):
        found = _numbers_in_text(text, european)
        if len(found) != 1:
            continue
        shown, decimals, suffix = found[0]
        if not math.isfinite(shown):
            continue
        unit = 0.5 * 10.0**-decimals if decimals else _precision(shown)
        if not decimals and shown >= 10:
            # Trailing zeros are rounding, but to two significant digits at most: "4000" is
            # good to 50, so 3593 does not verify it (half a leading digit would be 500).
            unit = min(unit, 0.5 * 10.0 ** (math.floor(math.log10(shown)) - 1))
        reading = (shown, unit, tuple(dict.fromkeys((1.0, *SCALES.get(suffix or "", ())))))
        if reading not in readings:
            readings.append(reading)
    return readings


def _reads_as(reading: Reading, candidates: Iterable[float]) -> bool:
    """Does any candidate round to the shown number, at any scale its suffix allows?

    A tie rounds either way: 253.85 may be written 253.8 or 253.9. The slack absorbs float
    error there (253.9 - 253.85 is 0.05000000000001137).
    """
    number, unit, scales = reading
    pool = [abs(c) for c in candidates]
    return any(
        abs(c - number * s) <= unit * s + 1e-12 * max(number * s, 1.0) for s in scales for c in pool
    )


# A standalone group of 1-3 digits and a space just before: "1 " in "1 146". Not "P50 " or
# "r2 ", where the digits belong to a label.
GROUP_BEFORE = re.compile(r"(?:^|[^\w.,])\d{1,3}[ \u00a0\u202f]$")
# Window lengths in calendar units ("30-day median", "12 month forecast", "the next 12
# months", "365 days after the cutoff") describe a method, not the data, so they need no
# declaring. Any other duration ("45-day shut-in", "30 days of shut-in") does.
CALENDAR_WINDOW = re.compile(
    r"\b(?:12|24|30|365)[-\s](?:month|hour|day)\b|\b(?:12\s+months|24\s+hours|365\s+days)\b"
)


def _pinned(reading: Reading, value: float) -> Reading | None:
    """The reading at the scales its declared value takes, or None if none fits.

    A full value fixes the scale: "72%" declared 0.72 must be 0.72 in the evidence, not a
    72.3 choke; "1146k" declared 1146000 is not 1145.72. A short value ("4.58" written
    "4.58 MSm3", "72" for "72%") leaves the scale open, as before.
    """
    number, unit, scales = reading
    fitting = tuple(s for s in scales if _reads_as((number, unit, (s,)), [value]))
    if not fitting:
        return None
    return reading if fitting == (1.0,) else (number, unit, fitting)


def _standalone(text: str, written: str) -> list[tuple[int, int]]:
    """Spans where a declared text stands as a whole number: "2" in "2 years", not the 2 in
    "2 100", "12", "0.2" or "Sm3"; "276,078" in "P50 276,078"."""
    pattern = re.compile(rf"(?<![\w.,]){re.escape(written)}(?!\d|[.,]\d|[ \u00a0\u202f]\d{{3}}\b)")
    return [m.span() for m in pattern.finditer(text) if not GROUP_BEFORE.search(text, 0, m.start())]


def _mask(text: str) -> str:
    """Dates, well names and calendar windows blanked out: their digits are not numbers to
    declare."""
    return CALENDAR_WINDOW.sub(" ", DATE.sub(" ", WELL.sub(" ", text)))


def _as_written(value: float, decimals: int, suffix: str | None) -> str:
    return f"{value:.{max(decimals, 0)}f}{suffix or ''}"


def _declared(
    label: str,
    text: str,
    declared: list[DeclaredNumber],
    log: EvidenceLog,
    calcs: dict[str, _Calc],
    citable: set[str] | None,
    pool: _Pool,
    free: set[float],
) -> list[str]:
    """Check declared numbers, then that the text holds no number left undeclared.

    No number is parsed out of prose: the model says what each written number is worth and
    where it comes from; the only parsing left is spotting digits nobody declared.
    """
    problems = []
    text = _normalize(text)
    declared = [d.model_copy(update={"text": _normalize(d.text)}) for d in declared]
    masked = _mask(text)
    # A declaration with no number in the text hides nothing (every written number must be
    # declared anyway), so it costs no correction round; it is only named as a hint below.
    unused = [d for d in declared if not _standalone(masked, d.text)]
    for number in declared:
        if number in unused:
            continue
        where = number.evidence_id
        item = log.get(where)
        # A unit the text writes right after the declared number scales it too: "4.58"
        # declared, "4.58 MSm3" written.
        unit = SUFFIX_AFTER.match(masked, _standalone(masked, number.text)[0][1])
        # (+2 keeps the "Sm" that "M" and "k" need after them)
        readings = _readings(number.text + (masked[unit.start() : unit.end() + 2] if unit else ""))
        fits = [f for r in readings if (f := _pinned(r, number.value))]
        exempt = [n for n in _numbers_in_text(number.text) if _free(*n, pool, free)]
        if fits and exempt:
            continue  # "3 producers", "2010": declared though it needs no evidence
        if not readings:
            problems.append(
                f"{label}: '{number.text}' must be one number exactly as written, e.g. "
                "'4 050' or '72%'; declare each number separately."
            )
        elif not fits:
            problems.append(
                f"{label}: '{number.text}' does not show its declared value "
                f"{number.value:g}; write the number the evidence has."
            )
        elif citable is not None and where not in citable:
            problems.append(f"{label}: '{number.text}' comes from {where}, which it does not cite.")
        elif item is None:
            problems.append(f"{label}: '{number.text}' comes from unknown evidence {where}.")
        elif where in calcs and calcs[where].unsupported:
            problems.append(
                f"{label}: '{number.text}' comes from {where}, which calculates from "
                f"{', '.join(calcs[where].unsupported)}; no evidence contains those."
            )
        elif not any(_reads_as(r, (numbers := _Pool([item]).numbers)) for r in fits):
            nearest = sorted(set(numbers), key=lambda n: abs(n - abs(number.value)))[:3]
            hint = f" Closest there: {', '.join(f'{n:g}' for n in nearest)}." if nearest else ""
            problems.append(f"{label}: '{number.text}' = {number.value:g} is not in {where}.{hint}")

    rest = _mask(QUOTED.sub(" ", text))  # numbers inside a quote are checked with the quote
    for number in sorted(declared, key=lambda d: -len(d.text)):
        for start, end in reversed(_standalone(rest, number.text)):
            rest = rest[:start] + " " + rest[end:]
    leftover: list[str] = []
    # Either convention: an undeclared "1.234.567" is a number too.
    for european in (False, True):
        for value, decimals, suffix in _numbers_in_text(rest, european):
            if _free(value, decimals, suffix, pool, free):
                continue
            written = _as_written(value, decimals, suffix)
            if written not in leftover:
                leftover.append(written)
    if leftover:
        hint = (
            f" Declared but not written as such: {', '.join(repr(d.text) for d in unused)}."
            if unused
            else ""
        )
        problems.append(
            f"{label}: undeclared number(s) {', '.join(leftover)}. Declare every number you "
            f"write, with its value and evidence id, or remove it.{hint}"
        )
    return problems


def _question_numbers(question: str) -> set[float]:
    """Whole numbers the question names, plus the reports' years: the model may repeat them."""
    named = {
        v for v, decimals, suffix in _numbers_in_text(question) if decimals == 0 and not suffix
    }
    return named | REPORT_YEARS


def _blank(text: str, *patterns: re.Pattern) -> str:
    """The text with each match turned into spaces of the same length: positions kept."""
    for pattern in patterns:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def display_spans(text: str | None, numbers: list[DeclaredNumber] | None) -> list[dict]:
    """Where the answer panel marks the text, by the checker's own rules: each quoted
    passage, and each declared number where it stands as a whole number (not inside a
    date, a well name or a window), with the evidence it came from. Positions index the
    text as written, so the UI only slices; it re-implements no rule."""
    if not text:
        return []
    spans: list[dict] = [
        {"kind": "quote", "start": m.start(), "end": m.end()} for m in QUOTED.finditer(text)
    ]
    plain = text.replace("\u2212", "-").replace("\u2013", "-")  # same length
    masked = _blank(plain, DATE, WELL, CALENDAR_WINDOW, DAY_MONTH_YEAR, MONTH_DAY_YEAR, MONTH_YEAR)
    declared: dict[str, list[DeclaredNumber]] = {}
    for number in numbers or []:
        written = number.text.replace("\u2212", "-").replace("\u2013", "-")
        declared.setdefault(written, []).append(number)
    taken: list[tuple[int, int]] = []
    for written in sorted(declared, key=len, reverse=True):  # "2 100" before "2"
        # The same text declared twice: its occurrences take the declarations in order.
        options = declared[written]
        for i, (start, end) in enumerate(_standalone(masked, written)):
            if any(start < e and end > s for s, e in taken):
                continue
            taken.append((start, end))
            number = options[min(i, len(options) - 1)]
            spans.append(
                {
                    "kind": "number",
                    "start": start,
                    "end": end,
                    "value": number.value,
                    "evidence_id": number.evidence_id,
                }
            )
    return sorted(spans, key=lambda s: (s["start"], s["kind"] != "quote"))


def check_submission(submission: Submission, log: EvidenceLog, question: str = "") -> list[str]:
    """Return problems as sentences for the model; an empty list means accepted."""
    free = _question_numbers(question)
    free_dates = REPORT_DATES | set(DATE.findall(_normalize(question)))
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
        numbers, dates = _unsupported(finding.claim, pool, free, free_dates)
        if finding.numbers is not None:
            problems += _declared(
                f"Finding {n}",
                finding.claim,
                finding.numbers,
                log,
                calcs,
                set(finding.evidence_ids),
                pool,
                free,
            )
        elif numbers:  # an answer from before declared numbers: read them from the text
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
        cause = CAUSAL.search(_own_words(finding.claim, pool))
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
    numbers, dates = _unsupported(submission.summary, everything, free, free_dates)
    if submission.summary_numbers is not None:
        problems += _declared(
            "Summary",
            submission.summary,
            submission.summary_numbers,
            log,
            calcs,
            None,
            everything,
            free,
        )
        numbers = []
    missing = (
        numbers + dates + [f"'{q[:80]}'" for q in _missing_quotes(submission.summary, everything)]
    )
    if missing:
        problems.append(f"Summary: {', '.join(missing)} not found in any evidence.")
    own = _normalize(_own_words(submission.summary, everything))  # "May 2010" is a date
    for sentence in SENTENCE.split(own):
        cause = CAUSAL.search(sentence)
        if cause and not HEDGE.search(sentence):
            problems.append(
                f"Summary states a cause as certain ('{cause.group(0)}'). The data shows "
                "timing and size, not causes: hedge it in the same sentence, e.g. 'most "
                "likely because' or 'consistent with'."
            )
            break
    return problems
