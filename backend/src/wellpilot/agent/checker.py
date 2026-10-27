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

A number found in its evidence must also be the one the text says it is, as far as the
evidence states outright: not only another well's (its well argument, a record's "well", a
table's well column), not only another quantity (uptime written as water cut), not a fall
written as a rise, not another event's date, not a before and after swapped. Dates a
finding gives a well must be that well's too.

Known limit: meaning is not checked. "Pressure collapsed to 0" passes when the evidence
says the gauge reads 0; whether the reasoning holds is for the reader, who can open every
source.
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

# A well name: Volve's "F-12", "15/9-F-11 B", "F-1 C", or another field's "P-1", "INJ-2".
WELL = re.compile(r"\b(?:15/9-)?[A-Z]{1,3}-\d+(?:\s?[A-Z]\b)?")
DATE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?)\b")
# English month names, capitalised ("May 2010" is a month, "may" is not), with a year.
_MONTH = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
# Only years of this or the last century: "Jan 1363" is a month's value, not January 1363.
_YEAR = r"((?:19|20)\d{2})"
DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}\s+{_YEAR}\b")
MONTH_DAY_YEAR = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR}\b")
MONTH_YEAR = re.compile(rf"\b{_MONTH}\s+{_YEAR}\b")
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
# A number as the text writes it, so a rejection can name it that way: "0,80", "4 050",
# "1,133,053", "2.5x". (Parsing it into values may split or rewrite it.)
WRITTEN_NUMBER = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)*(?:[eE][-+]?\d+)?"
    r"(?:\s*(?:%|percent\b|pct\b|x\b|×|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm)|k\b))?"
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
    r"(?<!water )(?<!gas )(?<!aquifer )(?<!depletion )(?<!gravity )"
    r"dr(?:ove|ives?|iven|iving|ivers?)|"
    r"due to|led to|leads? to|result(?:s|ed)? (?:from|in)(?! E\d)|(?:as a )?result of|"
    r"responsible for|attribut(?:able|ed) to|triggered by|owing to|thanks to|"
    r"(?:a )?consequence of|contribut(?:ed|es|ing) to|stem(?:s|med)? from|induced by|"
    r"the reason|accounts? for|explain(?:s|ed by)?|reflect(?:s|ed|ing)?|(?:direct )?effect of|"
    r"(?:a |the )?direct effect)\b",
    re.IGNORECASE,
)
# A quoted passage: opened after a space or "(", closed before a space or punctuation, so
# apostrophes inside words ("oil's") do not end it.
# Curly quotes and guillemets open after any non-letter ("—“..."), straight quotes only
# after a space, "(" or the start. A quote cannot run across another opening quote, so two
# scare-quoted words ('plateau' ... 'decline') are not read as one quotation.
QUOTED = re.compile(
    r"""(?:(?:(?<=[\s(\[])|^)['"]|(?<!\w)[‘“«])((?:(?!\s['"‘“«]).){15,}?)['"’”»]"""
    r"""(?=[\s.,;:)/\]?!—–-]|$)"""
)
# A quote with its translation after it, in brackets or parentheses: “Produksjonen
# avtar...” [Production declines due to...]. A translation found in the evidence (the
# fact's statement_en) reports the document's claim, so its causal words are not the
# finding's; one written freely is checked like the finding's own words.
QUOTED_CLAIM = re.compile(QUOTED.pattern + r"(\s*(?:\[[^\]]*\]|\([^)]*\)))?")
MIN_QUOTE_CHARS = 12  # letters and digits; shorter quotes are phrases, not citations
# A translation written inside the quote marks: 'På det tidspunkt er ... omlag 6 MSm3 [At
# the time of water breakthrough ...]'. Moved after the closing mark, where translations are
# read; the text keeps its length, so positions still index it as written. "[...]" is an
# elision, not a translation.
INNER_TRANSLATION = re.compile(r"""(\s*)(\[(?!\s*(?:…|\.\.\.)\s*\])[^\[\]]{4,}\])(['"’”»])""")


def _translations_outside(text: str) -> str:
    return INNER_TRANSLATION.sub(lambda m: m[3] + m[1] + m[2], text)


# Checked per sentence: a hedge in one sentence does not license a cause in another.
HEDGE = re.compile(
    r"\b(likely|probabl[ey]|possibly|perhaps|presumabl[ey]|plausibl[ey]|consistent with|"
    r"suggests?|seems?|appears? to|could|would|may|might|hypothes\w*|interpret\w*|"
    r"points? to)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"(?<=[.;!?])\s+")
# A warned number's hedge may come after a semicolon ("...335 thousand; the fit suggests
# ..."), so that rule reads whole sentences; a cause must be hedged in its own clause.
FULL_SENTENCE = re.compile(r"(?<=[.!?])\s+")


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
    claim: str = Field(
        description="One finding in one or two plain sentences a manager can follow: at most "
        "three numbers, rounded, with units; no fit statistics."
    )
    detail: str | None = Field(
        None,
        description="Optional support for an engineer: the fit behind the claim (model, "
        "window, r2_log, di, b) and exact values, copied as the tools give them. Its numbers "
        "need no declaring, but each must be in the evidence the finding cites.",
    )
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


def _numbers_in_text(
    text: str, european: bool = False, signed: bool = False
) -> list[tuple[float, int, str | None]]:
    """(value, decimals, suffix) for each number, ignoring well names and ISO dates.

    european=True reads Norwegian report text: "9.000" is nine thousand, "0,80" is 0.8.
    signed=True keeps a minus written right before a number ("(-55%)"); one after a digit
    is a range ("4 000-5 000"), not a sign.
    """
    text = DATE.sub(" ", WELL.sub(" ", text)).replace("−", "-")
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
        value = float(digits)
        before = text[: match.start()]
        if signed and re.search(r"(?:^|[^\w.,])-$", before):
            value = -value
        found.append((value, decimals, suffix))
    return found


def _matches(candidates: Iterable[float], value: float, decimals: int, suffix: str | None) -> bool:
    """Does any candidate round to the claimed value at the precision it was written?"""
    pool = list(candidates)
    # A unit may be written in the evidence too ("6 MSm3" in the plan): the bare number
    # counts as well as the scaled one.
    for scale in dict.fromkeys((*SCALES.get(suffix or "", ()), 1.0)):
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
    inputs: int = 0  # inputs taken from evidence (not unit constants): 2+ is a comparison


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
        missing, warnings, inputs = [], [], 0
        years = _Pool(earlier).years
        # One expression, or several in one call: inputs counts the most any one uses.
        expressions = item.arguments.get("expressions") or [item.arguments.get("expression", "")]
        found = []
        for expression in expressions:
            used = 0
            for value, decimals, _ in _numbers_in_text(str(expression)):
                if _is_constant(value) or (decimals <= 0 and value <= SMALL_INTEGER):
                    continue
                if value.is_integer() and str(int(value)) in years:
                    continue  # year arithmetic: "2010 - 2008"
                used += 1
                found.append((value, decimals))
            inputs = max(inputs, used)
        for value, decimals in found:
            holders = [e for e in earlier if _Pool([e]).has_number(value, decimals, None)]
            if not holders:
                missing.append(f"{value:g}")
            elif all(e.warnings or calcs.get(e.id, _Calc((), ())).warnings for e in holders):
                for e in holders:
                    warnings.extend(e.warnings or calcs[e.id].warnings)
        calcs[item.id] = _Calc(tuple(missing), tuple(dict.fromkeys(warnings)), inputs)
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


def _looked_up(text: str, pool: _Pool, free: set[float]) -> list[str]:
    """Numbers a text writes (outside quotes) that its evidence does not hold, read as a
    declared number would be: either convention, trailing zeros as rounding ("1500" is good
    to 50), a unit's scale or the unit written in the evidence too, and cut or rounded at the
    last digit. For a finding's detail, whose numbers are looked up rather than declared."""
    missing = []
    quotes_out = QUOTED.sub(" ", _translations_outside(_normalize(text)))
    for match in WRITTEN_NUMBER.finditer(_mask(quotes_out)):
        written = match.group(0).strip()
        values = [n for european in (False, True) for n in _numbers_in_text(written, european)]
        if not values or all(_free(*n, pool, free) for n in values):
            continue
        # A copied value may be cut rather than rounded (0.5536 as "0.553"): a whole unit of
        # the last digit either way.
        if not any(
            _reads_as((n, 2 * unit, sc), pool.numbers) for n, unit, sc in _readings(written)
        ):
            missing.append(written)
    return list(dict.fromkeys(missing))


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
    quotes = [
        q for q in QUOTED.findall(_translations_outside(text)) if len(squash(q)) >= MIN_QUOTE_CHARS
    ]
    return [q for q in quotes if not _quoted_in(q, pool)]


def _own_words(text: str, pool: _Pool) -> str:
    """The text without quotes, and without translations of quotes found in the evidence:
    what the finding itself claims."""

    def strip(match: re.Match) -> str:
        translation = match.group(2)
        if translation and squash(translation.strip()[1:-1]) not in pool.text:
            return " " + translation
        return " "

    return QUOTED_CLAIM.sub(strip, _translations_outside(text))


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
    r"\b(?:12|24|30|90|180|365)[-\s](?:month|hour|day)\b"
    r"|\b(?:12\s+months|24\s+hours|(?:90|180|365)\s+days)\b"
)
DECIMAL_COMMA = re.compile(r"\d,\d{1,2}(?!\d)")
DOTTED_THOUSANDS = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+")
# A reference to a part of a document names it; its digits are no value to declare:
# "Figure 4.2", "Tabell 4.5", "section 4.5.1", "§ 4.5.1", "page 23", "p. 23", "pages 23-25".
REFERENCE = re.compile(
    r"\b(?:fig(?:ure|ur)?|tab(?:le|ell)?|sections?|chapters?|kap(?:ittel|itel)?|appendix|"
    r"vedlegg|pages?|side|p)\.?\s*\d+(?:\.\d+)*(?:\s*[-–]\s*\d+)?\b|§\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
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
    return REFERENCE.sub(" ", CALENDAR_WINDOW.sub(" ", DATE.sub(" ", WELL.sub(" ", text))))


# A number the text ties to a month or day: "0.85 in 2009-06", "1 146 Sm3/d on 2010-07-18".
TIED = (
    r"(?:\s*(?:[^\W\d]+/?[^\W\d]*|%)){0,3}?"  # a unit or two ("Sm3/d", "%"); no comma
    r"\s+(?:in|on|at|during|by|for)\s+(\d{4}-\d{2}(?:-\d{2})?)\b"
    r"(?!\s*(?:to|until|through|and|-|–)\s*\d{4})"  # a range of dates is no one row
)
DATE_COLUMNS = {"month", "date", "day", "week", "period"}


def _mask_keep_dates(text: str) -> str:
    """Well names, windows and references blanked, dates kept: the text for tying numbers
    to the dates next to them."""
    return REFERENCE.sub(" ", CALENDAR_WINDOW.sub(" ", WELL.sub(" ", text)))


def _tables(value: Any) -> Iterable[dict]:
    """Column-wise tables anywhere in a tool result."""
    if isinstance(value, dict):
        if isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list):
            yield value
        for v in value.values():
            yield from _tables(v)
    elif isinstance(value, list):
        for v in value:
            yield from _tables(v)


def _row_mismatch(
    text: str, written: str, readings: list[Reading], item: Evidence
) -> tuple[str, str] | None:
    """(date, what that row holds) when the text ties a number to a date whose table row in
    the evidence does not hold it. The number is in the evidence (checked before); this
    catches it read off the wrong row. A date with no row in the table is not judged."""
    for start, end in _standalone(text, written):
        tied = re.match(TIED, text[end:])
        if not tied:
            continue
        when = tied.group(1)
        # "Cut to 40% in 2018-11": a level reached by a change during the month, whose row
        # averages the days before and after it; the next row holds it too.
        reached = bool(re.search(r"\bto\s*$", text[:start]))
        for table in _tables(item.result):
            columns = [str(c).lower() for c in table["columns"]]
            key = next((i for i, c in enumerate(columns) if c in DATE_COLUMNS), None)
            if key is None:
                continue
            # The same grain only: a day against a daily row, a month against a monthly one.
            at = [i for i, r in enumerate(table["rows"]) if str(r[key])[:10] == when]
            if len(at) != 1:
                continue  # no row, or several (a month tied in a daily table): not judged
            rows = table["rows"][at[0] : at[0] + (2 if reached else 1)]
            values = [
                float(v)
                for row in rows
                for v in row
                if isinstance(v, int | float) and not isinstance(v, bool)
            ]
            if not any(_reads_as(r, values) for r in readings):
                held = ", ".join(
                    f"{c} {v}"
                    for c, v in zip(table["columns"], rows[0], strict=False)
                    if v is not None
                )
                return when, held[:200]
    return None


@dataclass(frozen=True)
class _Leaf:
    """One value a tool result holds, with where it sits in the result."""

    value: float | None  # a number, with its sign
    day: str | None  # or an ISO date (a day or a month)
    labels: str  # its key (or column) and its parent's, and its records' signal or kind
    well: str | None  # the well that part of the result is about; None: field-wide or unknown
    record: int | None  # the record it belongs to: an event, a table row, an item of a list
    record_days: frozenset[str]  # the dates that name its record: start, end, since, ...


# String fields that say what a record measures ("signal": "water_cut"): they label its values.
RECORD_LABEL_KEYS = {"signal", "kind", "metric", "column", "field"}
# Fields that date a record: an event's start and end, an issue's since, a row's date.
RECORD_DATE_KEYS = {"start", "end", "since", "date", "day", "month", "week"}


def _leaves(item: Evidence) -> list[_Leaf]:
    """Every number and date in a tool result, with its labels, well and record.

    The well comes from the call's well argument, from a record's "well" key, or from a
    table's "well" column; a subtree whose key starts with "field" (field_voidage) is
    field-wide. Model-typed SQL literals are left out, as in the pool.
    """
    typed_numbers, typed_strings = _typed_by_model(item)
    found: list[_Leaf] = []
    records = iter(range(1_000_000))

    def leaf(value: Any, labels: tuple[str, ...], well, record, days) -> None:
        label = "|".join(labels)
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            if abs(float(value)) not in typed_numbers:
                found.append(_Leaf(float(value), None, label, well, record, days))
        elif isinstance(value, str) and value not in typed_strings:
            for european in (False, True):
                for number, _, _ in _numbers_in_text(value, european, signed=True):
                    found.append(_Leaf(number, None, label, well, record, days))
            for day in DATE.findall(normalize_dates(value)):
                found.append(_Leaf(None, day, label, well, record, days))

    def walk(node: Any, keys: tuple[str, ...], kinds: tuple[str, ...], well, record, days):
        # A value's labels: its own key (or column) and its parent's, and what the records
        # around it measure. Container names further up ("wor_fit") say nothing about it.
        if isinstance(node, dict) and isinstance(node.get("columns"), list):
            columns = [str(c).lower() for c in node["columns"]]
            at_well = columns.index("well") if "well" in columns else None
            for row in node.get("rows") or []:
                row_well = str(row[at_well]) if at_well is not None and row[at_well] else well
                rid = next(records)
                # A table row is no event: its date tie is _row_mismatch's, so no dates here.
                for column, value in zip(columns, row, strict=False):
                    leaf(value, (*keys[-1:], column, *kinds), row_well, rid, frozenset())
        elif isinstance(node, dict):
            here = node["well"] if isinstance(node.get("well"), str) else well
            kinds = kinds + tuple(
                str(v).lower()
                for k, v in node.items()
                if k in RECORD_LABEL_KEYS and isinstance(v, str)
            )
            for key, child in node.items():
                key = str(key).lower()
                field_wide = key.startswith("field")
                walk(child, (*keys[-1:], key), kinds, None if field_wide else here, record, days)
        elif isinstance(node, list):
            for child in node:
                if isinstance(child, dict) and not isinstance(child.get("columns"), list):
                    child_days = frozenset(
                        str(child[k])[:10]
                        for k in RECORD_DATE_KEYS
                        if isinstance(child.get(k), str) and DATE.match(str(child[k]))
                    )
                    walk(child, keys, kinds, well, next(records), child_days)
                else:
                    walk(child, keys, kinds, well, record, days)
        else:
            leaf(node, (*keys[-2:], *kinds), well, record, days)

    well = item.arguments.get("well")
    walk(item.result, (), (), str(well) if well else None, None, frozenset())
    return found


# Quantities a number may be named as, and how a tool result labels each one. A number the
# text names as one quantity must not be, in its evidence, only another.
QUANTITIES = {
    "water cut": (r"water[\s-]?cut", r"water_?cut"),
    "uptime": (r"uptime|on[\s-]stream", r"uptime|on_stream"),
    "choke": (r"choke", r"choke"),
    "GOR": (r"\bGOR\b|gas[\s/-]oil ratio", r"(?:^|[|_])gor(?:[|_]|$)"),
    "downhole pressure": (
        r"\bBHP\b|downhole pressure|bottom[\s-]?hole pressure",
        r"(?:^|[|_])bhp(?:[|_]|$)|downhole",
    ),
    "wellhead pressure": (r"\bWHP\b|wellhead pressure", r"(?:^|[|_])whp(?:[|_]|$)"),
    "WOR": (r"\bWOR\b|water[\s/-]oil ratio", r"(?:^|[|_])wor(?:[|_]|$)"),
}
_NAMES = re.compile("|".join(f"(?:{t})" for t, _ in QUANTITIES.values()), re.IGNORECASE)
RISE_BEFORE = re.compile(
    r"\b(?:rose|risen|rising|increased|increasing|grew|grown|jumped|climbed|gained)"
    r"(?:\s+by)?(?:\s+(?:about|around|some|nearly|almost))?\s*[~≈]?\s*$",
    re.IGNORECASE,
)
RISE_AFTER = re.compile(r"^\s*(?:%|percent)?\s*(?:increase|rise|jump|gain)\b", re.IGNORECASE)
# A number tied to the date of one record (an event, an issue): "55% on 2010-05-02",
# "81 days since 2010-10-12". As TIED, with "around" and "since".
RECORD_TIED = TIED.replace(
    r"(?:in|on|at|during|by|for)", r"(?:in|on|at|during|by|for|around|since)"
)
CLAUSE = re.compile(r"[.;!?](?=\s)|\n")
# What may stand between a quantity's name and its number for the number to be that
# quantity: "water cut on <date> was 0.98", "uptime of 0.92". Any other word ("to",
# "limit", "producing") ends the name's reach.
LINK = re.compile(
    r"(?:'s)?(?:\s+|\s*[:=~≈]\s*|\b(?:of|at|is|was|were|on|in|by|reached|reaching|hit|"
    r"stood|near|about|around|approximately|roughly|averaged|averaging|first|then|now|"
    r"exceeded|above|below|median|mean|daily|monthly|weekly)\b)*",
    re.IGNORECASE,
)


def _named(text: str, start: int) -> set[str]:
    """The wells a number at `start` belongs to, by the nearest that names any:
    - its part of the sentence (up to the next ", "), with the wells listed before it in the
      sentence ("F-12, F-14 and F-11 had 0.9, 0.8 and 0.7");
    - its sentence before it;
    - the text before its sentence (P-1, named in the first sentence, is still the subject
      of "water cut stayed near 0.90, and replacement fell after injector I-2 stopped");
    - anywhere in the text."""

    def wells(part: str) -> set[str]:
        return {_well_key(m.group(0)) for m in WELL.finditer(part)}

    bounds = [0, *(m.end() for m in CLAUSE.finditer(text)), len(text)]
    begin = max(b for b in bounds if b <= start)
    end = min((b for b in bounds if b > start), default=len(text))
    part_end = text.find(", ", start, end)
    part_end = end if part_end < 0 else part_end
    part_begin = text.rfind(", ", begin, start)
    part_begin = begin if part_begin < 0 else part_begin
    if wells(text[part_begin:part_end]):
        return wells(text[begin:part_end])
    return wells(text[begin:start]) or wells(text[:begin]) or wells(text)


def _quantity(blanked: str, start: int, end: int) -> str | None:
    """The quantity the text names a number as: "water cut 0.98", "water cut on <date> was
    0.98", "0.69 water cut", "53% choke"; None when it names none next to it."""
    before = blanked[max(0, start - 60) : start]
    last = None
    for m in _NAMES.finditer(before):
        last = m
    if last and LINK.fullmatch(before[last.end() :]):
        name = last.group(0)
    else:
        after = re.match(rf"\s*(?:%|percent)?\s*({_NAMES.pattern})", blanked[end:], re.IGNORECASE)
        if not after:
            return None
        name = after.group(1)
    return next(q for q, (text, _) in QUANTITIES.items() if re.fullmatch(text, name, re.I))


def _attribution(
    text: str,
    blanked: str,
    written: str,
    fits: list[Reading],
    item: Evidence,
    leaves: list[_Leaf],
) -> str | None:
    """Why the number, though in its evidence, is not the one the text says, or None.

    Each check judges only what the evidence states outright: a value found anywhere
    without a well, label or sign that rules it out passes.
    - well: in the evidence only for other wells than its clause names (F-12's 1 551 from
      a field-wide screen written as F-14's);
    - quantity: named as one quantity but held only as others (uptime 0.98 as water cut);
    - sign: written as a rise but held only as a fall ("rose 55%" against -55%).
    """
    held = [
        lf for lf in leaves if lf.value is not None and any(_reads_as(r, [lf.value]) for r in fits)
    ]
    if not held:
        return None
    for start, end in _standalone(blanked, written):
        named = _named(text, start)
        wells = {lf.well for lf in held}
        if named and None not in wells and not {_well_key(w) for w in wells} & named:
            owners = ", ".join(sorted(wells))
            return f"is in {item.id} only for {owners}, not for the well its sentence names"
        quantity = _quantity(blanked, start, end)
        if quantity:
            pattern = QUANTITIES[quantity][1]
            other = [
                q
                for q, (_, label) in QUANTITIES.items()
                if q != quantity and all(re.search(label, lf.labels) for lf in held)
            ]
            if not any(re.search(pattern, lf.labels) for lf in held) and other:
                return f"is named {quantity}, but in {item.id} it is the {other[0]}"
        rise = (
            RISE_BEFORE.search(blanked[max(0, start - 40) : start])
            or RISE_AFTER.match(blanked[end:])
            or blanked[max(0, start - 1) : start] == "+"
        )
        if rise and all(lf.value < 0 for lf in held):
            return f"is written as a rise, but {item.id} has it as a fall ({held[0].value:g})"
    return None


def _record_mismatch(
    text: str, written: str, fits: list[Reading], leaves: list[_Leaf]
) -> tuple[str, str] | None:
    """(date, what that record holds) when the text ties a number to a date that names a
    record of its evidence (an event, an issue) and no record of that date holds it: "55%
    on 2010-06-04" when 55% is the 2010-05-02 step and 2010-06-04 a water-cut crossing. A
    date that names no record, or another grain, is not judged."""
    for _, end in _standalone(text, written):
        tied = re.match(RECORD_TIED, text[end:])
        if not tied:
            continue
        when = tied.group(1)
        dated = {lf.record for lf in leaves if lf.record is not None and when in lf.record_days}
        if not dated:
            continue
        values = [lf.value for lf in leaves if lf.record in dated and lf.value is not None]
        if not any(_reads_as(r, values) for r in fits):
            holders = [
                lf
                for lf in leaves
                if lf.value is not None and any(_reads_as(r, [lf.value]) for r in fits)
            ]
            dates = sorted({d for lf in holders for d in lf.record_days})
            return when, f"in its evidence it belongs to {', '.join(dates) or 'no dated record'}"
    return None


def _date_attribution(
    text: str, cited: list[Evidence], leaves_of, free_dates: set[str]
) -> list[str]:
    """Dates the text gives a well that its evidence has only for other wells: "F-14's
    gauge has read 0 since 2010-10-12" when 2010-10-12 is F-12's gauge failure. free_dates
    are dates no well owns: the reports', the question's, and the data's first and last day
    ("produced to 2016-12-01" is the field's end, though only F-4's rows reach it)."""
    wrong = []
    for match in DATE.finditer(text):
        day = match.group(1)
        if day in free_dates:
            continue
        named = _named(text, match.start())
        held = [
            lf
            for item in cited
            for lf in leaves_of(item)
            if lf.day is not None and (lf.day == day or (len(day) == 7 and lf.day.startswith(day)))
        ]
        wells = {lf.well for lf in held}
        if named and held and None not in wells and not {_well_key(w) for w in wells} & named:
            wrong.append(f"{day} (evidence has it only for {', '.join(sorted(wells))})")
    return list(dict.fromkeys(wrong))


FROM_TO = re.compile(r"\s*(?:[^\W\d]+(?:/[^\W\d]+)?\s*)?(?:to|->|→)\s*")


def _reversed(label, blanked, declared, unused, log, leaves_of) -> list[str]:
    """ "from X to Y" where the evidence has a change the other way: a record whose before
    value is Y and after value is X ("from 1 888 to 4 207" for a step from 4 207 to 1 888)."""
    spans = sorted(
        (start, end, id(d), d)
        for d in declared
        if d not in unused
        for start, end in _standalone(blanked, d.text)[:1]
    )
    problems = []
    for (a_start, a_end, _, a), (b_start, _, _, b) in zip(spans, spans[1:], strict=False):
        if a.evidence_id != b.evidence_id or not re.search(r"\bfrom\s*$", blanked[:a_start]):
            continue
        if (
            not FROM_TO.fullmatch(blanked[a_end:b_start])
            or (item := log.get(a.evidence_id)) is None
        ):
            continue
        sides: dict[int, dict[str, list[float]]] = {}
        for lf in leaves_of(item):
            side = "before" if "before" in lf.labels else "after" if "after" in lf.labels else None
            if lf.record is not None and side and lf.value is not None:
                sides.setdefault(lf.record, {}).setdefault(side, []).append(lf.value)

        def holds(record: dict, side: str, text: str) -> bool:
            return any(_reads_as(r, record.get(side, [])) for r in _readings(text))

        backwards = any(
            holds(r, "before", b.text) and holds(r, "after", a.text) for r in sides.values()
        )
        forwards = any(
            holds(r, "before", a.text) and holds(r, "after", b.text) for r in sides.values()
        )
        if backwards and not forwards:
            problems.append(
                f"{label}: 'from {a.text} to {b.text}' runs the wrong way; in {item.id} the "
                f"change is from {b.text} to {a.text}."
            )
    return problems


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
    leaves_of=None,
) -> list[str]:
    """Check declared numbers, then that the text holds no number left undeclared.

    No number is parsed out of prose: the model says what each written number is worth and
    where it comes from; the only parsing left is spotting digits nobody declared.
    """
    problems = []
    leaves_of = leaves_of or _leaves
    text = _normalize(text)
    declared = [d.model_copy(update={"text": _normalize(d.text)}) for d in declared]
    masked = _mask(text)
    blanked = _blank(text, WELL, DATE, CALENDAR_WINDOW, REFERENCE)  # positions as in text
    # A declaration with no number in the text hides nothing (every written number must be
    # declared anyway), so it costs no correction round; it is only named as a hint below.
    unused = [d for d in declared if not _standalone(masked, d.text)]
    masked_dates = _mask_keep_dates(text)
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
        elif (wrong_row := _row_mismatch(masked_dates, number.text, fits, item)) is not None:
            problems.append(
                f"{label}: '{number.text}' is in {where}, but not in its {wrong_row[0]} row, "
                f"which the text ties it to; that row has {wrong_row[1]}."
            )
        elif wrong := _attribution(text, blanked, number.text, fits, item, leaves_of(item)):
            problems.append(f"{label}: '{number.text}' {wrong}. Write what the evidence says.")
        elif record := _record_mismatch(masked_dates, number.text, fits, leaves_of(item)):
            problems.append(
                f"{label}: '{number.text}' is in {where}, but the text ties it to {record[0]}, "
                f"and {record[1]}. Give each number the date of its own event."
            )

    problems += _reversed(label, blanked, declared, unused, log, leaves_of)
    # Numbers inside a quote are checked with the quote.
    rest = _mask(QUOTED.sub(" ", _translations_outside(text)))
    for number in sorted(declared, key=lambda d: -len(d.text)):
        for start, end in reversed(_standalone(rest, number.text)):
            rest = rest[:start] + " " + rest[end:]
    leftover: list[str] = []

    # A value already declared may be written again in another form ("94 %" declared, "94%"
    # in the translation after a quote): it was checked once, it needs no second entry.
    def exact(text: str) -> set[float]:
        """A number as written, percent as a fraction: "94 %" and "94%" are 0.94, "0,80" and
        "80%" are 0.8; no rounding, so "2%" is not a declared "1.77%". Dotted thousands as
        the plan writes them ("9.000") also read as the whole number; "0.100" does not."""
        readings = _numbers_in_text(text, bool(DECIMAL_COMMA.search(text)))
        if DOTTED_THOUSANDS.fullmatch(text.strip()):
            readings += _numbers_in_text(text, True)
        return {round(v / 100 if suffix == "%" else v, 9) for v, _, suffix in readings}

    declared_values = set().union(*(exact(d.text) for d in declared)) if declared else set()
    for match in WRITTEN_NUMBER.finditer(rest):
        written = match.group(0).strip()
        # Either convention: "0,80" is 0 and 80, or 0.8; "1.234.567" is a number too.
        values = [n for european in (False, True) for n in _numbers_in_text(written, european)]
        if values and all(_free(*n, pool, free) for n in values):
            continue
        # Read as written: a Norwegian reading counts only with a decimal comma ("0,80").
        if exact(written) & declared_values and (
            exact(written) <= declared_values or DOTTED_THOUSANDS.fullmatch(written)
        ):
            continue
        if values and written not in leftover:
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
            f"{_suggest(leftover, citable, log)}"
        )
    return problems


def _suggest(written: list[str], citable: set[str] | None, log: EvidenceLog) -> str:
    """Declarations for undeclared numbers its evidence does hold, so a true number is
    declared rather than deleted ("21 months after first oil" is in E5 as 20.73)."""
    items = [log.get(i) for i in sorted(citable)] if citable is not None else list(log)
    found = []
    for text in written[:3]:
        for item in (i for i in items if i is not None):
            numbers = _Pool([item]).numbers
            hit = next((n for r in _readings(text) for n in numbers if _reads_as(r, [n])), None)
            if hit is not None:
                found.append(f'{{"text": "{text}", "value": {hit:g}, "evidence_id": "{item.id}"}}')
                break
    return f" These are in the evidence; declare them as {', '.join(found)}." if found else ""


def _well_key(name: str) -> str:
    return re.sub(r"\s+", "", name.removeprefix("15/9-")).upper()


def _wells_without_evidence(claim: str, detail: str, cited: list[Evidence]) -> list[str]:
    """Wells the finding names that none of its evidence is about. Evidence without a well
    argument (SQL, field-wide tools, calculations, report searches without a filter) may
    hold any well, so it covers every name. The detail may also name a well the results
    name (F-12's shut-in "field_wide with F-14"); the claim, what a reader takes away, only
    wells the evidence is about."""
    if any(e.arguments.get("well") is None for e in cited):
        return []
    about = {_well_key(str(e.arguments["well"])) for e in cited}
    mentioned = about | {
        _well_key(m.group(0))
        for e in cited
        for m in WELL.finditer(json.dumps(e.result, default=str))
    }
    named = dict.fromkeys(m.group(0).strip() for m in WELL.finditer(claim))
    strangers = [w for w in named if _well_key(w) not in about]
    in_detail = dict.fromkeys(m.group(0).strip() for m in WELL.finditer(detail))
    return strangers + [w for w in in_detail if _well_key(w) not in mentioned | set(strangers)]


def _unhedged_warned(
    summary: str, declared: list[DeclaredNumber], log: EvidenceLog, calcs: dict[str, _Calc]
) -> list[str]:
    """Summary numbers from evidence with warnings (a poor fit, a calculation from one)
    that their sentence states as settled. A finding marks such a number inference; the
    summary has no label, so the sentence itself must say it is uncertain."""
    text = _normalize(summary)
    masked = _mask(text)
    ends = [m.start() for m in FULL_SENTENCE.finditer(text)] + [len(text)]
    problems = []
    for number in declared:
        item = log.get(number.evidence_id)
        if item is None:
            continue
        warnings = item.warnings or (calcs[item.id].warnings if item.id in calcs else ())
        if not warnings:
            continue
        poor = any(w.startswith(("poor fit", "no decline")) for w in warnings)
        if (
            poor
            and item.id in calcs
            and calcs[item.id].inputs >= 2
            and _standalone(masked, _normalize(number.text))
        ):
            # A ratio or difference of two unreliable numbers is noise however it is hedged
            # ("decline sped up 7.89 times" from a plateau fit that has no decline rate).
            problems.append(
                f"Summary: '{number.text}' compares numbers ({item.id}) that rest on evidence "
                f"with warnings ({warnings[0]}). Leave it out of the summary; a finding may keep "
                "it, marked inference."
            )
            continue
        for start, _ in _standalone(masked, _normalize(number.text)):
            end = next(e for e in ends if e >= start)
            begin = max([0] + [e for e in ends if e < start])
            if not HEDGE.search(text[begin:end]):
                problems.append(
                    f"Summary: '{number.text}' comes from {item.id}, which has warnings "
                    f"({warnings[0]}), but its sentence states it as settled. Say it is "
                    "uncertain in the same sentence (e.g. 'the fit suggests', 'likely') or "
                    "leave it out of the summary."
                )
                break
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


def lookup_spans(text: str | None, log: EvidenceLog | None) -> list[dict]:
    """Spans for a text whose numbers are looked up rather than declared (a finding's
    detail): each number with the first evidence that holds it, and each quote."""
    if not text:
        return []
    spans = [s for s in display_spans(text, [], log) if s["kind"] == "quote"]
    plain = text.replace("\u2212", "-").replace("\u2013", "-")
    masked = _blank(plain, DATE, WELL, CALENDAR_WINDOW, REFERENCE, DAY_MONTH_YEAR, MONTH_YEAR)
    pools = [(e.id, _Pool([e])) for e in log or []]
    for match in WRITTEN_NUMBER.finditer(masked):
        found = _numbers_in_text(match.group(0))
        if len(found) != 1:
            continue
        value, decimals, suffix = found[0]
        source = next((i for i, pool in pools if pool.has_number(value, decimals, suffix)), None)
        if source:
            spans.append(
                {
                    "kind": "number",
                    "start": match.start(),
                    "end": match.end(),
                    "value": value,
                    "evidence_id": source,
                }
            )
    return sorted(spans, key=lambda s: (s["start"], s["kind"] != "quote"))


def display_spans(
    text: str | None, numbers: list[DeclaredNumber] | None, log: EvidenceLog | None = None
) -> list[dict]:
    """Where the answer panel marks the text, by the checker's own rules: each quoted
    passage (with the first evidence that holds it, given the log), and each declared
    number where it stands as a whole number (not inside a date, a well name or a window),
    with the evidence it came from. Positions index the text as written, so the UI only
    slices; it re-implements no rule."""
    if not text:
        return []
    spans: list[dict] = []
    for m in QUOTED.finditer(_translations_outside(text)):
        span = {"kind": "quote", "start": m.start(), "end": m.end()}
        source = next((e.id for e in log or [] if _quoted_in(m.group(1), _Pool([e]))), None)
        if source:
            span["evidence_id"] = source
        spans.append(span)
    plain = text.replace("\u2212", "-").replace("\u2013", "-")  # same length
    masked = _blank(
        plain, DATE, WELL, CALENDAR_WINDOW, REFERENCE, DAY_MONTH_YEAR, MONTH_DAY_YEAR, MONTH_YEAR
    )
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


def finding_text(finding: Finding) -> str:
    """A finding's claim and detail as one text: for dates, quotes, causes and wells. Its
    numbers are checked apart: the claim's declared, the detail's looked up."""
    return finding.claim + (f"\n{finding.detail}" if finding.detail else "")


def check_submission(submission: Submission, log: EvidenceLog, question: str = "") -> list[str]:
    """Return problems as sentences for the model; an empty list means accepted."""
    free = _question_numbers(question)
    free_dates = REPORT_DATES | set(DATE.findall(_normalize(question)))
    calcs = _calculations(log)
    sound = [e for e in log if not (e.id in calcs and calcs[e.id].unsupported)]
    walked: dict[str, list[_Leaf]] = {}

    def leaves_of(item: Evidence) -> list[_Leaf]:
        if item.id not in walked:
            walked[item.id] = _leaves(item)
        return walked[item.id]

    days = sorted({lf.day for e in log for lf in leaves_of(e) if lf.day and len(lf.day) == 10})
    unowned = free_dates | ({days[0], days[-1]} if days else set())

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
        text = finding_text(finding)
        ids = ", ".join(finding.evidence_ids)
        for item in cited:
            if item.id in calcs and calcs[item.id].unsupported:
                problems.append(
                    f"Finding {n}: {item.id} calculates from "
                    f"{', '.join(calcs[item.id].unsupported)}, which no evidence contains. "
                    "Calculate only with numbers that tool results returned."
                )
        pool = _Pool(e for e in cited if not (e.id in calcs and calcs[e.id].unsupported))
        numbers, dates = _unsupported(text, pool, free, free_dates)
        if finding.numbers is not None:
            # The claim declares its numbers; the detail is supporting values copied from
            # the tools, each looked up in the cited evidence instead (declaring them all
            # cost every run of Oct 27 evening a round, and the model deleted the details).
            supported = _looked_up(finding.detail or "", pool, free)
            if supported:
                problems.append(
                    f"Finding {n}: in its detail, {', '.join(supported)} not found in {ids}. "
                    "Copy the values as the tools give them, or leave them out."
                )
            problems += _declared(
                f"Finding {n}",
                finding.claim,
                finding.numbers,
                log,
                calcs,
                set(finding.evidence_ids),
                pool,
                free,
                leaves_of,
            )
        elif numbers:  # an answer from before declared numbers: read them from the text
            problems.append(
                f"Finding {n}: {', '.join(numbers)} not found in {ids}. Cite the evidence it "
                "comes from; compute derived numbers with calculate; or remove them."
            )
        if dates:
            problems.append(f"Finding {n}: date(s) {', '.join(dates)} not found in {ids}.")
        # The claim only: a detail is tool-speak ("field_wide with F-1 C, F-15 D") whose
        # clauses do not say whose date it is.
        elif misdated := _date_attribution(_normalize(finding.claim), cited, leaves_of, unowned):
            problems.append(
                f"Finding {n} gives the well its sentence names date(s) {', '.join(misdated)}. "
                "Cite evidence about that well, or name the well the date belongs to."
            )
        missing_quotes = _missing_quotes(text, pool)
        if missing_quotes:
            listed = "; ".join(f"'{q[:80]}'" for q in missing_quotes)
            problems.append(
                f"Finding {n}: quoted text not found in {ids}: {listed}. Quote the evidence "
                "word for word (the original quote or its statement_en) or drop the quotes."
            )
        # A cause inside a quote is the report's claim, which the finding only reports.
        cause = CAUSAL.search(_own_words(text, pool))
        if finding.confidence == "fact" and cause:
            problems.append(
                f"Finding {n} states a cause ('{cause.group(0)}') but is marked fact. The data "
                "shows timing and size, not causes: mark it inference."
            )
        strangers = _wells_without_evidence(finding.claim, finding.detail or "", cited)
        if strangers:
            problems.append(
                f"Finding {n} names {', '.join(strangers)}, but {ids} "
                f"{'is' if len(cited) == 1 else 'are'} about other wells. Cite evidence for "
                f"{'that well' if len(strangers) == 1 else 'those wells'} or leave "
                f"{'it' if len(strangers) == 1 else 'them'} out."
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
            leaves_of,
        )
        problems += _unhedged_warned(submission.summary, submission.summary_numbers, log, calcs)
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
