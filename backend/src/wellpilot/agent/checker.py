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

import ast
import dataclasses
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field

from wellpilot.agent.evidence import Evidence, EvidenceLog
from wellpilot.ingest.operations import WELLS as WELLBORES
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
# A well name's digits are no day: "for P-2 October 2018" is P-2's October, not 2 October
# ("post-24 January 2010" keeps its day).
# ("30.Jun.2013", as the completion reports' tables write it, too.)
DAY_MONTH_YEAR = re.compile(
    rf"(?<![A-Z]-)\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+|\.){_MONTH}\s*{_YEAR}\b"
)
MONTH_DAY_YEAR = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR}\b")
MONTH_YEAR = re.compile(rf"\b{_MONTH}\s+{_YEAR}\b")
DOTTED_DATE = re.compile(rf"\b(\d{{1,2}})\.(\d{{1,2}})\.{_YEAR}\b")
SLASHED_DATE = re.compile(rf"\b(\d{{1,2}})/(\d{{1,2}})/{_YEAR}\b")
# A stretch of a year: "Q2 2010", "early 2010", "spring 2009" (as months, inclusive).
PERIOD = re.compile(
    rf"\b(?:(?P<q>Q[1-4])|(?P<part>early|mid|late|end of|start of)|"
    rf"(?P<season>spring|summer|autumn|fall|winter))[\s-]+(?:of\s+)?{_YEAR}\b(?![-/]\d)",
    re.IGNORECASE,
)
PERIOD_MONTHS = {
    "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
    "early": (1, 4), "start of": (1, 3), "mid": (5, 8), "late": (9, 12), "end of": (10, 12),
    "spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11), "winter": (12, 2),
}  # fmt: skip
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
SPACED_THOUSANDS = re.compile(r"\b\d{1,3}(?:[ \u00a0\u202f]\d{3})+\b")
# A number may run straight into a unit ("98765Sm3") or end a sentence, but not run into
# more digits.
# Suffixes that scale a number: "4.58 MSm3", "276 kSm3", "1.2 mill Sm3", "75.6 percent".
NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s*(%|percent\b|pct\b|x\b|×|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm)|km\b|k\b))?"
    r"(?!\d|\.\d)"
)
# A number as the text writes it, so a rejection can name it that way: "0,80", "4 050",
# "1,133,053", "2.5x". (Parsing it into values may split or rewrite it.)
WRITTEN_NUMBER = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)*(?:[eE][-+]?\d+)?"
    r"(?:\s*(?:%|percent\b|pct\b|x\b|×|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm)|k\b))?"
)
SUFFIX_AFTER = re.compile(
    r"\s*(?:%|percent\b|pct\b|million\b|mill\b\.?|thousand\b|M(?=Sm)|k(?=Sm)|km\b)"
)
# The rest of a range up to a scale word: " to 0.47" in "0.32 to 0.47 million", "-33" in
# "31-33%".
RANGE_SCALE = re.compile(
    r"\s*(?:(?:-|to|and)\s*(?:about\s+)?\d[\d.,]*"
    r"(?=\s*(?:million\b|mill\b|thousand\b|M(?=Sm)|k(?=Sm)))"
    # a percent only across a dash or "to": "0.60 and about 36%" is two numbers
    r"|(?:-|to)\s*\d[\d.,]*(?=\s*(?:%|percent\b)))"
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
    "km": (1e3,),
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

    def numeric(day: str, month: str, year: str) -> str:
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return f"{day}.{month}.{year}"

    # "12.10.2010" (Norwegian), "10/12/2010" (month first unless the first part is past 12)
    text = DOTTED_DATE.sub(lambda m: numeric(m[1], m[2], m[3]), text)
    text = SLASHED_DATE.sub(
        lambda m: numeric(m[1], m[2], m[3]) if int(m[1]) > 12 else numeric(m[2], m[1], m[3]),
        text,
    )
    text = DAY_MONTH_YEAR.sub(lambda m: iso(m[3], m[2], m[1]) or m[0], text)
    text = MONTH_DAY_YEAR.sub(lambda m: iso(m[3], m[1], m[2]) or m[0], text)
    return MONTH_YEAR.sub(lambda m: iso(m[2], m[1], None) or m[0], text)


DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2212", "-"))


class _Canonical(dict):
    """str.translate table, filled as characters come: every space separator (no-break,
    narrow, thin, ideographic) as a space, the hyphens, the en dash and the minus sign as
    "-", fullwidth ASCII as ASCII. The em dash, quote marks and the ellipsis stay: rules
    read them."""

    def __missing__(self, code: int) -> int | str:
        char = chr(code)
        if code < 128:
            value: int | str = code
        elif unicodedata.category(char) == "Zs":
            value = " "
        elif char in "\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d":
            value = "-"
        elif 0xFF01 <= code <= 0xFF5E:
            value = chr(code - 0xFEE0)
        else:
            value = code
        self[code] = value
        return value


CANONICAL = _Canonical()
WORD_APOSTROPHE = re.compile(r"(?<=\w)[\u2019\u02bc](?=[^\W\d_])")
# "69 per cent" as " percent": the same length, so positions still hold.
PER_CENT = re.compile(r"\bper (?=cent\b)", re.IGNORECASE)


def canonical(text: str) -> str:
    """Text with its typography made plain, one character for one, so every position still
    indexes the text as written: "2010\u201107\u201118", "F\u201112" and "4\u202f050" read as
    "2010-07-18", "F-12" and "4 050"; "F-12\u2019s" as "F-12's". The model's typography varies
    run to run; each variant used to be a false rejection of its own."""
    plain = WORD_APOSTROPHE.sub("'", text.translate(CANONICAL))
    return PER_CENT.sub(" per", plain) if "cent" in plain else plain


def _normalize(text: str) -> str:
    """The claim as the checker reads it: plain typography (canonical), ISO dates."""
    return normalize_dates(canonical(text))


# Unit conversions a calculation may use without citing them (besides powers of ten): days,
# hours, weeks and months; barrels per Sm3 and back; cubic feet per Sm3.
CONSTANTS = {7.0, 12.0, 24.0, 30.0, 30.44, 52.0, 60.0, 365.0, 365.25, 3600.0, 86400.0}
CONSTANTS |= {6.2898, 0.159, 0.158987, 35.3147}
# Causes are interpretations: the data shows timing and size, not why.
# "Water drive" and "gas drive" are recovery mechanisms, and "the fit results in E9" points
# at evidence: neither is a claim about why. The verb is one: "water drove the fall".
CAUSAL = re.compile(
    r"\b(because|caus(?:e|ed|es|ing)|"
    r"(?<!water )(?<!gas )(?<!aquifer )(?<!depletion )(?<!gravity )(?<!water-)drives?\b|"
    r"dr(?:ove|iven|iving|ivers?)|"
    r"due to|led to|leads? to|result(?:s|ed)? (?:from|in)(?! E\d)|(?:as a )?result of|"
    r"responsible for|(?:was|were|is|are|be) (?:\w+ )?responsible|attribut(?:able|ed) to|"
    r"triggered by|owing to|thanks to|brought on by|stemming from|resulting from|"
    r"(?:the )?outcome of|follow(?:s|ed|ing) from|that (?:[\w-]+ ){1,4}produced(?=\s*[.;,]|$)|"
    r"(?:a )?consequence of|contribut(?:ed|es|ing) to|stem(?:s|med)? from|induced by|"
    r"the reason|accounts? for|explain(?:s|ed by)?|reflect(?:s|ed|ing)?|(?:direct )?effect of|"
    r"(?:a |the )?direct effect|is why|leading to|lead(?:s|ing)? to|resulting in|"
    r"squeez(?:e|ed|es|ing)|forc(?:e|ed|es|ing) (?:\w+ )?out|"
    r"made (?:[\w-]+ ){0,3}?(?:drop|fall|rise|climb|decline|increase|decrease|go (?:up|down))|"
    r"push(?:ed|es|ing)? (?:[\w-]+ ){0,3}?(?:down|up|out)|brought (?:[\w-]+ ){0,3}?(?:down|up)|"
    r"let (?:the )?[\w-]+ (?:recharge|recover|rebuild|repressur\w*)|gave rise to|"
    r"came from (?:the )?(?:shut[\s-]?in|workover|plug|job|choke|injection|stimulation)|"
    r"(?<!water )(?<!water-)cut (?:F-\d+|its|their|the well|oil))\b",
    re.IGNORECASE,
)
# A conclusion drawn in a fact ("..., so water displaced oil", "which shows the water came
# from the reservoir"): a finding that draws one is an inference. Judged in claims only.
CONSEQUENCE = re.compile(
    r",\s*(?:and\s+)?(?:so|thus|hence|therefore)\b(?!\s+(?:far|called|much|many|long)\b)|"
    r"\b(?:which|this|that)\s+(?:shows|means|proves|implies)\b|\bis behind\b|\bdid it\b|"
    r"\bproduced (?:the|this|that|most|all|part of the)\s+(?:\w+\s+){0,2}?"
    r"(?:drop|fall|rise|decline|increase|change|jump|gain|loss)\b",
    re.IGNORECASE,
)
# A quoted passage: opened after a space or "(", closed before a space or punctuation, so
# apostrophes inside words ("oil's") do not end it.
# Curly quotes and guillemets open after any non-letter ("—“..."), straight quotes only
# after a space, "(" or the start. A quote cannot run across another opening quote, so two
# scare-quoted words ('plateau' ... 'decline') are not read as one quotation.
QUOTED = re.compile(
    r"""(?:(?:(?<=[\s(\[])|^)['"]|(?<!\w)[‘“«„])((?:(?!\s['"‘“«„]).){15,}?)['"’”»“]"""
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
# Within it, the hedge must govern what it hedges: before it, or right after it in the
# same clause ("359 thousand, likely"), not a "may" about something else ("as engineers may
# expect", "which could be costly").
HEDGE = re.compile(
    r"\b(likely|probabl[ey]|possibly|perhaps|presumabl[ey]|plausibl[ey]|consistent with|"
    r"suggests?|seems?|appears? to|could|would|may|might|hypothes\w*|interpret\w*|"
    r"points? to|impl(?:y|ies)|uncertain\w*|low confidence)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"(?<=[.;!?])\s+")
HEDGE_AFTER_WORDS = 4


def _hedged(sentence: str, start: int, end: int, stops: str = ",;") -> bool:
    """Is what sits at sentence[start:end] hedged? A hedge before it in the sentence counts;
    after it, a plain hedge ("the fit suggests", "likely", "low confidence") counts anywhere
    in the sentence, a modal verb (may, could) only within a few words with no comma (or
    other stop) between: "as engineers may expect" or ", which could be costly" is about
    something else."""
    for m in HEDGE.finditer(sentence):
        if m.start() < start:
            return True
        if m.start() < end:
            continue
        between = sentence[end : m.start()]
        modal = re.fullmatch(r"could|would|may|might", m.group(0), re.I)
        near = len(between.split()) <= HEDGE_AFTER_WORDS and not any(c in between for c in stops)
        if near:
            return True
        # After a semicolon, a hedge covers what came before only if it points back to it
        # ("...; the fit suggests so"), not when it opens its own claim ("...; the figure
        # likely shows ...").
        clause = sentence[m.start() :].split(";")[0]
        own_claim = ";" in between and not re.search(r"\b(?:so|this|that|these|those|it)\b", clause)
        if not modal and not own_claim:
            return True
    return False


# A warned number's hedge may come after a semicolon ("...335 thousand; the fit suggests
# ..."), so that rule reads whole sentences; a cause must be hedged in its own clause.
FULL_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class DeclaredNumber(BaseModel):
    text: str = Field(
        description="The number exactly as written in the text, e.g. '2 870', '38%', '0.3127', "
        "'2.47 million'."
    )
    value: float = Field(
        description="Its full value as the evidence has it: 2870 for '2 870' or '2.87k', "
        "2468915 for '2.47 million', 0.38 or 38 for '38%'."
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
    """Numbers and strings the model wrote into a SQL query's SELECT list, or into its own
    Python: echoes, not data."""
    if item.tool == "run_python":
        return _literals(str(item.arguments.get("code", "")))
    if item.tool == "look_at_page":
        return _unmeasured(item)
    if item.tool != "run_sql":
        return frozenset(), frozenset()
    return _echoes(str(item.arguments.get("sql", "")))


def _unmeasured(item: Evidence) -> tuple[frozenset[float], frozenset[str]]:
    """A page reading SAM 3.1 confirmed carries no warning, but only its compared values are
    measured: the model's free answer, its labels and its other readings stay its own words
    ("a 9,500 Sm3/d plateau" in the answer is no evidence for a fact, numbers or quoted)."""
    result = item.result if isinstance(item.result, dict) else {}
    if item.warnings or not result.get("cross_check"):
        return frozenset(), frozenset()

    def numbers(value: Any) -> set[float]:
        found = set()
        for leaf in _flatten(value):
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, int | float):
                found.add(abs(float(leaf)))
            elif isinstance(leaf, str):
                for european in (False, True):
                    found |= {abs(n) for n, _, _ in _numbers_in_text(leaf, european)}
        return found

    measured = numbers(result.get("measured_by_sam")) | {
        abs(float(c[k]))
        for c in result["cross_check"]
        if c.get("agree")
        for k in ("muse", "sam")
        if isinstance(c.get(k), int | float)
    }
    said = numbers([result.get("answer"), result.get("readings")])
    # The free answer is the model's prose: not quotable as evidence either ('the reading
    # says "a plateau of about 9,500 Sm3/d"' is no fact).
    answer = result.get("answer")
    return frozenset(said - measured), frozenset([answer] if isinstance(answer, str) else [])


def _literals(code: str) -> tuple[frozenset[float], frozenset[str]]:
    """The numbers and strings typed into the agent's code. A result equal to one is the
    model's own number, wherever the code put it. Integers up to 10 and unit constants (24,
    100, 1000) are not counted: code uses them to compute, and they need no evidence anyway.
    Code that does not parse has no result to judge."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset(), frozenset()
    numbers, strings = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            if isinstance(node.value, int | float) and math.isfinite(node.value):
                value = abs(float(node.value))
                if not (value <= 10 and value == int(value)) and not _is_constant(value):
                    numbers.add(value)
            elif isinstance(node.value, str):
                strings.add(node.value)
    return frozenset(numbers), frozenset(strings)


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
                    # Report text may use either convention, so keep both readings. A number
                    # the model typed is no evidence in printed text either.
                    for european in (False, True):
                        for v, _, suffix in _numbers_in_text(leaf, european):
                            if abs(v) not in typed_numbers:
                                self.numbers.append(v)
                                # "omlag 6 MSm3" is also 6 million, and a tool's "55% lower"
                                # also 0.55: a claim's "6 million" or "55%" declared as a
                                # fraction finds it either way.
                                self.numbers.extend(
                                    v * k
                                    for k in SCALES.get(suffix or "", ())
                                    if k >= 1000 or (k == 0.01 and suffix in ("%", "percent"))
                                )
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


def _free(
    value: float,
    decimals: int,
    suffix: str | None,
    pool: _Pool,
    free: set[float],
    year: bool = True,
) -> bool:
    """Whole numbers that need no evidence: integers up to 10, years (an evidence year, a
    year the data returns as a number, a report's year) and whole numbers the question
    itself names ("the next 12 months"). year=False: written as no year can be ("2,010",
    "2010 Sm3/d")."""
    if decimals != 0 or suffix or not value.is_integer():
        return False
    if value <= SMALL_INTEGER or value in free:
        return True
    return (
        year
        and 1900 <= value <= 2100
        and (str(int(value)) in pool.years or pool.has_number(value, 0, None))
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
    # A pair written as a list ("[3147,3161]", "3147,3161"): a comma before four digits is
    # no thousands separator, and a 4-digit decimal comma is not how tools write.
    listed = re.sub(r"(?<=\d),(?=\d{4}\b)", ", ", quotes_out)
    for match in WRITTEN_NUMBER.finditer(_mask(listed)):
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
# declaring. Any other duration ("45-day shut-in", "30 days of shut-in") does. 60 days is
# the window production_log measures a plug by ("in the 60 days after").
CALENDAR_WINDOW = re.compile(
    r"\b(?:12|24|30|60|90|180|365)[-\s](?:month|hour|day)\b"
    r"|\b(?:12\s+months|24\s+hours|(?:90|180|365)\s+days)\b"
    r"|(?:(?<=next )|(?<=first )|(?<=the ))60\s+days\b"
    r"|\b60\s+days(?=\s+(?:after|before|later|on average))"
    r"|\btwelve[-\s]months?\b"  # "the next twelve months", as "12 months" is
)
DECIMAL_COMMA = re.compile(r"\d,\d{1,2}(?!\d)")
DOTTED_THOUSANDS = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+")
# A reference to a part of a document names it; its digits are no value to declare:
# "Figure 4.2", "Tabell 4.5", "section 4.5.1", "§ 4.5.1", "page 23", "p. 23", "pages 23-25".
REFERENCE = re.compile(
    r"\b(?:fig(?:ure|ur)?|tab(?:le|ell)?|sections?|chapters?|kap(?:ittel|itel)?|appendix|"
    r"vedlegg|pages?|side|p)\.?\s*\d+(?:\.\d+)*(?:\s*[-–]\s*\d+)?\b|§\s*\d+(?:\.\d+)*|"
    # Model names: "SAM 3.1", "muse-spark-1.3".
    r"\bSAM[\s-]?\d+(?:\.\d+)?\b|\bmuse[\w-]*?\d+(?:\.\d+)+\b",
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


SCALED_KEYS = re.compile(r"msm3|ksm3|thousand|million|mill|mbbl|kbbl", re.IGNORECASE)
PERCENT_KEYS = re.compile(r"pct|percent", re.IGNORECASE)


def _scale_mismatch(fits: list[Reading], leaves: list["_Leaf"], item: Evidence) -> str | None:
    """A scale the text writes that the value does not have: "1,551 thousand Sm3/d" for a
    rate of 1551, "0.69%" for a water cut of 0.69. The number as written may stand for the
    value only where the evidence keeps that scale too ("4.58" under an msm3 key, "-55%")."""
    for number, unit, scales in fits:
        others = tuple(sc for sc in scales if sc != 1.0)
        if 1.0 not in scales or not others:
            continue
        values = [lf for lf in leaves if lf.value is not None]
        if any(_reads_as((number, unit, others), [lf.value]) for lf in values):
            return None
        at_one = [lf for lf in values if _reads_as((number, unit, (1.0,)), [lf.value])]
        if not at_one:
            return None
        held = at_one[0].value
        if others[0] > 1:
            # A calculation carries no unit: "1.76 + 2.68" adds the plan's MSm3, so "4.44
            # MSm3" is how its result reads (run d2f409964a50 ended unverified on this).
            if item.tool == "calculate":
                return None
            if not any(
                SCALED_KEYS.search(lf.labels) or lf.unit in ("M", "k", "million", "mill", "mill.")
                for lf in at_one
            ):
                return (
                    f"is written as {number:g} x {others[0]:g}, but {item.id} has {held:g}; "
                    "drop the scale word"
                )
        elif (
            abs(number) < 1
            and not (item.tool == "calculate" and re.search(r"\*\s*100\b", str(item.arguments)))
            and all(
                abs(lf.value) <= 1 and lf.unit != "%" and not PERCENT_KEYS.search(lf.labels)
                for lf in at_one
            )
        ):
            return (
                f"is written as {number:g}%, a share of {number / 100:g}, but {item.id} has "
                f"{held:g}, which is {held * 100:g}%"
            )
    return None


# A year is four plain digits with no unit after them: "2,010 Sm3/d" and "2010 bar" are not.
YEAR_UNIT = re.compile(
    r"\s*(?:%|percent|pct|x\b|×|bar|psi|m\b|mmd|mtvd|ft\b|feet|°|deg|k?sm|m?bbl|barrels|stb|"
    r"days?|weeks?|months?|hours?)",
    re.IGNORECASE,
)


def _yearlike(written: str, after: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", written.strip())) and not YEAR_UNIT.match(after)


def _standalone(text: str, written: str) -> list[tuple[int, int]]:
    """Spans where a declared text stands as a whole number: "2" in "2 years", not the 2 in
    "2 100", "12", "0.2" or "Sm3"; "276,078" in "P50 276,078"."""
    pattern = re.compile(rf"(?<![\w.,]){re.escape(written)}(?!\d|[.,]\d|[ \u00a0\u202f]\d{{3}}\b)")
    # GROUP_BEFORE ends at the match and spans at most 5 characters ("x123 "): searching only
    # that window keeps a long text linear (from 0 it was quadratic: 9-14 s for a reader's
    # 2,000-character edit, audit Oct 31).
    return [
        m.span()
        for m in pattern.finditer(text)
        if not GROUP_BEFORE.search(text, max(0, m.start() - 6), m.start())
    ]


# A time of day ("from 2013-06-30 14:15 to ... 20:00") is part of its date.
TIME_OF_DAY = re.compile(r"(?<![\d.,:])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?!\d|[.,:]\d)")


def _mask(text: str) -> str:
    """Dates, times of day, well names and calendar windows blanked out: their digits are
    not numbers to declare."""
    text = TIME_OF_DAY.sub(" ", DATE.sub(" ", WELL.sub(" ", text)))
    return REFERENCE.sub(" ", CALENDAR_WINDOW.sub(" ", text))


# A number the text ties to a month or day: "0.85 in 2009-06", "1 146 Sm3/d on 2010-07-18".
# A word between a number and its date may be its unit, not "before" or "after": "97% before
# and 31% after in late 2014" ties neither to late 2014.
_UNIT_WORD = r"(?:(?!(?:before|after|later|earlier|prior)\b)[^\W\d]+\d?(?:/[^\W\d]+)?|%)"
TIED = (
    rf"(?:\s*{_UNIT_WORD}){{0,3}}?"  # a unit or two ("Sm3/d", "%"); no comma
    r"\s+(?:in|on|at|during|by|for)\s+(\d{4}-\d{2}(?:-\d{2})?)\b"
    r"(?!\s*(?:to|until|through|and|-|–)\s*\d{4})"  # a range of dates is no one row
)
PERIOD_TIED = re.compile(
    rf"(?:\s*{_UNIT_WORD}){{0,3}}?\s+(?:in|during)\s+(?:the\s+)?"
    r"(?P<part>early|mid|late)[\s-]+(?P<year>(?:19|20)\d{2})\b"
    r"(?!\s*(?:to|until|through|and|-|–)\s*(?:the\s+)?(?:early|mid|late|(?:19|20)\d{2}\b))",
    re.IGNORECASE,
)
DATE_FIRST = re.compile(
    # "reached 80% in 2012-06 and peaked at 86%": a new clause after "and" is not that month's.
    r"\b(?:in|on|during|for)\s+(\d{4}-\d{2}(?:-\d{2})?)\b,?\s+"
    r"(?:(?!(?:and|but|while|whereas|then)\b)[^\s\d,;.]+\s+){0,6}?"
    r"(?:was|were|is|had|made|produced|averaged|read|reached|at|of)\s+"
    r"(?:about|around|roughly|approximately|~)?\s*$",
    re.IGNORECASE,
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
        period = PERIOD_TIED.match(text[end:])
        if period:
            # "72% in early 2012": a level at a part of a year must stand in one of its
            # monthly rows.
            year, part = period["year"], period["part"].lower()
            months = {"early": range(1, 5), "mid": range(5, 9), "late": range(9, 13)}[part]
            wanted = {f"{year}-{m:02d}" for m in months}
            for table in _tables(item.result):
                columns = [str(c).lower() for c in table["columns"]]
                key = next((i for i, c in enumerate(columns) if c in DATE_COLUMNS), None)
                if key is None:
                    continue
                rows = [
                    r for r in table["rows"] if str(r[key])[:7] in wanted and len(str(r[key])) == 7
                ]
                if not rows:
                    continue
                values = [
                    float(v)
                    for row in rows
                    for v in row
                    if isinstance(v, int | float) and not isinstance(v, bool)
                ]
                if not any(_reads_as(r, values) for r in readings):
                    return f"{part} {year}", f"rows {min(wanted)} to {max(wanted)}"
            continue
        # A design limit or capacity written with a date in its sentence is the plan's, not
        # that month's ("hit 80% in 2012-06 but never exceeded its capacity of 13,000").
        if re.search(
            r"\b(?:capacit\w*|limits?|maximum)\s*(?:of|at|is|was|:)?\s*(?:about|around|~)?\s*$",
            text[:start],
            re.I,
        ) or re.match(
            r"\s*(?:Sm3(?:/d)?\s*)?(?:(?:liquid|oil|water|injection)\s+)?(?:capacity|limit|design)",
            text[end:],
            re.I,
        ):
            continue
        tied = re.match(TIED, text[end:])
        # "...in 2013-12, while the choke averaged 77% in 2011": a number with its own year
        # after it is that year's, not the earlier date's.
        own_year = re.match(r"\s*(?:%|percent)?\s*(?:in|during|for)\s+(?:19|20)\d{2}\b", text[end:])
        if not tied and not own_year:
            # "In 2010-12, F-12's oil was 44,468": the date first, the level after it.
            begin = max([0, *(m.end() for m in CLAUSE.finditer(text[:start]))])
            tied = DATE_FIRST.search(text[begin:start])
        if not tied:
            continue
        when = tied.group(1)
        # "Cut to 40% in 2018-11": a level reached by a change during the month, whose row
        # averages the days before and after it; the next row holds it too.
        # ("cut to about 40%" too)
        reached = bool(
            re.search(r"\bto\s*(?:(?:about|around|roughly|approximately)\s+|~\s*)?$", text[:start])
        )
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
            if len(rows) == 2:
                # The next row counts only where the tied row has that column at all: "rose
                # to 862 Sm3/d in 2014-12" when the well did not produce in 2014-12 is
                # January's rate.
                rows = [rows[0]] + [
                    [v if rows[0][i] not in (None, 0) else None for i, v in enumerate(rows[1])]
                ]
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
    unit: str | None = None  # the suffix a text wrote after it ("%" in "-22%")
    dim: str | None = None  # what it is measured in: rate (Sm3/d), volume (Sm3), days, ...
    sign: int | None = None  # +1 a rise, -1 a fall, where the result says which
    # A calculation's result: the (evidence id, record) of the values it was computed from,
    # so a share calculated from the 3,210 m row is still that row's.
    origin: tuple[tuple[str, int], ...] = ()
    # A table cell's row date ("2010", "2010-07", "2010-07-18"), for _best_fit.
    row_day: str | None = None


# The unit a text writes right after a number: what the number is measured in. Digits that
# are in the evidence still mean something else when a volume is written as a rate, a total
# as a monthly figure, days as months, or Sm3 as barrels. Read the same way in claims and in
# the tools' own text ("81 days, 123,258 Sm3 oil produced"), with the plan's Norwegian units.
UNIT_AFTER = re.compile(
    r"\s*(?:(?:thousand|million|mill)\b\.?\s*)?"
    r"(?:(?P<bbl>(?:k|M)?(?:bbl|barrels?|stb|bopd|bwpd|bpd)\b)"
    r"|(?P<sm3>(?:k|M)?S?m(?:3|³)(?!\w))"
    # "Sm3 oil per month": the phase between the unit and its time base
    r"(?:\s+(?:of\s+)?(?:oil|water|liquid|gas)(?=\s+(?:per|a)\s+(?:month|year|day)\b))?"
    r"(?P<per>\s*/\s*(?:producing[\s-])?(?:d|day)\b|\s+(?:per|a)\s+(?:producing[\s-])?day\b"
    r"|\s+daily\b)?"
    r"(?P<other>\s*/\s*(?:month|mo|year|yr)\b|\s+(?:per|a)\s+(?:month|year)\b)?"
    r"|(?P<dur>days?|dager|weeks?|uker|months?|måneder|mnd|years?|år)\b"
    r"|(?P<pert>(?:per|a)\s+(?:month|year|annum)|/\s*(?:month|mo|year|yr))\b"
    r"|(?P<psi>psi[ag]?)\b|(?P<bar>bar[ag]?)\b|(?P<ft>ft|feet|foot)\b"
    r"|(?P<metres>m|metres?|meters?|m\s?MD|m\s?TVD)\b|(?P<degc>°\s?C|deg\s?C)\b|(?P<degf>°\s?F|deg\s?F)\b"
    # A weight on the string ("2500 m 4400 lbs" in a drilling report) is no depth.
    r"|(?P<weight>k?lbs?|kg|tonnes?|tons?)\b)",
    re.IGNORECASE,
)
DURATIONS = {
    "day": "days", "dager": "days", "week": "weeks", "uker": "weeks", "month": "months",
    "måneder": "months", "mnd": "months", "year": "years", "år": "years",
}  # fmt: skip
UNIT_NAMES = {
    "barrels": "barrels",
    "rate": "a rate per day (Sm3/d)",
    "rate_other": "a rate per month or year",
    "volume": "a volume (Sm3)",
    "days": "days",
    "weeks": "weeks",
    "months": "months",
    "years": "years",
    "psi": "psi",
    "bar": "bar",
    "ft": "feet",
    "metres": "metres",
    "degc": "°C",
    "degf": "°F",
    "weight": "a weight (lbs, kg, tonnes)",
    "per_month": "a rate per month",
    "per_year": "a rate per year",
}


def _written_unit(text: str, end: int) -> str | None:
    """The unit written right after position `end`: barrels, rate, rate_other (per month or
    year), volume, or a duration; None for none (or a percent, judged elsewhere)."""
    m = UNIT_AFTER.match(text, end)
    if not m:
        return None
    if m["bbl"]:
        return "barrels"
    if m["sm3"]:
        return "rate" if m["per"] else "rate_other" if m["other"] else "volume"
    if m["pert"]:
        return "per_month" if re.search(r"month|mo\b", m["pert"]) else "per_year"
    for unit in ("psi", "bar", "ft", "metres", "degc", "degf", "weight"):
        if m[unit]:
            return unit
    word = m["dur"].lower()
    return DURATIONS.get(word) or DURATIONS.get(word.rstrip("s"))


DURATION_TABLES = ("days", "weeks", "months", "years")
# Words that name an extreme: a single month's or day's value, whatever the grain.
EXTREME = (
    r"\b(?:peak\w*|maxim\w*|max|minim\w*|min|highest|lowest|top(?:ped)?|record|worst|best|"
    r"as\s+(?:high|low)\s+as)\b"
)
# Between two items of a list: "80%, 90% and 95%", "3% or 5%".
LIST_JOIN = re.compile(
    r"\s*(?:%|percent)?\s*(?:,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)(?:about\s+|~\s*)?"
)


def _key_dim(name: str, cell: bool = False) -> str | None:
    """What a result field measures in: "oil_rate_sm3_per_day" a rate, "zero_days" days,
    "cumulative_oil_sm3" a volume. A table cell's Sm3 ("oil_sm3" in a daily or monthly
    table) is a volume per row, which a text may call a rate or a total: not judged."""
    n = name.lower()
    # A decline rate is per year ("di", "di_per_year"); a text writing "per month" read it
    # twelve times too fast.
    if re.fullmatch(r"di|di_per_year|.*_per_year|decline_per_year", n):
        return "per_year"
    if re.fullmatch(r".*_per_month", n):
        return "per_month"
    if re.search(
        r"(?:^|_)rates?(?:_|$)|per_(?:producing_)?day|_per_d$|sm3_d$|(?:^|_)daily(?:_|$)|"
        r"(?:^|_)max_day_",
        n,
    ):
        return "rate"
    for unit in ("days", "weeks", "months", "years"):
        # "di_per_year" is a rate per year, not a number of years.
        if re.search(rf"(?:^|_){unit[:-1]}s?(?:_|$)", n) and "per_" not in n:
            return unit
    if re.search(r"sm3|bbl", n):
        return None if cell else "volume"
    # Pressures in bar, depths in metres, temperatures in °C: a text writing psi, feet or
    # °F has read another unit (the numbers would be 14.5, 3.28 or 1.8 times apart).
    if re.search(r"(?:^|_)bar(?:_|$)", n):
        return "bar"
    if re.search(r"(?:^|_)(?:m|mmd|md|tvd)$|depth|(?:^|_)m_md(?:_|$)", n):
        return "metres"
    if re.search(r"(?:^|_)(?:c|degc)$|temperature", n):
        return "degc"
    return None


RAW_NUMBER = re.compile(
    r"(?<![\w.,])-?\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?(?!\d)"
    r"|(?<![\w.,])-?\d+(?:\.\d+)?(?!\d)"
)
FALL_WORDS = r"lower|down|less|drops?|dropped|fell|falls?|falling|declin\w*|decreas\w*|shrank"
RISE_WORDS = (
    r"higher|up|more|rises?|rising|rose|increas\w*|grew|gains?|gained|jump\w*|climb\w*|"
    r"improv(?:ed|es|ing)"
)
# Words before a number in a tool's text that say which of a record's numbers it is: the
# change across a shift or between its periods, the calibrated range or the bootstrap one.
CONTEXT_WORDS = ("across", "between", "calibrated", "bootstrap")
# What a depth in well work is: a plug's, a perforation's, a packer's, a gauge's. Read from
# the words between a number and the one before it, in the tools' text and in claims alike.
WORK_WORDS = {"plug": r"plug", "perforation": r"perforat", "packer": r"packer", "gauge": r"gauge"}


def _work(between: str) -> list[str]:
    return [w for w, pattern in WORK_WORDS.items() if re.search(pattern, between[-40:], re.I)]


@lru_cache(maxsize=4096)
def _string_marks(text: str) -> dict[float, tuple[str | None, int | None, str]]:
    """For each number a tool's text writes: its unit, its direction ("55% lower" is a fall)
    and its labels: the quantity the text names it ("123,258 Sm3 oil"), its side of a change
    ("4,207 -> 1,888": before, after) and context words. Read by the claim's own rules."""
    blanked = _blank(text, WELL, DATE)
    marks: dict[float, tuple[str | None, int | None, str]] = {}
    for m in RAW_NUMBER.finditer(blanked):
        raw = m.group(0)
        try:
            value = abs(float(re.sub(r"[,\u00a0\u202f ]", "", raw)))
        except ValueError:
            continue
        start, end = m.span()
        after = blanked[end : end + 30]
        sign = (
            -1
            if raw.startswith("-")
            or re.match(rf"\s*(?:%|percent)?\s*(?:{FALL_WORDS})\b", after, re.I)
            else 1
            if re.match(rf"\s*(?:%|percent)?\s*(?:{RISE_WORDS})\b", after, re.I)
            else None
        )
        labels = []
        if quantity := _quantity(blanked, start, end):
            labels.append("q_" + quantity.replace(" ", "_").lower())
        if re.match(r"\s*(?:->|→|\bto\b)\s*-?\d", blanked[end:]):
            labels.append("before")
        elif re.search(r"\d\s*(?:->|→|\bto\b)\s*$", blanked[:start]):
            labels.append("after")
        near = blanked[max(0, start - 60) : start].lower() + blanked[end : end + 12].lower()
        labels += [f"ctx_{w}" for w in CONTEXT_WORDS if w in near]
        previous = [n.end() for n in RAW_NUMBER.finditer(blanked, 0, start)]
        labels += [f"work_{w}" for w in _work(blanked[previous[-1] if previous else 0 : start])]
        mark = (_written_unit(blanked, end), sign, "|".join(labels))
        marks.setdefault(value, mark)
        # The plan's and the reports' Norwegian decimals: "3126,2" is 3126.2 too.
        if re.fullmatch(r"-?\d+,\d{1,2}", raw):
            marks.setdefault(abs(float(raw.replace(",", "."))), mark)
    return marks


# Labels of a signed change (a step, a shift, a relative change), not a level.
CHANGE_LABELS = r"change|relative|step|shift|delta|diff|decline_pct|_drop|_rise"


# String fields that say what a record measures ("signal": "water_cut"): they label its values.
RECORD_LABEL_KEYS = {"signal", "kind", "metric", "column", "field"}
# A regime change's levels, measured in its signal's unit.
LEVEL_OF_KEYS = {"before_mean", "after_mean", "across_before", "across_after"}
# Keys that measure their record's signal: a regime change's means and changes.
SIGNAL_OF_KEYS = {
    "relative_change", "change_across", "change", "before_mean", "after_mean",
    "across_before", "across_after",
}  # fmt: skip
# Fields that date a record: an event's start and end, an issue's since, a row's date.
RECORD_DATE_KEYS = {"start", "end", "since", "date", "day", "month", "week"}


SIGNAL_DIMS = {"bar", "metres", "degc"}
SHARE_KEYS = r"share|ratio|cut|relative|fraction|pct|percent"


def _leaves(item: Evidence) -> list[_Leaf]:
    """Every number and date in a tool result, with its labels, well and record.

    The well comes from the call's well argument, from a record's "well" key, or from a
    table's "well" column; a subtree whose key starts with "field" (field_voidage) is
    field-wide. Model-typed SQL literals are left out, as in the pool.
    """
    typed_numbers, typed_strings = _typed_by_model(item)
    found: list[_Leaf] = []
    records = iter(range(1_000_000))

    def leaf(value: Any, labels: tuple[str, ...], well, record, days, names=(), cell=False):
        label = "|".join(labels)
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            if abs(float(value)) not in typed_numbers:
                # Its own key first ("months_at_or_above"), then its parent's
                # ("cumulative_oil_sm3" over "p50"); a table cell has only its column.
                # A list of rows named "months" or "days" holds months' values, not
                # numbers of months: its name gives its cells no unit.
                dim = next(
                    (
                        d
                        for i, n in enumerate(names)
                        if not (i and record is not None and n in DURATION_TABLES)
                        and (d := _key_dim(n, cell))
                    ),
                    None,
                )
                # A change by its own key ("relative_change"), not by its record's kind:
                # a regime change's before_mean is a level.
                change = any(re.search(CHANGE_LABELS, n) for n in names)
                sign = -1 if value < 0 else 1 if value > 0 and change else None
                # A record's signal gives its readings' unit when their own key does not:
                # a gauge failure on bhp_bar has its last good value in bar.
                if dim is None and not change and not re.search(SHARE_KEYS, label):
                    dim = next(
                        (d for n in labels[len(names) :] if (d := _key_dim(n)) in SIGNAL_DIMS),
                        None,
                    )
                # A regime change's means are of its signal: oil_rate's are Sm3/d.
                if dim is None and names and names[0] in LEVEL_OF_KEYS:
                    dim = next((d for n in labels[len(names) :] if (d := _key_dim(n))), None)
                found.append(_Leaf(float(value), None, label, well, record, days, None, dim, sign))
        elif isinstance(value, str) and value not in typed_strings:
            marks = _string_marks(value)
            for european in (False, True):
                for number, _, suffix in _numbers_in_text(value, european, signed=True):
                    if abs(number) not in typed_numbers:
                        dim, sign, extra = marks.get(abs(number), (None, None, ""))
                        sign = -1 if number < 0 else sign
                        if dim is None and sign is None and suffix is None:
                            dim = next(
                                (d for n in labels[1:] if (d := _key_dim(n)) in SIGNAL_DIMS),
                                None,
                            )
                        found.append(
                            _Leaf(
                                number,
                                None,
                                f"{label}|{extra}" if extra else label,
                                well,
                                record,
                                days,
                                suffix,
                                dim,
                                sign,
                            )
                        )
            for day in DATE.findall(normalize_dates(value)):
                found.append(_Leaf(None, day, label, well, record, days))

    def walk(node: Any, keys: tuple[str, ...], kinds: tuple[str, ...], well, record, days):
        # A value's labels: its own key (or column) and its parent's, and what the records
        # around it measure. Container names further up ("wor_fit") say nothing about it.
        if isinstance(node, dict) and isinstance(node.get("columns"), list):
            columns = [str(c).lower() for c in node["columns"]]
            at_well = columns.index("well") if "well" in columns else None
            at_day = next(
                (i for i, c in enumerate(columns) if c in DATE_COLUMNS or c == "year"), None
            )
            for row in node.get("rows") or []:
                row_well = str(row[at_well]) if at_well is not None and row[at_well] else well
                rid = next(records)
                start = len(found)
                # A table row is no event: its date tie is _row_mismatch's, so no dates here.
                for column, value in zip(columns, row, strict=False):
                    leaf(
                        value,
                        (*keys[-1:], column, *kinds),
                        row_well,
                        rid,
                        frozenset(),
                        (column,),
                        True,
                    )
                if at_day is not None and row[at_day] is not None:
                    day = str(row[at_day])[:10]
                    for i in range(start, len(found)):
                        found[i] = dataclasses.replace(found[i], row_day=day)
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
                # The plan's limits ("design_limits": {"water_cut": {"value": 0.8}}) are
                # limits, whatever their own keys say.
                marks = ("limit",) if re.fullmatch(r"(?:design_)?limits?", key) else ()
                walk(
                    child,
                    (*keys[-1:], key),
                    kinds + marks,
                    None if field_wide else here,
                    record,
                    days,
                )
        elif isinstance(node, list):
            for child in node:
                if isinstance(child, dict) and not isinstance(child.get("columns"), list):
                    child_days = frozenset(
                        str(child[k])[:10]
                        for k in RECORD_DATE_KEYS
                        if isinstance(child.get(k), str) and DATE.match(str(child[k]))
                    )
                    # A record inside a dated one (a log's shares inside its survey) is of
                    # that date.
                    start = len(found)
                    walk(child, keys, kinds, well, next(records), child_days or days)
                    # A row of a table written as records ({"month": "2014-11", ...}) has
                    # its row's date, as a column table's row has.
                    day = next(
                        (
                            str(child[k])[:10]
                            for k in child
                            if str(k).lower() in (*DATE_COLUMNS, "year")
                            and re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?\S*", str(child[k]))
                        ),
                        None,
                    )
                    if day and not child_days:
                        for i in range(start, len(found)):
                            found[i] = dataclasses.replace(found[i], row_day=day)
                else:
                    walk(child, keys, kinds, well, record, days)
        else:
            leaf(node, (*keys[-2:], *kinds), well, record, days, tuple(reversed(keys[-2:])))

    well = item.arguments.get("well")
    walk(item.result, (), (), str(well) if well else None, None, frozenset())
    return found


# Quantities a number may be named as, and how a tool result labels each one. A number the
# text names as one quantity must not be, in its evidence, only another.
QUANTITIES = {
    "water cut": (r"water[\s_-]?cut", r"water_?cut"),
    "oil cut": (r"oil[\s_-]?cut", r"oil_?cut"),
    "uptime": (r"uptime|on[\s-]stream", r"uptime|on_stream"),
    "choke": (r"choke", r"choke"),
    "GOR": (r"\bGOR\b|gas[\s/-]oil ratio", r"(?:^|[|_])gor(?:[|_]|$)"),
    "downhole pressure": (
        r"\bBHP\b|downhole pressure|bottom[\s-]?hole pressure",
        r"(?:^|[|_])bhp(?:[|_]|$)|downhole",
    ),
    "wellhead pressure": (r"\bWHP\b|wellhead pressure", r"(?:^|[|_])whp(?:[|_]|$)"),
    "WOR": (r"\bWOR\b|water[\s/-]oil ratio", r"(?:^|[|_])wor(?:[|_]|$)"),
    # The phase or stream a volume or rate is of, and the forecast's labels.
    "oil": (
        r"\boil\b(?![\s-]*(?:in[\s-]place|prices?|compan\w*|fields?|cut))",
        r"(?:^|[|_])oil(?:_|\||$)",
    ),
    "water": (
        r"\bwater\b(?![\s/_-]*(?:cut|oil|inj\w*|handling|breakthrough|drive|flood\w*))",
        r"(?:^|[|_])water(?!_?cut|_inj)(?:_|\||$)",
    ),
    "liquid": (r"\bliquids?\b", r"liquid"),
    "injection": (r"\binject(?:ion|ed|ing|ors?)\b", r"inj"),
    "P10": (
        r"\bP10\b|high (?:case|end|estimate)|upper (?:bound|end)",
        r"(?:^|[|_])p10(?:[|_]|$)",
    ),
    "P50": (r"\bP50\b|central estimate", r"(?:^|[|_])p50(?:[|_]|$)"),
    "P90": (
        r"\bP90\b|low (?:case|end|estimate)|lower (?:bound|end)",
        r"(?:^|[|_])p90(?:[|_]|$)",
    ),
    "calibrated range": (r"\bcalibrated\b", r"calibrated"),
    "change across": (r"\bacross\b", r"across"),
    "Pearson": (r"\bPearson\b", r"pearson"),
    "Spearman": (r"\bSpearman\b", r"spearman"),
}
_NAMES = re.compile("|".join(f"(?:{t})" for t, _ in QUANTITIES.values()), re.IGNORECASE)
RISE_BEFORE = re.compile(
    # "up" only as a change ("went up 55%", "up by 55%"), not "makes up 69%"
    r"\b(?:rose|risen|rising|increased|increasing|grew|grown|jumped|climbed|gained|improved|"
    r"(?:went|gone|goes|is|was|are|were|been|shifted|moved|stepped|ticked|edged|came)\s+up|"
    r"up(?=\s+by\b))"
    r"(?:\s+by)?(?:\s+(?:about|around|some|nearly|almost|roughly))?\s*[~≈]?\s*$",
    re.IGNORECASE,
)
FALL_BEFORE = re.compile(
    r"\b(?:fell|fallen|falling|dropped|dropping|declined|declining|decreased|decreasing|"
    r"(?<!shut )(?<!broke )(?<!slowed )(?<!wound )down|"
    r"lost|shrank)(?:\s+by)?(?:\s+(?:about|around|some|nearly|almost|roughly))?\s*[~≈]?\s*$",
    re.IGNORECASE,
)
FALL_AFTER = re.compile(
    r"^\s*(?:%|percent)?\s*(?:drop|decline|decrease|fall|loss|lower|down|less|"
    r"(?:below|under)(?=\s+(?:the\s+|its\s+|a\s+|an\s+)?(?:[\w%.,]+\s+){0,3}?"
    r"(?:limit|capacity|plan|design|target|threshold|maximum)\b))\b",
    re.IGNORECASE,
)
# Shares a well holds at a time: written as a change ("down 69%") they are misread.
LEVELS = r"water_?cut|uptime|on_stream|choke"
# A step, not a gradual shift: the words that make a change between two periods' means
# read as a jump at one date.
STEP_WORDS = re.compile(
    r"\b(?:abrupt\w*|sudden\w*|overnight|sharp(?:ly)?|step(?:ped|s)?[\s-]+(?:down|up)|plung\w*|"
    r"jump(?:ed|s|ing)?|spik(?:e|ed|es)|"
    r"collaps\w*|within (?:a few |\d+ |two |three )?days|in (?:a single|one|a) (?:week|day)|"
    r"in (?:\d+|two|three|four|five|six) (?:weeks|days))\b",
    re.IGNORECASE,
)
MODELS = re.compile(r"exponential|hyperbolic|harmonic", re.IGNORECASE)
# A number written as a duration: "21 months after first oil", "81 days".
DURATION_AFTER = re.compile(r"\s*(?:days?|weeks?|months?|years?|hours?)\b", re.IGNORECASE)
RISE_AFTER = re.compile(
    r"^\s*(?:%|percent)?\s*(?:increase|rise|jump|gain|growth|higher|more|up|"
    r"(?:above|over)(?=\s+(?:the\s+|its\s+|a\s+|an\s+)?(?:[\w%.,]+\s+){0,3}?"
    r"(?:limit|capacity|plan|design|target|threshold|maximum)\b))\b",
    re.IGNORECASE,
)
# A number tied to the date of one record (an event, an issue): "55% on 2010-05-02",
# "81 days since 2010-10-12". As TIED, with "around" and "since".
RECORD_TIED = TIED.replace(
    r"(?:in|on|at|during|by|for)",
    r"(?:in|on|at|during|by|for|around|since|after|from|following|before)",
)
CLAUSE = re.compile(r"[.;!?](?=\s)|\n")
# What may stand between a quantity's name and its number for the number to be that
# quantity: "water cut on <date> was 0.98", "uptime of 0.92". Any other word ("to",
# "limit", "producing") ends the name's reach.
LINK = re.compile(
    r"(?:'s)?(?:\s+|\s*[:=~≈]\s*|\b(?:of|at|is|was|were|on|in|by|reached|reaching|hit|"
    r"stood|near|about|around|approximately|roughly|averaged|averaging|first|then|now|"
    r"exceeded|above|below|median|mean|daily|monthly|weekly|rate|production|produced|"
    r"volume|capacity|peak|peaked|total|band|range|forecast|correlation|coefficient|r|"
    r"stayed|held|holding|remained|high|low|has|had|have|been|being|"
    # "uptime fell from about 97%": a verb of change, and "from" only right after one
    r"(?:rose|risen|rising|fell|fallen|falling|dropped|dropping|declined|declining|"
    r"increased|decreased|climbed|jumped|went(?: up| down)?|moved|shifted|changed)"
    r"(?:\s+(?:continuously|steadily|"
    r"gradually|sharply|slowly|quickly))?(?:\s+from)?)\b,?)*",
    re.IGNORECASE,
)


SAME_TIME = re.compile(
    rf"\bsame\s+time(?:\s+as)?\s+(?:(?:{WELL.pattern})[\s,]*(?:and\s+)?)*$", re.I
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
        # "(+366% same time F-1 C F-15 D)": wells set beside the one a result is about own
        # none of its numbers.
        return {
            _well_key(m.group(0))
            for m in WELL.finditer(part)
            if not SAME_TIME.search(part[: m.start()])
        }

    bounds = [0, *(m.end() for m in CLAUSE.finditer(text)), len(text)]
    begin = max(b for b in bounds if b <= start)
    end = min((b for b in bounds if b > start), default=len(text))
    part_end = text.find(", ", start, end)
    part_end = end if part_end < 0 else part_end
    part_begin = text.rfind(", ", begin, start)
    part_begin = begin if part_begin < 0 else part_begin
    # "F-14 ... about 61% and F-12's is 34% lower": a well's possessive owns what follows it
    # in its part, unless it is one of a list ("F-12's and F-14's water cuts").
    owners = list(WELL.finditer(text, part_begin, start))
    if owners and re.match(r"['’]s\b", text[owners[-1].end() :]):
        last = owners[-1]
        listed = re.search(
            rf"(?:{WELL.pattern})['’]s\s*(?:,|and|&)\s*$", text[part_begin : last.start()]
        )
        listed = listed or re.match(rf"['’]s\s*(?:,|and|&)\s*{WELL.pattern}", text[last.end() :])
        if not listed:
            return {_well_key(last.group(0))}
    if wells(text[part_begin:part_end]):
        return wells(text[begin:part_end])
    return wells(text[begin:start]) or wells(text[:begin]) or wells(text)


def _quantity(blanked: str, start: int, end: int, after: bool = True) -> str | None:
    """The quantity the text names a number as: "water cut 0.98", "water cut on <date> was
    0.98", "0.69 water cut", "53% choke"; None when it names none next to it. A name right
    after the number wins ("1 551 Sm3/d oil at 0.69 water cut": 0.69 is the water cut).
    Tool-speak ("oil_rate 185.7 water_cut 0.96") names a value before it: after=False."""
    name = None
    if after:
        # "0.69 water cut", "53% choke", "1 551 Sm3/d oil", "123 258 Sm3 of water"
        found = re.match(
            rf"\s*(?P<pct>%|percent)?(?:{UNIT_AFTER.pattern})?\s*(?:of\s+(?:the\s+|its\s+)?)?"
            # "34% less oil", "10% more water"
            rf"(?:(?:less|more)\s+)?(?P<name>{_NAMES.pattern})",
            blanked[end:],
            re.IGNORECASE,
        )
        # "at a liquid rate of 5 130 oil fell": a name with its own verb after it starts the
        # next clause; the number keeps the name before it.
        if found and re.match(
            rf"\s+(?:{FALL_WORDS}|{RISE_WORDS}|held|stayed|remained|went)\b",
            blanked[end + found.end() :],
            re.IGNORECASE,
        ):
            found = None
        if found:
            name = found.group("name")
            # "97% water": a share of the stream is its water cut.
            # ("38% of the water" is a share of the water, not a water cut.)
            share = found.group("pct") or blanked[start:end].rstrip().endswith("%")
            # ("shut off 32% water" is a share of the water a plug would cut off.)
            cut_off = re.search(
                r"\b(?:shut(?:s|ting)?\s+off|cut(?:s|ting)?(?:\s+off)?|remov\w+|block\w*|"
                r"isolat\w+|eliminat\w+)\s+(?:about\s+|around\s+|roughly\s+|~\s*)?$",
                blanked[:start],
                re.I,
            )
            if share and re.fullmatch(r"water", name, re.I) and " of" not in found.group(0):
                return None if cut_off else "water cut"
            # "water makes up 69% of its liquids": a share of the liquid, so a water or oil
            # cut by its subject, which is not read here.
            if share and " of" in found.group(0) and re.fullmatch(r"liquids?", name, re.I):
                return None
    if name is None:
        # "made water about 21 months after first oil": a duration, not an amount of water,
        # also when the unit is part of the declared text ("passed 5% water at 21 months").
        if DURATION_AFTER.match(blanked[end:]) or re.search(
            r"\b(?:days?|weeks?|months?|years?|hours?)\s*$", blanked[start:end], re.I
        ):
            return None
        before = blanked[max(0, start - 60) : start]
        # "a correlation between water cut and choke of 0.60": the number is the
        # correlation's, not the last quantity named before it.
        if re.search(
            r"\b(?:correlation|coefficient|Pearson|Spearman)\b[^.;]*\bbetween\b", before, re.I
        ):
            return None
        last = None
        for m in _NAMES.finditer(before):
            last = m
        if not (last and LINK.fullmatch(before[last.end() :])):
            return None
        name = last.group(0)
        # "when water was 30%": a share of the stream named before it is its water cut too;
        # "oil loss ~76.8% and water ~92.5%" (no verb) is a share of the water.
        if (
            re.fullmatch(r"water", name, re.I)
            and re.search(r"\b(?:was|is|were|are|at|reached|hit|stood)\b", before[last.end() :])
            and (
                blanked[start:end].rstrip().endswith("%")
                or re.match(r"\s*(?:%|percent)", blanked[end:])
            )
        ):
            return "water cut"
    return next(q for q, (text, _) in QUANTITIES.items() if re.fullmatch(text, name, re.I))


def _attribution(
    text: str,
    blanked: str,
    written: str,
    fits: list[Reading],
    item: Evidence,
    leaves: list[_Leaf],
    own_part: bool = False,
    spans: list[tuple[int, int]] | None = None,
    exact: float | None = None,
) -> str | None:
    """Why the number, though in its evidence, is not the one the text says, or None.
    Judged where the text writes it (spans: only there).

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
    # The declared value, where the evidence holds it exactly (at either scale of a
    # percentage), says which value is meant: "about 97%" declared 0.9665 is that month's
    # water cut, not an uptime near 0.97. Used for what the value is named, only.
    scales = [] if exact is None else [exact, exact * 100, exact / 100]
    # A declared value may be a rounding of the value it means ("0.676" for 0.6758): at its
    # own precision, which may also be another value's exact one.
    places = 0 if exact is None else max(0, -Decimal(repr(abs(exact))).as_tuple().exponent)

    def at_scale(lf: _Leaf) -> bool:
        return any(
            abs(lf.value - x) <= 1e-9 * max(1, abs(x))
            or (places and round(lf.value, places) == round(x, places) and x == exact)
            for x in scales
        )

    meant = [lf for lf in held if at_scale(lf)] or held
    # Names as written: "P50" is no page reference here.
    names = _blank(text, WELL, DATE, CALENDAR_WINDOW)
    for start, end in spans if spans is not None else _standalone(blanked, written):
        # A detail is tool-speak ("12 days, field_wide with F-14"): only a well named next
        # to the number is its well.
        part = _part(text, start)
        named = (
            set()
            if own_part and re.search(r"\bwith\b|field[\s_-]?wide|same\s+time", part, re.I)
            else {_well_key(m.group(0)) for m in WELL.finditer(part)}
            if own_part
            else _named(text, start)
        )
        wells = {lf.well for lf in held}
        if named and None not in wells and not {_well_key(w) for w in wells} & named:
            owners = ", ".join(sorted(wells))
            return f"is in {item.id} only for {owners}, not for the well its sentence names"
        quantity = _quantity(names, start, end, after=not own_part)
        # Tool-speak names a value before it ("water_cut 0.96") or after it ("51431 oil 99543
        # water"): with a name on both sides, a detail's number is not judged.
        if own_part and quantity and _quantity(names, start, end) != quantity:
            quantity = None
        # Tool-speak's "water 0.3107 to 0.7745" is the water cut: a fraction named water
        # that the evidence holds only as a water cut (a log's water share stays water).
        if (
            own_part
            and quantity == "water"
            and all(abs(lf.value) <= 1 for lf in held)
            and not any(re.search(QUANTITIES["water"][1], lf.labels) for lf in held)
        ):
            quantity = "water cut"
        if quantity == "change across":
            # "with 72% across": the change across the date itself, not between the periods.
            changes = written.rstrip().endswith(("%", "percent")) or all(
                lf.sign is not None for lf in held
            )
            if changes and not any(_is_across(lf) for lf in held):
                across = _across_text(leaves, {lf.record for lf in held})
                if across:
                    return (
                        f"is called the change across, but in {item.id} it is the change "
                        f"between the periods; across the change itself {item.id} has {across}"
                    )
        elif quantity:
            pattern = QUANTITIES[quantity][1]

            # What a value is, by its own key ("oil_sm3_during") or the name its text gives
            # it, not by the record it sits in: a shut-in's "on_stream_hrs" signal does not
            # make its rate_before an uptime.
            def own(lf: _Leaf) -> str:
                parts = lf.labels.split("|")
                # A change point's means and changes are of its signal ("oil_rate").
                if any(p in SIGNAL_OF_KEYS for p in parts[:2]) and "regime_change" in parts:
                    return "|".join(parts)
                return "|".join(parts[:2] + [p for p in parts if p.startswith("q_")])

            other = [
                q
                for q, (_, label) in QUANTITIES.items()
                if q != quantity and all(re.search(label, own(lf)) for lf in meant)
            ]
            # A percentage the model declared as a fraction may stand in the evidence as a
            # percent ("choke about 53%" declared 0.5317, choke_pct 53.17): either scale
            # names it.
            rescaled = (
                [lf for lf in leaves if lf.value is not None and at_scale(lf)]
                if written.rstrip().endswith(("%", "percent"))
                else []
            )
            named_so = any(re.search(pattern, lf.labels) for lf in [*meant, *rescaled])
            if not named_so and other:
                return f"is named {quantity}, but in {item.id} it is the {other[0]}"

        # One model's P50 written as the forecast ("forecast at about 302,000" for the
        # exponential fit's), whatever it is called, unless the text names the model.
        if (
            not own_part
            and all("by_model" in lf.labels for lf in meant)
            and not MODELS.search(_part(blanked, start))
        ):
            model = next((m for m in MODELS.findall(meant[0].labels)), "another")
            return f"is the {model} model's P50 in {item.id}, not the forecast's own"
        unit = _written_unit(blanked, end)
        # Tool-speak "oil_sm3_per_day 1850 days 48": "days" is the next value's name.
        if own_part and re.match(
            r"\s*[a-z_]*(?:days?|weeks?|months?|years?)\s+-?\d", blanked[end:], re.I
        ):
            unit = None
        dims = {lf.dim for lf in held}
        known = dims - {None}
        # A day's volume ("max_day_water_sm3": "10240 Sm3 on its worst day") is a volume too.
        daily_volume = unit == "volume" and any(
            re.search(r"(?:^|\|)[a-z_]*_sm3(?:\||$)", lf.labels) for lf in held
        )
        if unit and known and unit not in known and not daily_volume:
            known = " or ".join(UNIT_NAMES[d] for d in sorted(known))
            tail = " (1 Sm3 is 6.29 barrels)" if unit == "barrels" else ""
            return f"is written in {UNIT_NAMES[unit]}, but in {item.id} it is {known}{tail}"
        # "water cut reached 81%" for "zero_days: 81": a percentage is no count of days. (Not
        # judged for rates and volumes: a change in a rate is a percentage that carries the
        # rate's unit.)
        # A value under "before_60_days" may still be a share ("before_60_days|water_cut"):
        # only values measured in time and named as no share are judged.
        if (
            held
            and written.rstrip().endswith(("%", "percent"))
            and all(
                lf.dim in ("days", "weeks", "months", "years")
                and not re.search(
                    r"water_?cut|uptime|on_stream|choke|pct|percent|share|ratio|change",
                    lf.labels,
                    re.I,
                )
                for lf in held
            )
        ):
            known = " or ".join(UNIT_NAMES[d] for d in sorted(known))
            return f"is written as a percentage, but in {item.id} it is {known}"
        # "+55%" for a fall of 55%: the plus sign written into the number is a rise.
        if written.lstrip().startswith("+") and all(
            (lf.sign is not None and lf.sign < 0) or (lf.sign is None and lf.value < 0)
            for lf in held
        ):
            return f"is written as a rise (+), but {item.id} has it as a fall ({held[0].value:g})"
        # "-0.68" for a correlation of +0.68: a sign written out on a value the evidence
        # holds only the other way (changes and percentages are judged by their words).
        if (
            (
                written.lstrip().startswith("-")
                or (
                    blanked[max(0, start - 1) : start] in ("-", "−")
                    and not own_part
                    and re.match(r"[\s(]", blanked[max(0, start - 2) : start - 1] or " ")
                )
            )
            and not written.rstrip().endswith(("%", "percent"))
            and all(
                lf.value > 0 and not re.search(CHANGE_LABELS, lf.labels) and lf.sign is None
                for lf in held
            )
        ):
            return f"is written negative, but in {item.id} it is {held[0].value:g}"
        if DURATION_AFTER.match(blanked[end:]) and all(lf.unit == "%" for lf in held):
            return f"is written as a duration, but {item.id} has it only as a percentage"
        before = blanked[max(0, start - 40) : start]
        rise = (
            RISE_BEFORE.search(before)
            or RISE_AFTER.match(blanked[end:])
            or blanked[max(0, start - 1) : start] == "+"
        )
        fall = FALL_BEFORE.search(before) or FALL_AFTER.match(blanked[end:])
        # "a water cut of 67% under the 80% limit", "peaked at 67% below it": a level named
        # before the number, then compared; not a difference from the limit.
        if re.search(
            r"\b(?:of|at|was|were|is|are|reached|peaked\s+at|stood\s+at|=)\s*"
            r"(?:about|around|roughly|nearly|almost|only|just)?\s*[~≈]?\s*$",
            before,
            re.IGNORECASE,
        ) and re.match(r"\s*(?:%|percent)?\s*(?:below|under|above|over)\b", blanked[end:], re.I):
            rise = fall = None
        # "rose 0.7232 to 0.9467", "fell 4 207 -> 1 888", "went from 60% down to 53%": where
        # it started, not how much.
        if re.match(
            r"\s*(?:%|percent)?\s*(?:(?:down|up)\s+)?(?:to|->|→)\s*(?:about\s+)?-?[\d.]",
            blanked[end:],
        ) or re.search(r"\bfrom\s*(?:about|around|roughly|nearly)?\s*[~≈]?\s*$", before):
            rise = fall = None
        if (rise or fall) and all(
            lf.dim is None
            and lf.sign is None
            and re.search(LEVELS, lf.labels)
            and not re.search(CHANGE_LABELS, lf.labels)
            for lf in held
        ):
            level = next(
                q
                for q in ("water cut", "uptime", "choke")
                if re.search(QUANTITIES[q][1], held[0].labels)
            )
            if re.match(r"\s*(?:%|percent)?\s*(?:below|under|above|over)\b", blanked[end:], re.I):
                return (
                    f"reads as a difference from the limit ('{written} below' it), but in "
                    f"{item.id} it is the {level} itself; write it as the level, e.g. "
                    f"'{level} of {written}, under the limit'"
                )
            return f"is written as a change, but in {item.id} it is the {level} itself"
        if rise and all(lf.sign == -1 for lf in held):
            return f"is written as a rise, but {item.id} has it as a fall ({held[0].value:g})"
        if fall and all(lf.sign == 1 for lf in held):
            return f"is written as a fall, but {item.id} has it as a rise ({held[0].value:g})"
        # "fell 72% around 2010-07-18": a fall at a date is the change across it; the change
        # between two periods' means is a shift ("shifted down", "between the periods").
        at_date = (
            re.search(
                rf"\b(?:{FALL_WORDS}|{RISE_WORDS})\b[^.;]{{0,25}}$",
                blanked[max(0, start - 40) : start],
                re.I,
            )
            and re.match(
                r"[^.;,]{0,30}?\b(?:around|on|at|in)\s+(?:\d{4}-\d{2}|(?:early|mid|late)[- ]\d{4})",
                text[end:],
                re.I,
            )
            and not re.search(
                r"between|period|shift|gradual", _part(blanked, start, commas=False), re.I
            )
        )
        # A number written as a duration ("flowing for about 1,830 days") is no change.
        if (
            (_steps(_own_stretch(blanked, start, end)) or at_date)
            and not DURATION_AFTER.match(blanked[end:])
            and _gradual(held, leaves)
        ):
            across = _across_text(leaves, {lf.record for lf in held})
            measured = f" {item.id} measures {across} across the change itself;" if across else ""
            return (
                f"is the change between the two periods' means, not a step at one date:"
                f"{measured} call it a gradual shift"
            )
        # "to 1 888 Sm3/d around 2010-05-02", "water cut to 81% around 2010-10-24": a
        # period's mean written as the level on one day (the rate that day was another).
        if (side := _period_mean(meant)) and (day := _level_at_day(text, start, end)):
            return (
                f"is the mean of the period {side} {day} in {item.id}, not the value on that "
                f"day; write it as an average ('averaged {written} {side} {day}')"
            )
    return None


# What may stand between a number and the name that is its own: "61% water cut", "123 258
# Sm3 of water".
OWN_NAME = re.compile(
    rf"\s*(?:%|percent)?(?:{UNIT_AFTER.pattern})?\s*(?:of\s+(?:the\s+|its\s+)?)?",
    re.IGNORECASE,
)

# Words that make a number a period's average, not a level on one day.
AVERAGED = re.compile(
    r"\b(?:averag\w*|means?|period|thereafter|typical\w*|from then|since then|"
    r"over the (?:following|next|months?|weeks?))\b",
    re.IGNORECASE,
)


def _period_mean(meant: list[_Leaf]) -> str | None:
    """'after' or 'before' when every value the number reads as is a regime change's period
    mean (its before_mean or after_mean, or the same number in the event's summary line)."""
    sides = set()
    for lf in meant:
        labels = lf.labels.lower()
        m = re.search(r"(?:^|\|)(before|after)_mean(?:\||$)", labels) or (
            labels.startswith("summary")
            and "regime_change" in labels
            and re.search(r"\|(before|after)(?:\||$)", labels)
        )
        if not m:
            return None
        sides.add(m.group(1))
    return sides.pop() if len(sides) == 1 else None


def _level_at_day(text: str, start: int, end: int) -> str | None:
    """The day a number is written as the level on ("to 1 888 around 2010-05-02", "around
    2010-05-02 to 1 888", "81% on 2010-10-24"), unless its clause calls it an average."""
    day = r"(\d{4}-\d{2}-\d{2})"
    # The whole clause: "weekly mean oil rate fell from 2,082 to 1,142" has a comma in a
    # number between the word and the mean.
    if AVERAGED.search(_part(text, start, commas=False)):
        return None
    if m := re.search(
        rf"\b(?:around|on|at)\s+{day}\s+(?:to|at)\s+(?:about\s+|~\s*)?$", text[:start]
    ):
        return m.group(1)
    after = re.match(
        rf"\s*(?:%|percent|[kM]?S?m3(?:\s*/\s*d(?:ay)?)?|bar)?\s*(?:\w+\s+){{0,3}}?"
        rf"(?:around|on|at)\s+{day}",
        text[end:],
    )
    reached = re.search(
        r"\b(?:to|at|reached|was|of)\s+(?:about\s+|~\s*)?$", text[max(0, start - 20) : start]
    )
    return after.group(1) if after and reached else None


def _own_stretch(text: str, start: int, end: int) -> str:
    """The stretch of a number's clause its step words are read in: from the number before
    it (or the clause's start) to the clause's end, so "water cut jumped to 0.51 ..., oil
    rate fell 72%" gives the jump to the 0.51, not to the 72%, while "fell 55% ... and
    liquid collapsing" still reads as a step."""
    bounds = [0, *(m.end() for m in CLAUSE.finditer(text)), len(text)]
    head = max(b for b in bounds if b <= start)
    tail = min((b for b in bounds if b > start), default=len(text))
    before = [m.end() for m in NUMBER.finditer(text, head, start)]
    return text[(before[-1] if before else head) : tail]


def _steps(part: str) -> list[re.Match]:
    """The step words a part of a sentence asserts: not one it denies ("a gradual decline
    rather than an abrupt choke cut", "not a sharp step")."""
    return [
        m
        for m in STEP_WORDS.finditer(part)
        if not re.search(
            r"\b(?:rather than|instead of|not|no|nor|never|without|than)\s+(?:an?\s+|the\s+)?"
            r"(?:\w+\s+)?$",
            part[: m.start()],
            re.I,
        )
    ]


# What the words around a number say it is, and how a result labels each: the side of a
# change, the statistic, the end of a range. _best_fit reads a number's description against
# every value of its result.
SIDE_LEFT = {
    "before": r"\b(?:down |up )?from\s+(?:about\s+|around\s+|roughly\s+|nearly\s+|~\s*)?$",
    "after": r"(?<!close )(?<!up )(?<!due )(?<!compared )(?<!equal )(?<!similar )"
    r"\bto\s+(?:about\s+|around\s+|roughly\s+|nearly\s+|~\s*)?$",
}
SIDE_RIGHT = {
    "before": r"^\s*(?:[%\w/]+\s+){0,2}?(?:before|earlier)\b",
    "after": r"^\s*(?:[%\w/]+\s+){0,2}?after\b(?!\s+(?:first|start))",
}
SIDE_LABEL = {"before": r"(?:^|[|_])before(?:[|_]|$)", "after": r"(?:^|[|_])after(?:[|_]|$)"}
STAT_LABEL = {
    "p10": r"(?:^|[|_])p10(?:[|_]|$)",
    "p50": r"(?:^|[|_])p50(?:[|_]|$)",
    "p90": r"(?:^|[|_])p90(?:[|_]|$)",
}
GRAIN_TEXT = {
    "year": r"\b(?:yearly|annual|per year|by year|average for the year|for the year)\b",
    "month": r"\b(?:monthly|per month|by month)\b",
}
GRAIN_LABEL = {"year": r"by_year|yearly|annual", "month": r"monthly|by_month"}
# A limit or what is left before one: "the 80% and 90% water cut limits were passed",
# "189,000 Sm3 left before a 95% water cut".
REF_TEXT = r"\b(?:limits?|reference|threshold|thresholds|left|remain\w*)\b"
REF_LABEL = r"to_limits|limit|threshold|remaining|design"
# Units that make a number something other than a quantity's level: a depth, a duration.
NOT_A_LEVEL = re.compile(
    r"\s*(?:m\b|metres?|meters?|m\s?MD|days?|weeks?|months?|years?|bar\b|psi\b|°)", re.I
)
STRONG = ("side", "stat", "dates", "grain", "ref", "unit", "work")
# Values a result names as no physical amount: shares, ratios, fit statistics, levels. A
# number written in Sm3/d, metres or per year is none of these.
SHARE_KEY = re.compile(
    r"^(?:\w+_)?(?:r2|b|aic|uptime|water_?cut|wc|shares?|ratio|pct|choke|attention|correlation|"
    r"pearson|spearman|wor|fraction|relative_change|change_across|confidence)(?:_\w+)?$"
)
# Units that cannot stand for each other; within a group they may (a day's volume is a rate).
UNIT_GROUPS = (
    {"rate", "volume", "rate_other"},
    {"metres", "ft"},
    {"bar", "psi"},
    {"days", "weeks", "months", "years"},
    {"per_year", "per_month"},
    {"degc", "degf"},
)


def _unit_kind(leaf: _Leaf) -> str | None:
    """What a value is measured in, for the fit: its unit, "share" for a value its result
    names as a share, ratio or statistic (or a text wrote as a percentage), else None."""
    # The value's own name first: a choke in a table keyed "months" is no number of months.
    if leaf.unit == "%" or any(SHARE_KEY.match(p) for p in leaf.labels.lower().split("|")[:2]):
        return "share"
    return leaf.dim


def _description(text: str, blanked: str, start: int, end: int) -> dict[str, tuple[str, int]]:
    """What the text says a number at [start, end) is, each with a weight: its side of a
    change ("from 4 050", "to 1 146", "185 before"), its statistic (P10, P50, P90, a range's
    low or high end), its quantity (named next to it, or between it and the number before
    it), the dates right beside it with no other number between, and the grain its clause
    names (yearly, monthly)."""
    said: dict[str, tuple[str, int]] = {}
    begin = max([0, *(m.end() for m in CLAUSE.finditer(blanked, 0, start))])

    def numbers(a: int, b: int | None = None) -> list[re.Match]:
        # A year is no number here: "689 067 Sm3 in 2009" keeps its date.
        found = NUMBER.finditer(blanked, a) if b is None else NUMBER.finditer(blanked, a, b)
        return [m for m in found if not re.fullmatch(r"(?:19|20)\d{2}", m.group(0).strip())]

    previous = [m.end() for m in numbers(begin, start)]
    own_start = previous[-1] if previous else begin
    following = [m.start() for m in numbers(end)]
    clause_end = min([len(blanked), *(m.start() for m in CLAUSE.finditer(blanked, end))])
    own_end = min([clause_end, *following[:1]])
    left = blanked[max(own_start, start - 30) : start]
    right = blanked[end:own_end]
    # A list ("80%, 90% and 95% water cut limits"): its items share the words after it.
    last_start, last_end = start, end
    for m in numbers(end):
        if m.start() >= clause_end or not LIST_JOIN.fullmatch(blanked[last_end : m.start()]):
            break
        last_start, last_end = m.start(), m.end()
    listed = last_end > end
    if listed:
        after = [m.start() for m in numbers(last_end)]
        list_right = blanked[last_end : min([clause_end, *after[:1]])]
    for side, pattern in SIDE_LEFT.items():
        if re.search(pattern, left, re.I):
            said["side"] = (side, 2)
    for side, pattern in SIDE_RIGHT.items():
        if "side" not in said and re.search(pattern, right[:30], re.I):
            said["side"] = (side, 2)
    # Statistics are read in the text as written (the blanked text hides "P10").
    head = text[max(0, start - 80) : start]
    if m := (
        re.search(
            r"\b(P10|P50|P90)\s*(?:of|at|is|was|:)?\s*(?:about\s+|around\s+|~\s*)?$", head, re.I
        )
        or re.match(r"\s*(?:Sm3(?:/d)?\s*)?\(?(P10|P50|P90)\b", text[end : end + 20], re.I)
    ):
        said["stat"] = (m.group(1).lower(), 2)
    elif re.search(
        r"\brange(?:\s+(?:of|from|between|was|is|at))?\s+(?:about\s+|around\s+|~\s*)?$", head, re.I
    ):
        said["stat"] = ("p90", 2)  # a range's first end is its low one
    elif re.search(
        r"\brange\b[^.;\d]{0,12}\d[\d\s,.]*\s*(?:Sm3\s*)?(?:to|and|-)\s+(?:about\s+|~\s*)?$",
        head,
        re.I,
    ):
        said["stat"] = ("p10", 2)
    elif re.search(
        r"\b(?:forecasts?|forecast to (?:make|produce)|(?:will|would) (?:make|produce)|"
        r"expected to (?:make|produce)|central estimate)\b[^.;,()\d]{0,30}$",
        head,
        re.I,
    ):
        said["stat"] = ("p50", 2)
    if not NOT_A_LEVEL.match(blanked, end):
        if name := _quantity(blanked, start, end) or (
            listed and _quantity(blanked, last_start, last_end)
        ):
            said["quantity"] = (name, 2)
        elif (
            names := [
                m
                for m in _NAMES.finditer(blanked, own_start, start)
                # "at 61% water cut and averaged 34% less oil": a name right after the number
                # before is that number's, not this one's.
                if not (previous and OWN_NAME.fullmatch(blanked[own_start : m.start()]))
                # "water cut and choke moved together with a correlation of 69%": the number is
                # the correlation's.
                and not re.search(
                    r"\b(?:correlation|coefficient|Pearson|Spearman)\b",
                    blanked[m.end() : start],
                    re.I,
                )
            ]
        ):
            said["quantity"] = (names[-1].group(0), 1)
    # Its dates: one written right after it with "in", "on", "at" ("31% in 2015-01"), or
    # one before it in its own phrase ("since 2010-10-12 for 58 days", "on 2010-10-12 oil
    # was 1 759", "around 2010-05-02 to 1 888"), with no comma, bracket, "and" or "vs" in
    # between. "In November 2014 to 30.33%": a period before "to" is the number before's.
    # (A range of dates, "in 2014-12-2015-01", is no one date.)
    when = (
        r"(?<!\d[-–])(?:(?:early|mid|late)[\s-]+)?"
        r"(\d{4}-\d{2}(?!\d)(?:-\d{2}(?!\d))?|(?:19|20)\d{2}(?!-\d))(?!\s*(?:-|–|to)\s*\d{4})"
    )
    days = []
    if m := re.match(
        rf"\s*(?:(?!(?:and|or|but|vs|versus|while)\b)[%\w/]+\s*){{0,2}}?\s*"
        rf"(?:in|on|at|during|by|for|around|since)\s+{when}",
        text[end:own_end],
        re.I,
    ):
        days.append(m.group(1))
    if m := re.search(
        rf"(?:\b(?P<prep>on|in|at|since|around|from|during|by|of)\s+)?{when}"
        # "stepped down 55% on 2010-05-02 (weekly mean 4 207": a bracket right after the date
        # spells out that date's event.
        r"(?P<tail>(?:\s*\()?[^,;()]{0,30})$",
        text[own_start:start],
        re.I,
    ):
        tail = m.group("tail")
        # The end of a range of dates ("a workover from 2014-11-22 to 2014-11-27 that set a
        # plug at 3150 m", "from 2011 to 2013 with a correlation of 69%") is no one date.
        range_end = re.search(
            r"\d{4}(?:-\d{2}){0,2}\s*(?:-|–|to|through|until)\s*$",
            text[own_start : own_start + m.start(2)],
            re.I,
        )
        # Only an event's own day opens a bracket on it ("on 2010-05-02 (weekly mean"), not
        # a period's end ("through 2012-12 (water cut 0.64 to 0.76").
        bracketed = tail.lstrip().startswith("(") and (m.group("prep") or "").lower() not in (
            "on",
            "around",
            "at",
            "since",
        )
        # A period before "to X" is the start's ("from 97% in late 2014 to about 31%", "the
        # late-2014 drop to about 0.31"), but an event's own date is its ("fell 55% around
        # 2010-05-02 to 1 888").
        period_to = (m.group("prep") or "").lower() not in ("around", "on", "at", "since") and (
            re.match(r"\s*(?:(?:drop|fall|decline|rise|jump|cut|shift)\s+)?to\b", tail)
        )
        # "A choke cut on 2018-11-07 from 70% to 40%": the date is the change's; the value
        # it started from stands in the row before it.
        from_value = said.get("side", ("",))[0] == "before"
        if (
            not re.search(r"\b(?:and|vs|versus|while|whereas|but)\b", tail)
            and not period_to
            and not from_value
            and not bracketed
            and not range_end
        ):
            days.append(m.group(2))
    # "Cut to 40% in 2018-11": a level reached during a month (a year) is the next month's
    # (year's) row too.
    if said.get("side", ("",))[0] == "after":
        for day in list(days):
            if re.fullmatch(r"\d{4}-\d{2}", day):
                y, mo = int(day[:4]), int(day[5:])
                days.append(f"{y + mo // 12}-{mo % 12 + 1:02d}")
            elif re.fullmatch(r"\d{4}", day):
                days.append(str(int(day) + 1))
    # "November 2014 oil 185.7 Sm3/d at 0.96 water cut vs January 2015 ...": a date that
    # opens the phrase dates every number in it, up to a comma, "vs", "and" or "to".
    if not days:
        breaks = list(
            re.finditer(
                r"[,;:()]|\b(?:vs|versus|and|while|but|whereas|to|from|then|after|before)\b|->|→",
                text[begin:start],
                re.I,
            )
        )
        phrase = text[begin + (breaks[-1].end() if breaks else 0) : start]
        # ("a workover from 2014-11-22 to 2014-11-27 that set a plug at 3150 m": the end of
        # a range of dates opens no phrase.)
        range_end = bool(
            breaks
            and breaks[-1].group(0).lower() == "to"
            and re.search(r"\d{4}(?:-\d{2}){0,2}\s*$", text[begin : begin + breaks[-1].start()])
        )
        if not range_end and (
            m := re.match(rf"\s*(?:(?:in|on|by|during|at)\s+)?{when}\b", phrase, re.I)
        ):
            days.append(m.group(1))
    # "to 35.12% at 70.96 bar WHP in January 2015, with ...": a phrase with one date, after
    # its numbers, dates them all.
    if not days:
        stop = re.search(
            r"[,;:()]|\b(?:vs|versus|and|while|but|whereas|to|from|then|after|before|since)\b"
            r"|->|→",
            text[end:clause_end],
            re.I,
        )
        # (Read to the clause's end, so a range of dates "2009-06 to 2012-06" stays one.)
        found = [
            m.group(1)
            for m in re.finditer(rf"\b(?:in|on|during|at)\s+{when}", text[end:clause_end], re.I)
            if not stop or m.start() < stop.start()
        ]
        if len(found) == 1:
            days.append(found[0])
    if days:
        said["dates"] = ("|".join(days), 2)
    if work := _work(blanked[own_start:start]):
        said["work"] = ("|".join(work), 2)
    # The unit written right after it: a rate, a volume, metres, per year (a percent says
    # nothing: a change in a rate is a percentage too).
    unit = _written_unit(blanked, end)
    # A range shares the unit written after its end: "from 3126.2 to 3130.7 mMD".
    if (
        unit is None
        and not re.search(r"%|percent|pct", blanked[start:end] + blanked[end : end + 9])
        and (
            m := re.match(
                r"\s*(?:to|-|–)\s*(?:about\s+|~\s*)?\d(?:[\d.,\u00a0\u202f]|\s(?=\d{3}\b))*",
                blanked[end:],
            )
        )
    ):
        unit = _written_unit(blanked, end + m.end())
    if unit and unit != "barrels":
        said["unit"] = (unit, 2)
    # The grain its own phrase names: "yearly average choke went from 77.1%", not "monthly
    # water cut" earlier in the sentence.
    own = blanked[own_start:own_end]
    for grain, pattern in GRAIN_TEXT.items():
        if re.search(pattern, own, re.I):
            said["grain"] = (grain, 2)
    # A month's extreme is a month's value whatever grain the sentence names ("peaked at
    # 5 817 Sm3/d in 2010").
    extreme = re.search(EXTREME, own, re.I)
    if extreme:
        said["extreme"] = ("", 0)
        said.pop("grain", None)
    elif "grain" not in said:
        # "yearly water cut was 57% in 2010 ..., while liquid held at 5 130 Sm3/d in 2010":
        # the grain the clause named earlier, weakly.
        for grain, pattern in GRAIN_TEXT.items():
            if re.search(pattern, blanked[begin:start], re.I):
                said["grain"] = (grain, 1)
    # "under the 80% limit and a field liquid rate of 10 700": a limit word with "and" (or
    # "with", "while", "but") between it and the number is the number before's.
    refs = list(re.finditer(REF_TEXT, blanked[max(own_start, start - 40) : start], re.I))
    if (
        (
            refs
            and not re.search(
                r"\b(?:and|with|while|but|whereas)\b",
                blanked[max(own_start, start - 40) + refs[-1].end() : start],
                re.I,
            )
        )
        or re.match(rf"[^,;]{{0,30}}?{REF_TEXT}", right, re.I)
        or (listed and re.match(rf"[^,;]{{0,30}}?{REF_TEXT}", list_right, re.I))
    ):
        said["ref"] = ("limit", 2)
    return said


def _fit_scores(said: dict[str, tuple[str, int]], leaf: _Leaf) -> dict[str, int]:
    """How well a value of a result fits each part of a number's description: + where its
    labels agree, - where they contradict, 0 where they say nothing."""
    out: dict[str, int] = {}
    labels = leaf.labels.lower()

    def by(table: dict[str, str], key: str) -> None:
        value, w = said[key]
        mine = [k for k, pattern in table.items() if re.search(pattern, labels)]
        out[key] = w if value in mine else -w if mine else 0

    if "side" in said:
        by(SIDE_LABEL, "side")
    if "stat" in said:
        by(STAT_LABEL, "stat")
    if "grain" in said:
        by(GRAIN_LABEL, "grain")
    if "ref" in said:
        # A limit or reference point; a month's or year's measurement is none.
        w = said["ref"][1]
        out["ref"] = w if re.search(REF_LABEL, labels) else -w if leaf.row_day else 0
    if "quantity" in said:
        name, w = said["quantity"]
        key = next((k for k, (t, _) in QUANTITIES.items() if re.fullmatch(t, name, re.I)), None)
        if key is not None:
            mine = [k for k, (_, lab) in QUANTITIES.items() if re.search(lab, labels)]
            out["quantity"] = w if key in mine else -w if mine else 0
    if "work" in said:
        # A depth the text calls a plug's, read against what the tool's text calls it.
        words, w = said["work"]
        mine = re.findall(r"work_(\w+)", labels)
        out["work"] = w if set(words.split("|")) & set(mine) else -w if mine else 0
    if "unit" in said:
        unit, w = said["unit"]
        kind = _unit_kind(leaf)
        group = next((g for g in UNIT_GROUPS if unit in g), {unit})
        out["unit"] = (
            w if kind == unit else -w if kind == "share" or (kind and kind not in group) else 0
        )
    if "dates" in said:
        written, w = said["dates"]
        dates = [d for d in written.split("|") if d]
        own = [*leaf.record_days, *([leaf.row_day] if leaf.row_day else [])]
        if own:
            hit = any(o.startswith(d) or d.startswith(o) for o in own for d in dates)
            # "in 2010" fits the 2010 row of a yearly table better than one of its months
            # (unless the text names an extreme: a month's peak in 2010).
            month_of_year = (
                hit
                and "extreme" not in said
                and leaf.row_day is not None
                and not leaf.record_days
                and all(len(d) == 4 for d in dates)
                and len(leaf.row_day) > 4
            )
            out["dates"] = w - 1 if month_of_year else w if hit else -w
    return out


def _best_fit(
    text: str, blanked: str, written: str, fits: list[Reading], leaves: list[_Leaf]
) -> str | None:
    """A number whose description fits another value of its result clearly better than any
    value it could be: "fell 55% around 2010-05-02 to 3 196" where 3 196 is the rate before
    that change, "range 324,000 to 339,803" where 339,803 is the low end, "since 2010-10-12
    for 58 days" where 58 is another event's. The value the words describe is looked up in
    the result first, then compared with the number written. Only a precise part of the
    description (a side, a statistic, a date) decides; quantity and grain only add."""
    spans = _standalone(blanked, written)
    if not spans:
        return None
    start, end = spans[0]
    said = _description(text, blanked, start, end)
    if not any(k in said for k in STRONG):
        return None
    held = [
        lf for lf in leaves if lf.value is not None and any(_reads_as(r, [lf.value]) for r in fits)
    ]
    if not held:
        return None
    # Rivals of the same kind as the value written: fractions against fractions (a water
    # cut of 0.69), levels against levels (a rate, a choke of 59%), not years or small counts.
    fraction = all(abs(lf.value) <= 1.5 for lf in held)
    rivals = [
        lf
        for lf in leaves
        if lf.value is not None
        and lf not in held
        and (
            abs(lf.value) <= 1.5
            if fraction
            else abs(lf.value) > 10 and not (1900 <= lf.value <= 2100 and lf.value == int(lf.value))
        )
    ]
    if not rivals:
        return None
    mine = max((_fit_scores(said, lf) for lf in held), key=lambda f: sum(f.values()))
    # The rival to name: one of the same family as the value written when one fits as well
    # (the 2013 row of the column it was read from, not a water cut that year).
    families = {_family(lf) for lf in held}
    best = max(
        rivals,
        key=lambda lf: (sum(_fit_scores(said, lf).values()), _family(lf) in families),
    )
    theirs = _fit_scores(said, best)
    # A unit or a well-work word decides only against a value it rules out (a share written
    # in Sm3/d, a weight written as a plug's depth), not for a rival that merely names it
    # where the value written has nothing said about it.
    weak = [k for k in ("unit", "work") if mine.get(k, 0) >= 0]
    theirs = {k: v for k, v in theirs.items() if k not in weak}
    mine = {k: v for k, v in mine.items() if k not in weak}
    strong = sum(theirs.get(k, 0) - mine.get(k, 0) for k in STRONG)
    if sum(theirs.values()) >= sum(mine.values()) + 3 and strong >= 2:
        what = ", ".join(f"{k} {v}" for k, (v, _) in said.items() if theirs.get(k, 0) > 0)
        return (
            f"is written as {what or 'that value'}, which fits {best.value:g} "
            f"({best.labels.replace('|', ' ')}) in the result better than {written}"
        )
    return None


def _part(text: str, start: int, commas: bool = True) -> str:
    """The part of a sentence that holds `start`: between commas or clause ends (or the
    whole clause, commas=False)."""
    bounds = [0, *(m.end() for m in CLAUSE.finditer(text)), len(text)]
    begin = max(b for b in bounds if b <= start)
    end = min((b for b in bounds if b > start), default=len(text))
    left = text.rfind(",", begin, start) if commas else -1
    right = text.find(",", start, end) if commas else -1
    return text[left + 1 if left >= 0 else begin : right if right >= 0 else end]


def _gradual(held: list[_Leaf], leaves: list[_Leaf]) -> bool:
    """Are these numbers the change between two periods of a shift the detector found
    gradual: a screen issue of kind shift_down, or an event whose change across the date
    itself is under 25%?"""
    if all("shift" in lf.labels for lf in held):
        return True
    records = {lf.record for lf in held if lf.record is not None}
    if not records or any(_is_across(lf) for lf in held):
        return False
    across = [lf.value for lf in leaves if lf.record in records and _is_across(lf)]
    return bool(across) and all(abs(v) < (25 if abs(v) > 1 else 0.25) for v in across)


def _across_text(leaves: list[_Leaf], records: set) -> str:
    """The change across a shift's date, as percents (points for a share such as the water
    cut, which changes by points), for the records given."""
    shown = sorted(
        {
            f"{abs(lf.value) * 100:.0f} points"
            if re.search(LEVELS, lf.labels, re.I) and abs(lf.value) <= 1
            else f"{abs(lf.value) * (100 if abs(lf.value) <= 1 else 1):.0f}%"
            for lf in leaves
            if lf.record in records and _is_across(lf)
        }
    )
    return ", ".join(shown)


def _is_across(leaf: _Leaf) -> bool:
    """The change across a shift's date: an event's change_across, or a percent its text
    writes next to "across" ("2,352 -> 1,838 (-22%)", "21% across it")."""
    if leaf.value is None:
        return False
    return "change_across" in leaf.labels or ("ctx_across" in leaf.labels and leaf.unit == "%")


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
        # A month names the records dated in it: "fell 55% in June 2010" against a step
        # dated 2010-05-02 and a water-cut crossing dated 2010-06-04.
        dated = {
            lf.record
            for lf in leaves
            if lf.record is not None and any(d.startswith(when) for d in lf.record_days)
        }
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
        field = _names_field(text, match.start())
        held = [
            lf
            for item in cited
            for lf in leaves_of(item)
            if lf.day is not None and (lf.day == day or (len(day) == 7 and lf.day.startswith(day)))
        ]
        wells = {lf.well for lf in held}
        if named and held and None not in wells and not {_well_key(w) for w in wells} & named:
            wrong.append(f"{day} (evidence has it only for {', '.join(sorted(wells))})")
        elif field and held and None not in wells:
            wrong.append(
                f"{day} (evidence has it only for {', '.join(sorted(wells))}, not the field)"
            )
    return list(dict.fromkeys(wrong))


# The whole field as a sentence's subject: "the field", "field totals", "the field's". Not
# "field-wide", which describes a well's shut-in.
FIELD_OWNER = re.compile(
    r"\b(?:the|whole|entire)\s+field\b(?![-\s]wide)|\bfield(?:'s|\s+totals?)\b", re.IGNORECASE
)


def _names_field(text: str, start: int) -> bool:
    """Is the part of the sentence holding `start` about the field, with no well named in
    it ("The field ... first went above 0.80 on <date>")?"""
    bounds = [0, *(m.end() for m in CLAUSE.finditer(text)), len(text)]
    begin = max(b for b in bounds if b <= start)
    end = min((b for b in bounds if b > start), default=len(text))
    part = text[begin:end]
    return bool(FIELD_OWNER.search(part)) and not WELL.search(part)


# Words that name the event a date is given to, and how a tool result labels that event's
# records: "gauge dead since <date>", "shut in from <date>", "stepped down on <date>".
EVENT_DATES = {
    "gauge failure": (r"\bgauges?\b|\bsensors?\b", r"gauge|bhp"),
    "shut-in": (r"\bshut[\s-]?in\b|\bshut down\b|\bshutdown\b", r"shut"),
    "step in rate": (r"\bstep(?:ped)?[\s-]down\b|\bregime change\b", r"regime|step"),
    "crossing": (r"\babove\b|\bexceed\w*|\bcross\w*|\bpast\b", r"threshold|crossing"),
}
# A crossing names its threshold before the date ("above the 80% design limit from"):
# numbers may stand between.
CROSSING_REACH = re.compile(
    r"^(?!.*\b(?:last|before|until|prior)\b)(?!.*\d{4}-\d{2})(?:\s*[\w.%'’-]+){0,5}?\s*"
    r"(?P<prep>since|from|on)\s*$",
    re.IGNORECASE,
)
EVENT_REACH = re.compile(
    # A few words in between, a zero reading too ("gauge has read 0 since <date>"), in one
    # clause. A last good reading ("last read 253.9 bar on <date>") dates no failure.
    # "no gauge issue is present, but from <date> its oil fell": a new clause is not the event.
    r"^(?!.*\b(?:last|good|valid|before|until|prior|worked|working|but|while|whereas|however|"
    r"although|yet)\b)"
    r"(?:[\s,]*(?:[^\W\d][\w'’-]*|0)){0,5}?\s*(?P<prep>since|from|on|at|around|in)\s*$",
    re.IGNORECASE,
)


def _date_events(
    text: str, cited: list[Evidence], leaves_of, events: tuple[str, ...] | None = None
) -> list[str]:
    """Dates given to an event that, in the evidence, date another: "gauge dead since
    2010-08-16" when 2010-08-16 starts a shut-in and the gauge failure starts on 2010-10-12.
    Judged only where the date names records and some record of that event is cited."""
    # Record numbers count within one result: key them by the result too.
    leaves = [
        replace(lf, record=(item.id, lf.record) if lf.record is not None else None)
        for item in cited
        for lf in leaves_of(item)
    ]
    wrong = []
    for match in DATE.finditer(text):
        day = match.group(1)
        before = text[max(0, match.start() - 60) : match.start()]
        for event, (words, labels) in EVENT_DATES.items():
            if events is not None and event not in events:
                continue
            named = None
            for m in re.finditer(words, before, re.IGNORECASE):
                named = m
            # An event denied ("no gauge issue", "without a shut-in") dates nothing.
            if named and re.search(
                r"\b(?:no|not|without|nor)\s+(?:\w+\s+)?$", before[: named.start()], re.I
            ):
                continue
            reach = (
                (CROSSING_REACH if event == "crossing" else EVENT_REACH).match(
                    before[named.end() :]
                )
                if named
                else None
            )
            if reach is None:
                continue
            dated = {lf.record for lf in leaves if lf.record is not None and day in lf.record_days}
            of_event = {
                lf.record for lf in leaves if lf.record is not None and re.search(labels, lf.labels)
            }
            if dated and of_event and not dated & of_event:
                wrong.append(
                    f"{day} is given as the {event}'s date, but no {event} in its evidence has it"
                )
                continue
            # A date no record names, outside every dated record of that event: "gauge dead
            # since 2010-10-11" when the gauge failure runs from 2010-10-12.
            spans = [
                (min(lf.record_days), max(lf.record_days))
                for lf in leaves
                if lf.record in of_event and lf.record_days
            ]
            if not dated and spans and not any(a[: len(day)] <= day <= b for a, b in spans):
                first, last = min(spans)
                wrong.append(
                    f"{day} is given as the {event}'s date, but the {event} in its evidence "
                    f"runs from {first}" + (f" to {last}" if last != first else "")
                )
                continue
            # "since"/"from" give its start: a day inside it, or its end, is not when it
            # began ("gauge dead since 2010-11-01" when it read 0 from 2010-10-12).
            starts = sorted({a for a, _ in spans})
            if (
                reach["prep"].lower() in ("since", "from")
                and starts
                and not any(a[: len(day)] == day for a in starts)
            ):
                wrong.append(
                    f"{day} is given as the start of the {event}, but the {event} in its "
                    f"evidence starts on {', '.join(starts)}"
                )
    return list(dict.fromkeys(wrong))


# The events a stretch of a year can date ("gauge dead since early 2010", "fell 55% in
# spring 2009"), by the words right before it, and the records that date them. A change in
# rate needs its size written before the stretch: "fell in late 2012" may be a decline.
PERIOD_EVENTS = {
    "gauge failure": (r"\bgauges?\b|\bsensors?\b", r"gauge|bhp", False),
    "shut-in": (r"\bshut[\s-]?in\b|\bshut down\b", r"shut", False),
    "change in rate": (
        r"\b(?:fell|dropped|declined|rose|jumped|step(?:ped)?(?:\s+down)?|shift(?:ed)?)\b",
        r"regime|step",
        True,
    ),
}
PERIOD_REACH = re.compile(
    r"(?!.*\b(?:last|before|until|prior|after)\b)(?:\s*[\w%.,'’-]+){0,5}?\s*"
    r"(?:since|from|in|on|around|during|by|at)\s*",
    re.IGNORECASE,
)


def _period_bounds(match: re.Match) -> tuple[str, str]:
    """First and last day of "Q2 2010", "early 2010", "winter 2010" (December to February)."""
    key = (match["q"] or match["part"] or match["season"]).lower()
    first, last = PERIOD_MONTHS[key]
    year = int(match.group(match.lastindex))
    start = date(year, first, 1)
    end_year = year + 1 if last < first else year
    end = date(end_year, last % 12 + 1, 1) if last < 12 else date(end_year, 12, 31)
    end = end if last == 12 else date.fromordinal(end.toordinal() - 1)
    return start.isoformat(), end.isoformat()


def _periods(
    text: str, cited: list[Evidence], leaves_of, events: tuple[str, ...] | None = None
) -> list[str]:
    """A stretch of a year given to an event that the evidence dates outside it: "gauge dead
    since early 2010" when it read 0 from 2010-10-12; "fell 55% in spring 2009" when the
    change point is 2010-05-02. Judged only where some record of that event is cited."""
    leaves = [
        replace(lf, record=(item.id, lf.record) if lf.record is not None else None)
        for item in cited
        for lf in leaves_of(item)
    ]
    wrong = []
    for match in PERIOD.finditer(text):
        clause_start = max(text.rfind(c, 0, match.start()) for c in ",;:.") + 1
        before = text[clause_start : match.start()]
        lo, hi = _period_bounds(match)
        for event, (words, labels, sized) in PERIOD_EVENTS.items():
            if events is not None and event not in events:
                continue
            named = next(reversed(list(re.finditer(words, before, re.IGNORECASE))), None)
            gap = before[named.end() :] if named else ""
            if not named or not PERIOD_REACH.fullmatch(gap):
                continue
            if sized and not re.search(r"\d\s*%", gap):
                continue
            starts = {
                min(lf.record_days)
                for lf in leaves
                if lf.record is not None and lf.record_days and re.search(labels, lf.labels)
            }
            if starts and not any(lo <= day <= hi for day in starts):
                wrong.append(
                    f"'{match.group(0)}' ({lo} to {hi}) is given as the {event}'s time, but the "
                    f"{event} in its evidence is on {', '.join(sorted(starts)[:3])}"
                )
                break
    return list(dict.fromkeys(wrong))


# The date first, the dead gauge after it: "From 2010-10-11 BHP reads zero".
GAUGE_AFTER_DATE = re.compile(
    r"\b(?P<prep>from|since|on)\s+(?P<day>\d{4}-\d{2}(?:-\d{2})?),?\s+(?:the\s+)?"
    r"(?:downhole\s+)?(?:BHP|pressure|gauge)\b[^.;]{0,25}?\b(?:reads?|read|was|is|went|"
    r"dropped|fell)\s+(?:to\s+)?(?:zero|0)\b",
    re.IGNORECASE,
)


def _gauge_dated_first(text: str, cited: list[Evidence], leaves_of) -> list[str]:
    starts = sorted(
        {
            min(lf.record_days)
            for item in cited
            for lf in leaves_of(item)
            if lf.record_days
            and re.search(r"gauge|bhp", lf.labels)
            and "gauge_failure" in lf.labels
        }
    )
    wrong = []
    for m in GAUGE_AFTER_DATE.finditer(text):
        if (
            starts
            and m["prep"].lower() in ("from", "since")
            and not any(s[: len(m["day"])] == m["day"] for s in starts)
        ):
            wrong.append(
                f"{m['day']} is given as the start of the gauge failure, but the gauge failure "
                f"in its evidence starts on {', '.join(starts)}"
            )
    return wrong


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
        if not re.search(r"\bfrom\s*$", blanked[:a_start]):
            continue
        if not FROM_TO.fullmatch(blanked[a_end:b_start]):
            continue
        # The two may be declared from different results ("from 3%" in an event, "to 57%"
        # in a yearly table): a change either result records settles which way it ran.
        items = [i for i in dict.fromkeys((a.evidence_id, b.evidence_id)) if log.get(i)]
        if not items:
            continue
        item = log.get(items[0])
        sides: dict[tuple[str, int], dict[str, list[float]]] = {}
        for source in items:
            for lf in leaves_of(log.get(source)):
                side = (
                    "before" if "before" in lf.labels else "after" if "after" in lf.labels else None
                )
                if lf.record is not None and side and lf.value is not None:
                    sides.setdefault((source, lf.record), {}).setdefault(side, []).append(lf.value)

        def holds(record: dict, side: str, text: str) -> bool:
            return any(_reads_as(r, record.get(side, [])) for r in _readings(text))

        backwards = [
            k for k, r in sides.items() if holds(r, "before", b.text) and holds(r, "after", a.text)
        ]
        forwards = any(
            holds(r, "before", a.text) and holds(r, "after", b.text) for r in sides.values()
        )
        if backwards and not forwards:
            item = log.get(backwards[0][0])
            problems.append(
                f"{label}: 'from {a.text} to {b.text}' runs the wrong way; in {item.id} the "
                f"change is from {b.text} to {a.text}."
            )
    return problems


def _spans_of(blanked: str, declared, unused) -> list:
    """Where each declaration is written, in order; of two at one place ("31%" and "31"),
    the longer."""
    spans = sorted(
        (start, -end, id(d), d)
        for d in declared
        if d not in unused
        for start, end in _standalone(blanked, d.text)[:1]
    )
    out = []
    for start, end, key, d in spans:
        if not out or out[-1][0] != start:
            out.append((start, -end, key, d))
    return out


def _fits_of(d) -> list[Reading]:
    return [f for r in _readings(d.text) if (f := _pinned(r, d.value))]


PAIR_FROM = re.compile(r"\bfrom\s*(?:about|around|roughly|nearly)?\s*[~≈]?\s*$", re.IGNORECASE)
PAIR_TO = re.compile(r"[^,;.]{0,48}?(?:\bto|->|→)\s*(?:about|around|roughly|nearly)?\s*[~≈]?\s*$")
PAIR_TO_FROM = re.compile(
    r"\s*(?:%|[^\W\d]+\d?(?:/[^\W\d]+)?)?\s*,?\s*(?:up|down)?\s*from\s*(?:about|around)?\s*[~≈]?\s*$"
)
RISE = re.compile(rf"\b(?:{RISE_WORDS}|risen|went up|gone up)\b", re.IGNORECASE)
FALL = re.compile(rf"\b(?:{FALL_WORDS}|fallen|went down|gone down)\b", re.IGNORECASE)


def _pairs(blanked: str, declared, unused) -> list[tuple]:
    """(first, last, start, end) for "from A to B" and "to B from A" between two declared
    numbers: the change as written, first value first."""
    spans = _spans_of(blanked, declared, unused)
    pairs = []
    for (a_start, a_end, _, a), (b_start, b_end, _, b) in zip(spans, spans[1:], strict=False):
        between = blanked[a_end:b_start]
        if re.search(r"(?<![A-Za-z])\d", between):  # another number between (not "Sm3")
            continue
        if PAIR_FROM.search(blanked[max(0, a_start - 20) : a_start]) and PAIR_TO.fullmatch(between):
            pairs.append((a, b, a_start, b_end))
        elif re.search(r"\bto\s*[~≈]?\s*$", blanked[max(0, a_start - 8) : a_start]) and (
            PAIR_TO_FROM.fullmatch(between)
        ):
            pairs.append((b, a, a_start, b_end))
    return pairs


def _time_order(label, blanked, declared, unused, items) -> list[str]:
    """ "from A to B" where a cited table holds both in one column and every row with A
    comes after every row with B: the change ran from B to A ("water cut rose from 0.31 to
    0.97" across a water shut-off that took it from 0.97 to 0.31)."""
    problems = []
    for first, last, _, _ in _pairs(blanked, declared, unused):
        if first.text.rstrip().endswith("%") != last.text.rstrip().endswith("%"):
            continue
        # The rows that hold each declared value (to 0.2%): a rounded "30%" also matches
        # an unrelated month at 0.30, so the text's rounding is not used.
        f_fits = [(abs(first.value), 0.002 * abs(first.value) + 1e-9, (1.0,))]
        l_fits = [(abs(last.value), 0.002 * abs(last.value) + 1e-9, (1.0,))]
        for item in items:
            for table in _tables(item.result):
                columns = [str(c).lower() for c in table["columns"]]
                key = next((i for i, c in enumerate(columns) if c in DATE_COLUMNS), None)
                if key is None:
                    continue
                for col in range(len(columns)):
                    if col == key:
                        continue

                    def when(
                        fits: list[Reading], col: int = col, table: dict = table, key: int = key
                    ) -> list[str]:
                        return [
                            str(row[key])[:10]
                            for row in table["rows"]
                            if isinstance(row[col], int | float)
                            and not isinstance(row[col], bool)
                            and any(_reads_as(r, [row[col]]) for r in fits)
                        ]

                    a, b = when(f_fits), when(l_fits)
                    if a and b and min(a) > max(b):
                        problems.append(
                            f"{label}: 'from {first.text} to {last.text}' runs backwards; in "
                            f"{item.id} the {columns[col]} was {last.text} in {max(b)} and "
                            f"{first.text} in {min(a)}."
                        )
                        break
                if problems and problems[-1].startswith(f"{label}: 'from {first.text}"):
                    break
    return list(dict.fromkeys(problems))


COMPARE = re.compile(
    r"\b(?P<word>above|higher than|greater than|more than|over|exceed\w*|in excess of|beyond|"
    r"below|lower than|less than|under|short of)\b",
    re.IGNORECASE,
)


def _inconsistent(label, blanked, declared, unused) -> list[str]:
    """What a sentence's own numbers contradict, whatever the evidence: a rise from a larger
    number to a smaller one ("fell to 862 from 186"), one number said above a larger one
    ("8,460 just above the 9,000"), a percentage that is not the change it names ("fell 22%
    ... from 4 050 to 1 146")."""
    problems = []
    pairs = _pairs(blanked, declared, unused)
    ends = {id(d) for pair in pairs for d in pair[:2]}
    for first, last, start, end in pairs:
        if first.text.rstrip().endswith("%") != last.text.rstrip().endswith("%"):
            continue  # "from 3 to 95%": declared on different scales, not compared
        a, b = abs(first.value), abs(last.value)
        if a == b:
            continue
        went_up = b > a
        # The verb that governs the change, right before it: "rose from", "fell to ... from".
        verb = re.search(
            r"\b(?P<rose>rose|risen|increased|grew|climbed|jumped|went up|gone up)|"
            r"\b(?P<fell>fell|fallen|dropped|declined|decreased|went down|gone down|shrank)",
            blanked[max(0, start - 26) : start],
        )
        rose = verb if verb and verb["rose"] else None
        fell = verb if verb and verb["fell"] else None
        if (rose and not went_up and not fell) or (fell and went_up and not rose):
            problems.append(
                f"{label}: '{(rose or fell).group(0)}' from {first.text} to {last.text} "
                f"is a {'fall' if not went_up else 'rise'}, not a "
                f"{'rise' if rose else 'fall'}."
            )
        # A percentage written as this change, next to it: "fell 22% ... from 4 050 to
        # 1 146", "from 4 050 to 1 146 (-72%)". Shares (from 3% to 57%) change in points.
        if first.text.rstrip().endswith("%"):
            continue
        for d in declared:
            if d in unused or id(d) in ends or not d.text.rstrip().endswith("%"):
                continue
            spans_d = _standalone(blanked, d.text)
            if not spans_d:
                continue
            d_start, d_end = spans_d[0]
            if d_end <= start:
                gap = blanked[d_end:start]
                lead = blanked[max(0, d_start - 20) : d_start]
                near = (
                    len(gap) <= 36
                    and not re.search(r"\d|;|\.\s", gap)
                    and bool(RISE.search(lead) or FALL.search(lead))
                )
            elif d_start >= end:
                gap = blanked[end:d_start]
                near = len(gap) <= 12 and not re.search(r"\d|;|\.\s", gap)
            else:
                near = False
            if not near:
                continue
            change = abs(b - a) / a
            shown = abs(float(re.sub(r"[^\d.]", "", d.text) or 0)) / 100
            if shown and abs(change - shown) > 0.015 + 0.005:
                problems.append(
                    f"{label}: '{d.text}' is not the change from {first.text} to "
                    f"{last.text}, which is {change:.0%}."
                )
    spans = _spans_of(blanked, declared, unused)
    for (_, a_end, _, a), (b_start, _, _, b) in zip(spans, spans[1:], strict=False):
        between = blanked[a_end:b_start]
        m = COMPARE.search(between)
        if not m or re.search(r"[;.]\s", between) or len(between) > 60:
            continue
        # "rose from 3% to over 50%": the word bounds the second number, it does not set the
        # first against it.
        if re.search(r"\b(?:to|from|at|reached|reaching)\s+$", between[: m.start()], re.I):
            continue
        # "above 50% by month 28 and above 80% by month 48": a second level, not a comparison.
        if re.search(r"\b(?:and|or|but|while|whereas|then)\b", between[: m.start()], re.I):
            continue
        # "81% of the oil entering below it, leaving 19%": the word has its own object, and
        # the next number starts another clause.
        if (
            re.match(r"\s+(?:it|them|this|that|which)\b", between[m.end() :], re.I)
            or "," in (between[m.end() :])
        ):
            continue
        # The newer words compare only right after the number, with nothing in between that
        # starts another subject ("10,800 under the 13,000 limit, but injection exceeded its
        # 16,000"), and "over" is no time span ("down 7.8% over the last 90 days").
        if re.fullmatch(
            r"under|over|exceed\w*|more than|in excess of|beyond", m["word"], re.I
        ) and (
            m.start() > 20
            or re.search(r"\b(?:but|while|whereas|and|with)\b", between[: m.start()])
            or re.match(r"over\s+(?:the\s+)?(?:last|past|next|\d)", between[m.start() :], re.I)
        ):
            continue
        above = not re.match(r"below|under|lower|less|short", m["word"], re.I)
        x, y = abs(a.value), abs(b.value)
        # Only like with like: two rates, two volumes, two percentages.
        unit_a = "%" if a.text.rstrip().endswith("%") else _written_unit(blanked, a_end)
        unit_b = (
            "%" if b.text.rstrip().endswith("%") else _written_unit(blanked, b_start + len(b.text))
        )
        if not unit_a or unit_a != unit_b:
            continue
        if (above and x < y) or (not above and x > y):
            problems.append(
                f"{label}: '{a.text}' is written {m['word']} '{b.text}', but it is "
                f"{'smaller' if above else 'larger'}."
            )
    return list(dict.fromkeys(problems))


NUMBER_WORD_VALUES = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twice": 2, "double": 2, "doubled": 2, "triple": 3, "tripled": 3,
}  # fmt: skip
TIMES = re.compile(
    r"\b(?P<q>more than|over|at least|nearly|almost|about|roughly|around|some|less than|"
    r"under)?\s*(?P<n>\d+(?:\.\d+)?|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:times|x|×|-fold|fold)(?![\w-])"
    r"|\b(?P<word>twice|doubled|tripled)\b",
    re.IGNORECASE,
)


def _ratios(label, blanked, declared, unused) -> list[str]:
    """ "N times" between two numbers its sentence declares: their ratio must be N ("more
    than ten times the 186" for 862 is 4.6)."""
    problems = []
    spans = _spans_of(blanked, declared, unused)
    for m in TIMES.finditer(blanked):
        n = m["n"] or m["word"]
        if "." in n:  # "8.2 times" is a calculated number, declared and checked as such
            continue
        value = NUMBER_WORD_VALUES.get(n.lower()) if not n[0].isdigit() else float(n)
        if not value:
            continue
        # "N times the 186": the second number follows right after.
        after = blanked[m.end() : m.end() + 24]
        if not m["word"] and not re.match(
            r"\s*(?:the|its|that of|as (?:high|large|much) as)?\s*[~≈]?\s*\d", after
        ):
            continue
        sentence_start = max([0, *(x.end() for x in CLAUSE.finditer(blanked[: m.start()]))])
        sentence_end = min(
            [len(blanked), *(m.end() + x.end() for x in CLAUSE.finditer(blanked[m.end() :]))]
        )
        near = [d for st, _, _, d in spans if sentence_start <= st < sentence_end]
        near = [d for d in near if d.value and not d.text.rstrip().endswith("%")]
        if len(near) < 2:
            continue
        before = [d for st, _, _, d in spans if sentence_start <= st < m.start() and d in near]
        after = [d for st, _, _, d in spans if m.end() <= st < sentence_end and d in near]
        if not before or not after:
            continue
        x, y = abs(before[-1].value), abs(after[0].value)
        ratio = max(x, y) / min(x, y)
        q = (m["q"] or "").lower()
        if q in ("more than", "over", "at least"):
            ok = ratio >= value * 0.98
        elif q in ("less than", "under"):
            ok = ratio <= value * 1.02
        else:
            ok = abs(ratio - value) <= max(0.5, 0.15 * value)
        if not ok:
            problems.append(
                f"{label}: '{m.group(0).strip()}' does not fit {before[-1].text} and "
                f"{after[0].text}: one is {ratio:.1f} times the other."
            )
    return problems


RANGE_BEFORE = re.compile(
    r"\b(?:range|band|interval|between)\b(?:\s+(?:of|from|is|was))?\s*$", re.IGNORECASE
)
RANGE_JOIN = re.compile(r"\s*(?:%|[^\W\d]+\d?(?:/[^\W\d]+)?)?\s*(?:to|and|-|–)\s*")


# The tokens that tell two ends of one quantity apart: before and after, then and now, the
# low and high of a band. Without them, the two ends of a range or change share a family.
END_TOKENS = re.compile(
    r"(?:(?<=[|_])|^)(?:before|after|then|now|start|end|first|last|low|high|p10|p50|p90|min|"
    r"max|ctx_\w+)(?=[|_]|$)"
)


def _family(leaf: _Leaf) -> str:
    """What a value is, without which end of it: "details|before_mean|regime_change|oil_rate"
    and "...after_mean..." share a family, "calibrated_band_sm3|p10" and
    "cumulative_oil_sm3|p10" (the bootstrap's) do not."""
    plain = END_TOKENS.sub("", leaf.labels.lower())
    return re.sub(r"_+(?=[|]|$)|(?<=[|])_+|\|+(?=\||$)", "", plain).strip("|_")


def _unpaired(label, blanked, declared, unused, log: EvidenceLog, leaves_of) -> list[str]:
    """Two numbers written as one range ("range 324,000 to 470,000") or one change ("from
    4 050 to 1 146", "from 63% then to 97% now") that their result holds only as different
    things: the low end of one band and the high end of another, the before of one event and
    the after of another, a water cut and an uptime. Two table rows of one column are a
    pair ("from 72% in 2011-01 to 95% in 2013-12")."""
    problems = []
    spans = _spans_of(blanked, declared, unused)
    for (a_start, a_end, _, a), (b_start, _, _, b) in zip(spans, spans[1:], strict=False):
        between = blanked[a_end:b_start]
        ranged = RANGE_BEFORE.search(blanked[max(0, a_start - 40) : a_start]) and (
            RANGE_JOIN.fullmatch(between)
        )
        changed = PAIR_FROM.search(blanked[max(0, a_start - 20) : a_start]) and re.fullmatch(
            r"\s*(?:%|Sm3(?:/d)?|bar|m)?\s*(?:then\s+|before\s+)?(?:to|->|→)\s*"
            r"(?:about|around|roughly|nearly)?\s*[~≈]?\s*",
            between,
        )
        if not (ranged or changed) or a.evidence_id != b.evidence_id:
            continue
        item = log.get(a.evidence_id)
        # A calculation's inputs are typed from other results: it says nothing of their kind.
        # Two years are a span of time ("from 2008 to 2010"), no change of a value.
        if (
            item is None
            or item.tool == "calculate"
            or all(1900 <= x.value <= 2100 and x.value == int(x.value) for x in (a, b))
        ):
            continue
        leaves = leaves_of(item)
        fits_a, fits_b = _fits_of(a), _fits_of(b)
        held_a = [
            x
            for x in leaves
            if x.value is not None and any(_reads_as(r, [x.value]) for r in fits_a)
        ]
        held_b = [
            y
            for y in leaves
            if y.value is not None and any(_reads_as(r, [y.value]) for r in fits_b)
        ]
        if not held_a or not held_b:
            continue

        def exactly(held: list[_Leaf], declared: DeclaredNumber) -> list[_Leaf]:
            # The declared value, where the result holds it exactly, says which is meant
            # ("31%" declared 0.3107 is the water cut, not a month of 31 days).
            v = declared.value
            return [
                lf
                for lf in held
                if any(abs(lf.value - x) <= 1e-9 * max(1, abs(x)) for x in (v, v * 100, v / 100))
            ] or held

        held_a, held_b = exactly(held_a, a), exactly(held_b, b)
        # Depths ("from 3,135 to 3,221 m" spans a log's zones) and numbers only a tool's prose
        # holds (its summary lines name them loosely) are not judged.
        if "metres" in (
            _written_unit(blanked, a_start + len(a.text)),
            _written_unit(blanked, b_start + len(b.text)),
        ) or any(
            all(
                re.match(r"summary\b", lf.labels) or re.search(r"_m\b|depth", lf.labels)
                for lf in held
            )
            for held in (held_a, held_b)
        ):
            continue

        def first_day(leaf: _Leaf) -> str:
            return min(leaf.record_days, default="")

        if not any(
            _family(x) == _family(y)
            and (
                (x.row_day is not None and y.row_day is not None)
                or x.record == y.record
                # across a series of events: the first one's before, a later one's after
                or (
                    x.record_days
                    and y.record_days
                    and re.search(SIDE_LABEL["before"], x.labels)
                    and re.search(SIDE_LABEL["after"], y.labels)
                    and first_day(x) <= first_day(y)
                )
            )
            for x in held_a
            for y in held_b
        ):
            problems.append(
                f"{label}: '{a.text}' and '{b.text}' are written as the two ends of one "
                f"{'range' if ranged else 'change'}, but in {item.id} they are "
                f"{held_a[0].labels.replace('|', ' ')} and {held_b[0].labels.replace('|', ' ')}."
            )
    return problems


def _range_order(label, blanked, declared, unused) -> list[str]:
    """A range written high to low ("calibrated range 470,000 to 324,000"): its ends read
    as each other's."""
    spans = sorted(
        (start, end, id(d), d)
        for d in declared
        if d not in unused
        for start, end in _standalone(blanked, d.text)[:1]
    )
    problems = []
    for (a_start, a_end, _, a), (b_start, _, _, b) in zip(spans, spans[1:], strict=False):
        if (
            RANGE_BEFORE.search(blanked[max(0, a_start - 40) : a_start])
            and RANGE_JOIN.fullmatch(blanked[a_end:b_start])
            and a.value > b.value
        ):
            problems.append(
                f"{label}: the range '{a.text} to {b.text}' runs from high to low; its low "
                f"end is {b.text}. Write a range low to high."
            )
    return problems


MOVED_TO = re.compile(
    rf"\b(?:(?P<rise>{RISE_WORDS}|risen|gone up|went up)|(?P<fall>{FALL_WORDS}|fallen|gone down|"
    r"went down))\s+to\s*(?:about|around|roughly|nearly|almost|some)?\s*[~≈]?\s*$",
    re.IGNORECASE,
)


def _moved_to(label, text, blanked, declared, unused, items, leaves_of) -> list[str]:
    """A level written as reached by a rise or a fall ("water cut has fallen to 69%") where
    the change its evidence records for that quantity ran the other way. Judged against the
    change dated in the sentence, else the latest one, of the well the sentence names."""
    problems = []
    for d in declared:
        if d in unused:
            continue
        for start, _ in _standalone(blanked, d.text)[:1]:
            begin = max([0, *(m.end() for m in CLAUSE.finditer(blanked[:start]))])
            moved = MOVED_TO.search(blanked[begin:start])
            if not moved:
                continue
            names = list(_NAMES.finditer(blanked[begin : begin + moved.start()]))
            if not names:
                continue
            quantity = next(
                q for q, (t, _) in QUANTITIES.items() if re.fullmatch(t, names[-1].group(0), re.I)
            )
            pattern = QUANTITIES[quantity][1]
            wells = _named(text, start)
            # The clause's dates, else the sentence's ("shut in from 2014-11-30; after it came
            # back the water cut had gone down to 31%"): a level after a dated event is no
            # verdict on an earlier change.
            sentence = max([0, *(m.end() for m in re.finditer(r"(?<=[.!?])\s+", text[:start]))])
            dates = DATE.findall(text[begin:start]) or DATE.findall(text[sentence:start])
            changes = [
                (min(lf.record_days), lf, item)
                for item in items
                for lf in leaves_of(item)
                if lf.sign is not None
                and lf.record_days
                and re.search(pattern, lf.labels)
                and (not wells or lf.well is None or _well_key(lf.well) in wells)
            ]
            if dates:
                changes = [
                    c for c in changes if any(day.startswith(dates[-1]) for day in c[1].record_days)
                ]
            elif changes:
                latest = max(c[0] for c in changes)
                changes = [c for c in changes if c[0] == latest]
            way = 1 if moved["rise"] else -1
            own = next((item for item in items if item.id == d.evidence_id), None)
            if not dates and _paired_move(d.value, own, leaves_of, way):
                continue  # its own result has it as an "after" reached that way
            if changes and all(lf.sign == -way for _, lf, _ in changes):
                when, _, item = changes[0]
                problems.append(
                    f"{label}: '{d.text}' is written as reached by a "
                    f"{'rise' if way == 1 else 'fall'}, but in {item.id} the {quantity}'s "
                    f"change on {when} is a {'fall' if way == 1 else 'rise'}."
                )
    return problems


def _paired_move(value: float, item: Evidence | None, leaves_of, way: int) -> bool:
    """Does the result hold this value as an "after" whose "before" (same record, same key)
    lies the written way: a plug job's oil 177 -> 857 is a rise, whatever the latest
    regime change did."""
    if item is None:
        return False
    leaves = [lf for lf in leaves_of(item) if lf.value is not None]
    for lf in leaves:
        parts = lf.labels.split("|")
        if "after" not in parts or abs(lf.value - value) > 1e-6 * max(1.0, abs(value)):
            continue
        for other in leaves:
            mine = other.labels.split("|")
            if (
                other.record == lf.record
                and "before" in mine
                and mine[-1] == parts[-1]
                and (lf.value - other.value) * way > 0
            ):
                return True
    return False


# Numbers written as words: fractions and tens ("two thirds" would need declaring as 67%).
# Multiples up to ten ("twice", "four times") are as free as the digits. "First half of
# 2010", "half-life" and "half below 3164 m" (the depth half the flow enters below) are not
# amounts.
NUMBER_WORDS = re.compile(
    r"\b(?:(?:two|three)[\s-]+(?:thirds|quarters|fifths)|(?:one|a)[\s-]+(?:third|quarter|fifth)"
    r"(?=\s+of\b)|one[\s-]+(?:third|quarter)|"
    r"(?<!first )(?<!second )(?<!latter )(?<!last )(?<!other )half(?![\s-]+(?:life|year|way)\b)"
    r"(?!\s+(?:of\s+)?(?:the\s+)?(?:\w+\s+){0,3}?(?:below|above|enters|entered|comes|came)\b)|"
    r"halved|halving|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b",
    re.IGNORECASE,
)


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

    def readings_of(number: DeclaredNumber) -> list[Reading]:
        # A unit the text writes right after the declared number scales it too: "4.58"
        # declared, "4.58 MSm3" written.
        spans = _standalone(masked, number.text)
        if not spans:  # declared but not written as such (a reader's edit can do this)
            return _readings(number.text)
        end = spans[0][1]
        unit = SUFFIX_AFTER.match(masked, end)
        # The first end of a range takes the scale written after the second: "0.32 to 0.47
        # million Sm3" (not "316,000 to 1.11 million", written in full).
        if (
            not unit
            and (span := RANGE_SCALE.match(masked, end))
            and all(abs(v) < 1000 for v, _, _ in _readings(number.text))
        ):
            unit = SUFFIX_AFTER.match(masked, span.end())
        # (+2 keeps the "Sm" that "M" and "k" need after them)
        readings = _readings(number.text + (masked[unit.start() : unit.end() + 2] if unit else ""))
        # "about 70%" for 69.39%: an approximate number is good to 1% of itself.
        if APPROX.search(masked[: spans[0][0]]):
            readings = [(v, max(u, 0.01 * abs(v)), sc) for v, u, sc in readings]
        return readings

    def fits_of(number: DeclaredNumber) -> list[Reading]:
        return [f for r in readings_of(number) if (f := _pinned(r, number.value))]

    def attributed(number: DeclaredNumber, fits: list[Reading], item: Evidence) -> str | None:
        """The attribution problem of a declared text where it is written. A text declared
        twice (P-4's 31% step, P-2's 31% water cut) holds where either declaration does; one
        written more often than declared repeats a value already checked."""
        spans = _standalone(blanked, number.text)
        same = [d for d in declared if d not in unused and d.text == number.text]

        def fails(d: DeclaredNumber, span: tuple[int, int]) -> str | None:
            other = item if d is number else log.get(d.evidence_id)
            if other is None:
                return "comes from unknown evidence"
            found = fits if d is number else fits_of(d)
            return _attribution(
                text, blanked, d.text, found, other, leaves_of(other), spans=[span], exact=d.value
            )

        failing = [(span, wrong) for span in spans if (wrong := fails(number, span))]
        failing = [
            (span, wrong)
            for span, wrong in failing
            if all(fails(o, span) for o in same if o is not number)
        ]
        # Only a well's attribution is tolerated on the extra occurrences: a unit, a sign or
        # a change written wrong is wrong wherever it stands ("19 years" beside "19 months").
        hard = [f for f in failing if "not for the well its sentence names" not in f[1]]
        if hard:
            return hard[0][1]
        return failing[0][1] if len(failing) > max(0, len(spans) - len(same)) else None

    for number in declared:
        if number in unused:
            continue
        where = number.evidence_id
        item = log.get(where)
        readings = readings_of(number)
        fits = [f for r in readings if (f := _pinned(r, number.value))]
        at = _standalone(masked, number.text)[0][1]
        year = _yearlike(number.text, masked[at:])
        exempt = [n for n in _numbers_in_text(number.text) if _free(*n, pool, free, year)]
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
        elif wrong := _scale_mismatch(fits, leaves_of(item), item):
            problems.append(f"{label}: '{number.text}' {wrong}.")
        elif (wrong_row := _row_mismatch(masked_dates, number.text, fits, item)) is not None:
            problems.append(
                f"{label}: '{number.text}' is in {where}, but not in its {wrong_row[0]} row, "
                f"which the text ties it to; that row has {wrong_row[1]}."
            )
        elif wrong := attributed(number, fits, item):
            # Often only the id is wrong: another cited result has the number as written.
            better = [
                e
                for e in sorted(citable or ())
                if e != where
                and (other := log.get(e)) is not None
                and any(_reads_as(r, _Pool([other]).numbers) for r in fits)
                and not attributed(number, fits, other)
            ]
            hint = f" {better[0]} has it as written: declare that id." if better else ""
            problems.append(
                f"{label}: '{number.text}' {wrong}. Write what the evidence says.{hint}"
            )
        elif record := _record_mismatch(masked_dates, number.text, fits, leaves_of(item)):
            problems.append(
                f"{label}: '{number.text}' is in {where}, but the text ties it to {record[0]}, "
                f"and {record[1]}. Give each number the date of its own event."
            )
        elif misfit := _best_fit(text, blanked, number.text, fits, leaves_of(item)):
            problems.append(f"{label}: '{number.text}' {misfit}. Write what {where} says.")

    problems += _reversed(label, blanked, declared, unused, log, leaves_of)
    items = [i for i in (log.get(e) for e in (citable or {d.evidence_id for d in declared})) if i]
    problems += _moved_to(label, text, blanked, declared, unused, items, leaves_of)
    problems += _range_order(label, blanked, declared, unused)
    problems += _unpaired(label, blanked, declared, unused, log, leaves_of)
    problems += _time_order(label, blanked, declared, unused, items)
    problems += _inconsistent(label, blanked, declared, unused)
    problems += _ratios(label, blanked, declared, unused)
    problems += _year_records(label, blanked, declared, unused, items, leaves_of)
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
    aliased: set[str] = set()
    for match in WRITTEN_NUMBER.finditer(rest):
        written = match.group(0).strip()
        # Either convention: "0,80" is 0 and 80, or 0.8; "1.234.567" is a number too.
        values = [n for european in (False, True) for n in _numbers_in_text(written, european)]
        year = _yearlike(written, rest[match.end() :])
        if values and all(_free(*n, pool, free, year) for n in values):
            continue
        # Read as written: a Norwegian reading counts only with a decimal comma ("0,80").
        if exact(written) & declared_values and (
            exact(written) <= declared_values or DOTTED_THOUSANDS.fullmatch(written)
        ):
            # Written in another form than declared ("3 150" for "3150"): still judged for
            # what it is said to be where it is written (a unit, a well, a quantity).
            source = next((d for d in unused if exact(d.text) & exact(written)), None)
            item = log.get(source.evidence_id) if source else None
            if source and item and written not in aliased:
                aliased.add(written)
                alias = source.model_copy(update={"text": written})
                fits = fits_of(alias)
                if fits and (wrong := attributed(alias, fits, item)):
                    problems.append(f"{label}: '{written}' {wrong}. Write what the evidence says.")
            continue
        if values and written not in leftover:
            leftover.append(written)
    words = list(dict.fromkeys(m.group(0) for m in NUMBER_WORDS.finditer(rest)))
    if words:
        problems.append(
            f"{label}: {', '.join(repr(w) for w in words)} is a number in words, which the "
            "checker cannot check; write the number from the evidence in digits and declare it."
        )
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
        # The value nearest the number as written, so "3 150" is declared as the 3150 the
        # evidence holds rather than a 3147 that also rounds to it; on a tie, the first item.
        # "0.69%" is no water cut of 0.69: a percent under 1 reads only as a share.
        def scales(r: Reading, text: str = text) -> tuple[float, ...]:
            if text.rstrip().endswith(("%", "percent", "pct")) and abs(r[0]) < 1:
                return tuple(sc for sc in r[2] if sc != 1.0)
            return r[2]

        hits = [
            (min(abs(n - r[0] * sc) / max(abs(r[0] * sc), 1e-9) for sc in r[2]), k, n, item.id)
            for k, item in enumerate(i for i in items if i is not None)
            for n in _Pool([item]).numbers
            for r in _readings(text)
            if scales(r) and _reads_as((r[0], r[1], scales(r)), [n])
        ]
        if hits:
            _, _, value, evidence_id = min(hits, key=lambda h: (h[0], h[1]))
            found.append(
                f'{{"text": "{text}", "value": {value:g}, "evidence_id": "{evidence_id}"}}'
            )
    return f" These are in the evidence; declare them as {', '.join(found)}." if found else ""


def _plain_well(name: str) -> str:
    return re.sub(r"\s+", "", name.removeprefix("15/9-")).upper()


# The reports name a wellbore ("F-11 B"), the production data its well ("F-11"): one well.
WELLBORE_OF = {_plain_well(k): _plain_well(v) for k, v in WELLBORES.items()}


def _well_key(name: str) -> str:
    key = _plain_well(name)
    return WELLBORE_OF.get(key, key)


RELATED_WELL = re.compile(r"\b(?:as|with|than|like|alongside|together with)\s+$", re.I)


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
    # A well the results name, set beside the one they are about ("at the same time as F-14",
    # from a shut-in's "Same time as F-14"), is not a claim about that well.
    named = dict.fromkeys(
        m.group(0).strip()
        for m in WELL.finditer(claim)
        if not (
            _well_key(m.group(0)) in mentioned
            and RELATED_WELL.search(claim[max(0, m.start() - 30) : m.start()])
        )
    )
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
    blanked = _blank(text, WELL, DATE, CALENDAR_WINDOW, REFERENCE)  # positions as in text
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
        for start, stop in _standalone(blanked, _normalize(number.text)):
            end = next(e for e in ends if e >= start)
            begin = max([0] + [e for e in ends if e < start])
            # "...335 thousand; the fit suggests so": the clause after a semicolon may hedge it.
            if not _hedged(text[begin:end], start - begin, stop - begin, stops=","):
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
    text = canonical(text)
    spans = [s for s in display_spans(text, [], log) if s["kind"] == "quote"]
    plain = text
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
    text = canonical(text)
    numbers = [n.model_copy(update={"text": canonical(n.text)}) for n in numbers or []]
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


# Readability, not truth: what a manager or a judge reads in a claim or the summary. The
# loop asks for it once per run, so style never costs an answer its verification.
READER_JARGON = re.compile(r"\br2(?:_log)?\b|r²|\bAIC\b|\bP(?:10|50|90)\b")


def _noisy(text: str, value: float) -> tuple[float, int] | None:
    """A declared number that reads as noise: more than 3 significant digits in a whole
    number of 10,000 or more ("359 007", "1,133,053") or a decimal under 10 ("0.6939").
    "1 551", "4,050" and "72.3%" read fine. Of the two conventions, the reading that shows
    the declared value counts ("9.000" for 9000 is nine thousand)."""
    readings = [
        (v, decimals)
        for european in (False, True)
        for v, decimals, _ in _numbers_in_text(text, european)
    ]
    if not readings:
        return None
    v, decimals = min(
        readings, key=lambda r: min(abs(r[0] * scale - value) for scale in (1, 0.01, 100))
    )
    if decimals <= 0:
        significant = len(str(round(v)).rstrip("0"))
        return (v, decimals) if v >= 10_000 and significant > 3 else None
    significant = len(f"{v:.{decimals}f}".replace(".", "").lstrip("0"))
    return (v, decimals) if v < 10 and significant > 3 else None


def _three_digits(v: float, decimals: int) -> str:
    if decimals > 0:
        return f"{v:.3g}"
    if v >= 1e6:
        return f"{v / 1e6:.3g} million"
    return f"{round(v, 3 - len(str(round(v)))):,.0f}"


# A claim a reader takes in at once holds a few numbers; the rest go to its detail.
MAX_CLAIM_NUMBERS = 4
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def claim_numbers(claim: str) -> list[str]:
    """The numbers a reader meets in a claim: not its dates, years, well names, windows or
    quotes."""
    outside = QUOTED.sub(" ", _translations_outside(canonical(claim)))
    return WRITTEN_NUMBER.findall(YEAR.sub(" ", _mask(outside)))


def readability(submission: Submission) -> list[str]:
    """Claims and a summary a reader takes in at a glance: numbers rounded to three
    significant digits, no forecaster's or fit labels, at most MAX_CLAIM_NUMBERS numbers in
    a claim. Exact values, labels and the other numbers belong in a finding's detail.
    Numbers inside quotes are the document's and stay as written."""
    submission = _canonical_submission(submission)
    parts = [
        (f"Finding {n}", "claim", f.claim, f.numbers or [])
        for n, f in enumerate(submission.findings, start=1)
    ]
    parts.append(("Summary", "summary", submission.summary, submission.summary_numbers or []))
    problems = []
    for label, where, text, declared in parts:
        outside = QUOTED.sub(" ", _translations_outside(text))
        long = []
        for number in declared:
            if number.text in outside and (noisy := _noisy(number.text, number.value)):
                long.append(f"'{number.text}' (about {_three_digits(*noisy)})")
        if long:
            detail = " and keep the exact values in the detail" if where == "claim" else ""
            problems.append(
                f"{label}: round {', '.join(dict.fromkeys(long))} in the {where} to three "
                f"significant digits{detail}; a reader takes in three. Declare the rounded "
                "text with the same value and evidence id."
            )
        if where == "claim" and (count := len(claim_numbers(text))) > MAX_CLAIM_NUMBERS:
            problems.append(
                f"{label}: the claim holds {count} numbers; keep the two or three that make "
                "the point and move the others to its detail, where numbers are looked up in "
                "the evidence, not declared."
            )
        if jargon := sorted({m.group(0) for m in READER_JARGON.finditer(outside)}):
            plain = []
            if any(j.startswith("P") for j in jargon):
                plain.append("say 'central estimate' for P50 and 'range' for P90 to P10")
            if any(not j.startswith("P") for j in jargon):
                plain.append("leave fit statistics out")
            labels = "is a label" if len(jargon) == 1 else "are labels"
            problems.append(
                f"{label}: {', '.join(jargon)} {labels} a reader may not know; in the {where}, "
                f"{' and '.join(plain)}. The detail can keep them."
            )
    return problems


# A pressure that fell or collapsed: the misreading of a downhole gauge that died and reads 0.
PRESSURE_DROP = re.compile(
    r"\b(?:pressure|BHP)\b(?:\s+[\w-]+){0,3}?\s+(?P<verb>collaps\w*|crash\w*|plung\w*|dropp?\w*|"
    r"fell|fall(?:s|ing|en)?|declin\w*|vanish\w*|depleted|went\s+to\s+(?:zero|0)|"
    r"to\s+(?:zero|0)\b|zero\b|0\b|lost\b|gone\b|depressur\w*)"
    # "the reservoir lost (its) pressure", "losing pressure"
    r"|\b(?P<lost>lost|losing|loses|lose)\s+(?:its\s+|all\s+|the\s+)?(?:reservoir\s+)?pressure\b"
    r"|\b(?P<noun>collapse|crash|plunge|drop|fall|loss|decline)\s+(?:in|of)\s+(?:the\s+)?"
    r"(?:reservoir\s+|downhole\s+|bottom[\s-]?hole\s+)?pressure\b",
    re.IGNORECASE,
)
SUDDEN = re.compile(r"collaps|crash|plung|vanish|zero|\b0\b|lost|los[ei]|gone|noun", re.IGNORECASE)
DENIED = re.compile(
    r"\b(?:not|no|never|without|rather than|instead of|ruled? out|rules out|ruling out|"
    r"exclud\w*|neither|nor|"
    r"misread\w*|mistak\w*|false|naive\w*|looks like|apparent(?:ly)?|seeming(?:ly)?|but)\b|n't\b",
    re.IGNORECASE,
)


def _gauge_failures(log: EvidenceLog, leaves_of) -> dict[str, tuple[str, str]]:
    """Each well's downhole gauge failure in the run's evidence: (first day, evidence id)."""
    failures: dict[str, tuple[str, str]] = {}
    for item in log:
        for lf in leaves_of(item):
            if lf.well and lf.record_days and re.search(r"gauge_failure|dead_gauge", lf.labels):
                day = min(lf.record_days)
                key = _well_key(lf.well)
                if key not in failures or day < failures[key][0]:
                    failures[key] = (day, item.id)
    return failures


def _pressure_misread(
    label: str, text: str, failures: dict[str, tuple[str, str]], about: set[str] | None = None
) -> list[str]:
    """A pressure said to fall or collapse where the evidence records the well's downhole
    gauge failing (it reads 0 from then on): the reading the time machine exists to refute.
    Judged when the sentence's dates reach the failure, or, with no date, when it says the
    pressure collapsed or went to zero; a sentence that denies it ("not a pressure
    collapse") or calls it a gauge reading passes."""
    if not failures:
        return []
    problems = []
    for sentence in FULL_SENTENCE.split(text):
        m = PRESSURE_DROP.search(sentence)
        # "not a pressure collapse", "the pressure did not collapse", "looks like a pressure
        # collapse but": denied before or within it. "..., not a sensor fault" after it is no
        # denial of the pressure.
        if not m or DENIED.search(sentence[: m.end()]):
            continue
        # "the BHP gauge went to zero", "BHP readings dropped to 0": the reading, not the
        # pressure. A denial in its own clause after it counts too ("... is not expected").
        if re.search(r"\b(?:gauge|sensor|readings?|data|signal|log)\b", m.group(0), re.I):
            continue
        # "BHP went to zero": the measurement's value, as the data shows it. A pressure said
        # to fall, or a collapse, is the reading of the reservoir.
        if re.match(r"BHP\b", m.group(0), re.I) and not re.search(
            r"collaps|crash|plung", m.group(0), re.I
        ):
            continue
        rest = sentence[m.end() :]
        if DENIED.search(rest[: rest.find(",") if "," in rest else len(rest)]):
            continue
        start = text.find(sentence)
        # The wells the sentence names, else those its evidence is about.
        named = {w for w in _named(text, start + m.start())} or (about or set())
        dates = DATE.findall(sentence)
        for well in sorted(named & set(failures)):
            day, eid = failures[well]
            # A collapse, a zero or a loss is never what the data shows for a well whose
            # gauge died: judged whatever the date. A fall is judged where its dates reach
            # the failure.
            sudden = SUDDEN.search(m.group(0) if m["verb"] else "noun")
            reached = sudden or (dates and any(d >= day[: len(d)] for d in dates))
            if reached:
                problems.append(
                    f"{label} says '{m.group(0).strip()}', but {eid} records {well}'s downhole "
                    f"gauge reading 0 from {day}: a dead gauge, not a pressure drop. Say the "
                    "gauge failed, or leave the pressure out."
                )
                break
        if problems:
            break
    return problems


# "Stayed within", "never exceeded", "not above": a claim that every limit held.
WITHIN = re.compile(
    r"\b(?:never|not|no|without)\s+(?:\w+\s+){0,4}?(?:exceed\w*|surpass\w*|breach\w*|broke|"
    r"went\s+over|gone\s+over|above|over)\b"
    # A history ("stayed", "has always run", "has been"), not a state at one date: "the
    # field is still below its design limits" at the end of 2010 is true after injection
    # passed its capacity in 2008.
    r"|\b(?:stayed|remained|kept|stays|remains|ran|run|operated|been)\s+"
    r"(?:\w+\s+){0,3}?(?:within|inside|below|under)\b"
    r"|\bwithin\s+(?:\w+\s+){0,4}?(?:design|limits?|capacit\w*)\b",
    re.IGNORECASE,
)
LIMIT_NAMES = {
    "liquid_sm3_per_day": r"\bliquid",
    "oil_sm3_per_day": r"\boil\b",
    "water_cut": r"water[\s_-]?cut",
    "water_inj_sm3_per_day": r"\binject",
    "water_sm3_per_day": r"water[\s-]+(?:handling|capacity|rates?|production)|produced water",
}


def _within_limits(label: str, text: str, items: list[Evidence]) -> list[str]:
    """A sentence saying the field stayed within its design limits while the cited
    field_vs_design result has a limit reached or passed: judged for the limits the
    sentence names, or all of them when it names none ("never exceeded any design limit").
    A sentence about one well, or bounded in time ("until 2012"), is not judged."""
    results = [e for e in items if e.tool == "field_vs_design" and isinstance(e.result, dict)]
    if not results:
        return []
    for sentence, m in (
        (part, m) for part in FULL_SENTENCE.split(text) for m in WITHIN.finditer(part)
    ):
        # Every such phrase of a sentence: "stayed within oil limits ... and never exceeded
        # water injection capacity" claims both.
        if not re.search(r"design|limit|capacit|plan", sentence, re.I):
            continue
        # Who stayed within: the clause before the phrase ("... F-12 exceeded the design
        # water cut while the platform stayed under all its capacities" is about the
        # platform). A clause about one well is not judged, nor is one bounded in time.
        stop = max(
            sentence.rfind(c, 0, m.start())
            for c in (",", ";", ":", " while ", " but ", " whereas ")
        )
        subject = sentence[stop + 1 if stop >= 0 else 0 : m.start()]
        whole_field = re.search(r"\b(?:field|platform|facilit\w*)\b", subject, re.I)
        bounded = re.search(
            r"\b(?:until|before|up to|through)\s+(?:the\s+)?(?:end\s+of\s+|early\s+|mid-?\s*|"
            r"late\s+)?(?:\d{4}|[A-Z][a-z]+\s+\d{4}|\d{4}-\d{2})"
            # A state at one date: "was not over its design limits at end-2010".
            r"|\b(?:at|as of|on)\s+(?:the\s+)?end[\s-]+(?:of\s+)?(?:\d{4}|[A-Z][a-z]+\s+\d{4})"
            r"|\b(?:as of|on)\s+\d{4}-\d{2}-\d{2}",
            sentence,
        )
        if (WELL.search(subject) or (WELL.search(sentence) and not whole_field)) or bounded:
            continue
        # The limits the claim is about: named after it in its clause ("without exceeding
        # liquid capacity"), else before it ("water handling stayed below capacity").
        part_end = sentence.find(",", m.end())
        after = sentence[m.start() : part_end if part_end >= 0 else len(sentence)]
        named_after = [k for k, pattern in LIMIT_NAMES.items() if re.search(pattern, after, re.I)]
        # The subject itself ("the field's oil, liquid and water handling rates stayed under
        # design limits") names the limits it means.
        named_before = [
            k for k, pattern in LIMIT_NAMES.items() if re.search(pattern, subject, re.I)
        ]
        named = named_after or named_before
        # "stayed within its design limits: peak oil ..., water cut ...": the limits as a
        # whole, with examples after; "design limits for liquid, oil or water cut" names
        # the ones it means.
        whole = (
            re.match(
                r"\s*(?:any|all|every)\s+(?:[\w-]+\s+){0,2}?(?:limits?|capacit\w*)\b"
                r"(?!\s+(?:for|on|of)\b)",
                sentence[m.end() :],
                re.I,
            )
            or re.match(
                r"[\w\s']{0,40}?\b(?:design\s+)?limits\b(?!\s+(?:for|on|of)\b)",
                sentence[m.end() - len("within") if m.group(0).endswith("within") else m.end() :],
                re.I,
            )
            or re.search(r"\b(?:any|all)\s+(?:\w+\s+){0,2}limits?\b", m.group(0), re.I)
        )
        # "within its oil and liquid design limits" names the ones it means.
        lead = re.match(r"[\w\s',]{0,70}?\b(?:limits?|capacit\w*)\b", sentence[m.end() :])
        listed = [
            k
            for k, pattern in LIMIT_NAMES.items()
            if lead and re.search(pattern, lead.group(0), re.I)
        ]
        if listed:
            named = listed
        elif named_before:
            named = named_before
        elif whole:
            named = []
        # A limit the sentence itself says was reached, in a clause of its own after a turn
        # ("..., but field water cut reached the 80% limit"; "; it reached the 0.80 design
        # water cut"), is not claimed as held. Not the claim's own clause: "within oil,
        # liquid and injection rate limits but exceeded the 80% water cut" holds injection.
        turn = re.search(
            r"(?:,?\s+\b(?:but|while|whereas|although|though|yet)\b|;)", sentence[m.end() :], re.I
        )
        tail = sentence[m.end() + turn.end() :] if turn else ""
        clause = re.split(r",?\s+\b(?:but|while|whereas|although|though|yet)\b|;", tail)[0]
        exceeded = re.search(
            r"\b(?:reach\w*|pass\w*|exceed\w*|hit|went over|was over|rose above|broke)\b",
            clause,
            re.I,
        )
        conceded = {
            k for k, pattern in LIMIT_NAMES.items() if exceeded and re.search(pattern, clause, re.I)
        }
        for item in results:
            against = item.result.get("field_against_limits") or {}
            limits = item.result.get("design_limits") or {}
            for key in [k for k in named or list(against) if k not in conceded]:
                months = (against.get(key) or {}).get("months_at_or_above") or 0
                if months > 0:
                    limit = limits.get(key) or {}
                    return [
                        f"{label} says the field stayed within its design limits, but {item.id} "
                        f"has the {limit.get('limit', key)} ({limit.get('value', '?')}) reached "
                        f"or passed in {months} months (up to {against[key].get('max')}). Say "
                        "which limits held and which did not."
                    ]
    return []


# A small whole number of days, weeks, months or years ("after about 6 years"): free of
# declaring as a number, but it still states a duration the evidence must hold.
SMALL_DURATION = re.compile(
    r"(?<![\w.,])(?P<n>\d{1,2}(?:\.\d)?)(?:\s*-\s*|\s+)(?P<u>years?|months?|weeks?|days?)\b",
    re.IGNORECASE,
)
BOUND_BEFORE = re.compile(
    r"\b(?:within|under|less than|more than|over|at least|at most|up to|every|each|per|next|"
    r"last|past|first|the|a)\s*$",
    re.IGNORECASE,
)
APPROX = re.compile(
    r"(?:about|around|roughly|approximately|nearly|near|almost|some|circa|ca\.|close to|"
    r"(?:just|slightly|a little|a bit) (?:under|over|below|above)|on the order of|~|≈)\s*$",
    re.I,
)
IN_MONTHS = {"days": 1 / 30.44, "weeks": 1 / 4.345, "months": 1.0, "years": 12.0}
# A duration the text gives the plan or a report is checked against the reports' own.
REPORT_TOOLS = {"search_reports", "read_report_page", "look_at_page", "plan_vs_actual"}
ABOUT_REPORT = re.compile(r"\b(?:plan|PUD|report|expected|assumed|forecast to)\b", re.I)


def _spans(value: Any) -> Iterable[tuple[str, str]]:
    """(start, end) dates of every window in a result or its arguments: a fit's window."""
    if isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
        dated = isinstance(start, str) and isinstance(end, str)
        if dated and DATE.fullmatch(start[:10]) and DATE.fullmatch(end[:10]):
            yield start[:10], end[:10]
        for v in value.values():
            yield from _spans(v)
    elif isinstance(value, list):
        for v in value:
            yield from _spans(v)


def _durations(
    label: str,
    text: str,
    cited: list[Evidence],
    leaves_of,
    free: set[float],
):
    """Small durations the text states that none of its evidence holds, in any unit: "about
    6 years" when the plan says 2 years and the well took 20.6 months. Bounds ("within 2
    years") and windows ("the next 3 months") are not judged, nor evidence with no
    durations at all."""

    def durations(items: list[Evidence]) -> list[float]:
        found = []
        for item in items:
            for lf in leaves_of(item):
                if lf.value is not None and lf.dim in IN_MONTHS:
                    found.append(abs(lf.value) * IN_MONTHS[lf.dim])
                if lf.record_days and len(lf.record_days) > 1:
                    first, last = min(lf.record_days), max(lf.record_days)
                    if len(first) == len(last) == 10:
                        days = (date.fromisoformat(last) - date.fromisoformat(first)).days + 1
                        found.append(days / 30.44)
            for first, last in [*_spans(item.result), *_spans(item.arguments)]:
                if len(first) == len(last) == 10:
                    days = (date.fromisoformat(last) - date.fromisoformat(first)).days + 1
                    found.append(days / 30.44)
        return found

    every = durations(cited)
    reports = durations([e for e in cited if e.tool in REPORT_TOOLS])
    # A gap the model calculated ("24 - 21" for "3 days before first oil") has no unit.
    calculated = [
        abs(lf.value)
        for item in cited
        if item.tool == "calculate"
        for lf in leaves_of(item)
        if lf.value is not None
    ]
    # Declared or not, a small whole number is free of the number check ("6" declared for
    # "6 years" is exempt there), so its duration is judged here.
    for m in SMALL_DURATION.finditer(text):
        held = reports if reports and ABOUT_REPORT.search(_part(text, m.start())) else every
        if not held:
            continue
        n = float(m["n"])
        if n > SMALL_INTEGER or n in free or BOUND_BEFORE.search(text[: m.start()]):
            continue
        unit = DURATIONS.get(m["u"].lower().rstrip("s"))
        if unit not in IN_MONTHS:
            continue
        if any(abs(c - n) <= 0.5 for c in calculated):
            continue
        approx = APPROX.search(text[: m.start()])
        values = [h / IN_MONTHS[unit] for h in held]
        if any(abs(v - n) <= 0.5 or (approx and abs(v - n) <= 0.3 * n) for v in values):
            continue
        nearest = sorted(values, key=lambda v: abs(v - n))[:2]
        shown = ", ".join(f"{v:.3g}" for v in nearest)
        return [
            f"{label}: '{m.group(0)}' is in none of its evidence, which has {shown} {unit} "
            "there. State the duration the evidence gives."
        ]
    return []


TREND = re.compile(
    rf"(?P<name>{_NAMES.pattern})\s+(?:is|are|has\s+been|have\s+been|keeps|continues\s+to\s+be)"
    r"\s+(?:still\s+)?(?P<dir>rising|falling|increasing|decreasing|declining|climbing|"
    r"dropping|growing|improving)\b",
    re.IGNORECASE,
)


def _trends(label: str, text: str, items: list[Evidence], leaves_of) -> list[str]:
    """ "Water cut is falling" for a well whose latest recorded change in water cut is a
    rise: the direction of a level said without numbers, against the evidence's latest
    change of that quantity for the sentence's well."""
    problems = []
    for m in TREND.finditer(text):
        sentence_start = max([0, *(x.end() for x in CLAUSE.finditer(text[: m.start()]))])
        sentence_end = min(
            [len(text), *(m.end() + x.end() for x in CLAUSE.finditer(text[m.end() :]))]
        )
        if DATE.search(text[sentence_start:sentence_end]):
            continue  # a dated trend is a statement about then
        quantity = next(
            (q for q, (t, _) in QUANTITIES.items() if re.fullmatch(t, m["name"], re.I)), None
        )
        if quantity is None:
            continue
        wells = _named(text, m.start())
        changes = [
            (min(lf.record_days), lf, item)
            for item in items
            for lf in leaves_of(item)
            if lf.sign is not None
            and lf.record_days
            and re.search(QUANTITIES[quantity][1], lf.labels)
            and (not wells or lf.well is None or _well_key(lf.well) in wells)
        ]
        if not changes:
            continue
        latest = max(c[0] for c in changes)
        signs = {lf.sign for day, lf, _ in changes if day == latest}
        way = -1 if re.match(r"fall|decreas|declin|dropp", m["dir"], re.I) else 1
        if quantity == "water cut" and m["dir"].lower() == "improving":
            way = -1
        if signs == {-way}:
            item = next(it for day, lf, it in changes if day == latest)
            problems.append(
                f"{label} says the {quantity} is {m['dir'].lower()}, but its latest change in "
                f"{item.id} ({latest}) is a {'rise' if way == -1 else 'fall'}."
            )
    return problems


GAUGE_CLAIM = re.compile(
    r"\b(?:dead|failed|broken|faulty)\s+(?:downhole\s+|pressure\s+|BHP\s+)*gauge"
    r"|\bgauge\s+(?:is\s+|was\s+|has\s+)?(?:dead|died|failed|broke)",
    re.IGNORECASE,
)


MODEL_CLAIM = re.compile(
    r"\b(?:from|on|with|using|by|an?|the)\s+(?:an?\s+|the\s+)?(?P<model>exponential|hyperbolic|"
    r"harmonic)\s+(?:fit|decline|model|curve|forecast)\b",
    re.IGNORECASE,
)


def _model_names(label: str, text: str, items: list[Evidence]) -> list[str]:
    """ "From an exponential fit" for a forecast whose fit is harmonic. Judged where the
    cited fits and forecasts all chose one model; "the exponential model said 302k" next
    to p50_by_model is a comparison, not the fit."""
    chosen = {
        str(
            _get(e.result, "model")
            or _get(_get(e.result, "best"), "model")
            or _get(_get(e.result, "fit"), "model")
            or ""
        ).lower()
        for e in items
        if e.tool in ("fit_decline", "forecast_production")
    } - {""}
    if len(chosen) != 1:
        return []
    problems = []
    for m in MODEL_CLAIM.finditer(text):
        model = m["model"].lower()
        by_model = [
            float(v)
            for e in items
            for k, v in (_get(e.result, "p50_by_model_sm3") or {}).items()
            if str(k).lower() == model and isinstance(v, int | float)
        ]
        nearby = [n for n, _, _ in _numbers_in_text(text[m.end() : m.end() + 60])]
        own_value = any(
            abs(n * sc - v) <= 0.01 * v for v in by_model for n in nearby for sc in (1, 1e3)
        )
        if (
            model not in chosen
            and not own_value
            and not re.search(r"\bsaid\b|by model|would", text[m.end() : m.end() + 40])
        ):
            problems.append(
                f"{label} names {m['model']} the fit, but the cited fit is {next(iter(chosen))}."
            )
    return problems


YEAR_RECORD = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})\s+(?:[\w-]+\s+){0,2}?(?:plug|perforations?|job|workover|"
    r"completion|shut[\s-]?in|log|survey|PLT)s?\b[^,;.]{0,30}$",
    re.IGNORECASE,
)


def _year_records(label, blanked, declared, unused, items, leaves_of) -> list[str]:
    """ "The 2010 plug at 3150 m" where 3150 belongs only to a record of another year (the
    2014 plug): a number given to one year's event that the evidence dates to another."""
    problems = []
    for d in declared:
        if d in unused:
            continue
        for start, end in _standalone(blanked, d.text)[:1]:
            m = YEAR_RECORD.search(blanked[max(0, start - 70) : start])
            if not m:
                continue
            # "the 2014 plug ... 97% of the water measured in 2010": the text dates the
            # number itself, after it.
            after = re.split(r"[;.]\s", blanked[end : end + 60])[0]
            fits = _fits_of(d)
            held = [
                lf
                for item in items
                for lf in leaves_of(item)
                if lf.value is not None and any(_reads_as(r, [lf.value]) for r in fits)
            ]
            dated = [lf for lf in held if lf.record_days]
            if len(dated) < len(held):  # an undated value (a log's depth table) may be the year's
                continue
            covered = any(
                any(day.startswith(m["year"]) for day in lf.record_days)
                for item in items
                for lf in leaves_of(item)
                if lf.record_days
            )
            years = sorted({day[:4] for lf in dated for day in lf.record_days})
            if (
                covered
                and dated
                and not any(y in after for y in years)
                and all(
                    all(not day.startswith(m["year"]) for day in lf.record_days) for lf in dated
                )
            ):
                problems.append(
                    f"{label}: '{d.text}' is given to {m['year']}, but the evidence has it "
                    f"only for {', '.join(years)}."
                )
    return problems


def _gauge_claims(label: str, text: str, dead: set[str]) -> list[str]:
    """A gauge said to be dead now on a well whose gauge, in the evidence, is not dead at the
    end of the data, while another well's is (F-12's dead gauge written as F-14's; F-14's
    gauge read 0 for 8 days in 2010 and came back). A dated clause is judged by its date."""
    if not dead:
        return []
    problems = []
    for m in GAUGE_CLAIM.finditer(text):
        if DENIED.search(text[max(0, m.start() - 25) : m.start()]):
            continue
        part_start = max([0, *(x.end() for x in CLAUSE.finditer(text[: m.start()]))])
        part_end = min([len(text), *(m.end() + x.end() for x in CLAUSE.finditer(text[m.end() :]))])
        part = text[part_start : m.end()]
        owners = [w.group(0) for w in WELL.finditer(part)]
        # The gauge's own part: "oil 34% lower since 2010-01-10, and its gauge is dead"
        # dates the oil, not the gauge.
        own_start = max(
            [part_start, *(part_start + x.end() for x in re.finditer(r",|;|:|\band\b", part))]
        )
        own_end = min(
            [
                part_end,
                *(
                    m.end() + x.start()
                    for x in re.finditer(r",|;|\band\b", text[m.end() : part_end])
                ),
            ]
        )
        if not owners or DATE.search(text[own_start:own_end]):
            continue
        if _well_key(owners[-1]) not in dead:
            problems.append(
                f"{label} gives {owners[-1]} a dead gauge, but at the end of the data the "
                f"evidence has a dead gauge only on {', '.join(sorted(dead))}."
            )
    return problems


def _dead_gauges(log: EvidenceLog, leaves_of) -> set[str]:
    """Wells whose downhole gauge failure runs to (about) the end of the data the evidence
    covers (forecasts' future dates left out)."""
    seen = [
        lf.day
        for item in log
        if item.tool != "forecast_production"
        for lf in leaves_of(item)
        if lf.day and len(lf.day) == 10 and not re.search(r"forecast|horizon", lf.labels)
    ]
    last = max(seen) if seen else None
    if not last:
        return set()
    cutoff = (date.fromisoformat(last[:10]) - timedelta(days=31)).isoformat()
    return {
        _well_key(lf.well)
        for item in log
        for lf in leaves_of(item)
        if lf.well
        and lf.record_days
        and re.search(r"gauge_failure|dead_gauge", lf.labels)
        and max(lf.record_days) >= cutoff
    }


MONTH_RANGE = re.compile(
    r"\b(\d{4})-(\d{2})(?!-\d)\s*(?:to|through|until|-|–)\s*(\d{4})-(\d{2})(?!-\d)"
)
MONTH_COUNT = re.compile(r"\b(\d{1,3})[\s-]*(months?|years?)\b", re.IGNORECASE)
YEAR_RANGE = re.compile(
    r"(?<![\d-])((?:19|20)\d{2})\s*(?:to|through|until|-|–)\s*((?:19|20)\d{2})(?![\d-])"
)
# A number as a claim writes it, thousands grouped or not: "13,000", "9 000", "0.80", "80%".
WRITTEN_RAW = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[,\u00a0\u202f ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\.?\d)"
    r"(?:\s*(?:%|percent\b))?"
)
# The words between a number and a quote that stands for it: "oil capacity 9,000 Sm3/d
# ("Oljekapasitet 9000 Sm3 /d")", "a maximum water cut of 0.80: “Maksimum vannkutt 0,80”".
BEFORE_QUOTE = re.compile(
    r"\s*(?:%|percent|[kM]?S?m3(?:\s*/\s*d(?:ay)?)?|m\s?MD|m|bar|months?|years?)?\s*"
    r"(?:\(|\[|:|,|—|–)?\s*(?:as\s+|quoted\s+as\s+)?$",
    re.IGNORECASE,
)


def _quote_neighbours(label: str, text: str) -> list[str]:
    """A number written right before a quote, or in the translation right after it, that
    the quote does not hold: "oil capacity 16,000 Sm3/d ("Oljekapasitet 9000 Sm3 /d")",
    "“Maks vanninnhold 80 %” [Maximum water content 150%]". The quote is checked against
    the page elsewhere; the number beside it must be the quote's own."""
    problems = []
    text = _translations_outside(canonical(text))
    for m in QUOTED_CLAIM.finditer(text):
        quote = m.group(1)
        held = {
            v
            for european in (False, True)
            for v, _, _ in _numbers_in_text(_blank(quote, REFERENCE), european)
        }
        if not held:
            continue
        around: list[str] = []
        before = list(WRITTEN_RAW.finditer(_blank(text[: m.start()], WELL, DATE, REFERENCE)))
        if before and BEFORE_QUOTE.fullmatch(text[before[-1].end() : m.start()]):
            around.append(before[-1].group(0))
        if m.group(2):
            translation = _blank(m.group(2), WELL, DATE, REFERENCE)
            around += [n.group(0) for n in WRITTEN_RAW.finditer(translation)]
        for written in around:
            readings = _numbers_in_text(written)
            if not readings:
                continue
            value, decimals, _ = readings[0]
            percent = written.rstrip().endswith(("%", "percent"))
            tolerance = 0.5 * 10**-decimals + 1e-9
            if value <= 10 and value == int(value) and not percent:
                continue
            if not any(
                abs(value - q) <= tolerance
                or (percent and abs(value / 100 - q) <= tolerance / 100)
                or abs(value - q * 100) <= tolerance
                for q in held
            ):
                problems.append(
                    f"{label}: '{written.strip()}' is written beside the quote “{quote}”, "
                    f"which says {', '.join(f'{q:g}' for q in sorted(held))}. Write the "
                    "quote's own number."
                )
    return problems


# "the choke stayed unchanged", "not a choke cut": the choke said not to have moved.
CHOKE_STILL = re.compile(
    r"\bchoke\b[^.;]{0,40}?\b(?:(?:was|remained|stayed|held)\s+)?(?:unchanged|constant|steady|"
    # ("cut at the same time" is when, not how much.)
    r"flat|(?<!at )the same(?!\s+(?:time|day|week|month|moment|date|period))|"
    r"did not (?:change|move)|didn't (?:change|move)|never (?:changed|moved))\b"
    r"|\b(?:not|no|without)\s+(?:a\s+|any\s+)?choke\s+(?:cut|change|move|back)",
    re.IGNORECASE,
)


def _choke_still(label: str, text: str, cited: list) -> list[str]:
    """A choke said not to have moved at a change the cited events record it moving at
    ("while the choke stayed unchanged" around 2010-05-02, when it went 83% -> 58%)."""
    problems = []
    for sentence in FULL_SENTENCE.split(text):
        if not CHOKE_STILL.search(sentence):
            continue
        days = [date.fromisoformat(d) for d in re.findall(r"\d{4}-\d{2}-\d{2}", sentence)]
        for e in cited:
            if e.tool != "detect_events" or not isinstance(e.result, list):
                continue
            for event in e.result:
                details = event.get("details") or {}
                if event.get("kind") != "regime_change" or not details.get("choke_moved"):
                    continue
                start = date.fromisoformat(str(event["start"])[:10])
                if any(abs((d - start).days) <= 14 for d in days):
                    before = details.get("choke_before_pct") or 0
                    after = details.get("choke_after_pct") or 0
                    problems.append(
                        f"{label} says the choke did not move, but {e.id} records it going "
                        f"from {before:.0f}% to {after:.0f}% across the {event['start']} "
                        "change. Say what the choke did."
                    )
    return list(dict.fromkeys(problems))


# What a forecast or a trend says about days not yet seen: a forecast's volumes, rates and
# ranges (not its fit's parameters, which describe the past), and the WOR trend carried to
# each reference water cut.
PROJECTED_LABELS = re.compile(
    r"(?:^|\|)(?:to_limits|cumulative_oil_sm3|calibrated_band_sm3|p50_by_model_sm3|"
    r"monthly_rate_sm3_per_producing_day)\|(?!coverage|median_abs_error|past_forecasts)"
)
REMAINING_OIL = r"\b(?:remain\w*|left)\b[^.;]{0,30}\boil\b|\boil\b[^.;]{0,30}\b(?:remain\w*|left)\b"


def _projections(finding: Finding, log: EvidenceLog, leaves_of) -> str:
    """The forecasts and extrapolations a finding's declared numbers come from, named: a
    forecast_production result, or values of the waterflood trend carried past the data."""
    named = []
    # "zero remaining oil to each reference water cut": the oil left, said in words.
    if re.search(REMAINING_OIL, finding.claim, re.I):
        named += [
            f"{e.id}'s trend carried past the data"
            for e in (log.get(i) for i in finding.evidence_ids)
            if e is not None and e.tool == "waterflood"
        ]
    for d in finding.numbers or []:
        item = log.get(d.evidence_id)
        if item is None or not d.value:
            continue
        held = [
            lf
            for lf in leaves_of(item)
            if lf.value is not None
            and any(
                abs(lf.value - x) <= 1e-9 * max(1, abs(x))
                for x in (d.value, d.value * 100, d.value / 100)
            )
        ]
        if held and all(PROJECTED_LABELS.search(lf.labels + "|") for lf in held):
            what = (
                "forecast" if item.tool == "forecast_production" else "trend carried past the data"
            )
            named.append(f"'{d.text}' is {item.id}'s {what}")
    return "; ".join(dict.fromkeys(named))


def _month_spans(label: str, text: str) -> list[str]:
    """A month range and a count of months in one sentence that disagree: "over 2011-01 to
    2012-12 ... across 36 months" (the range is 24), "87 months of 2011-2013" (at most 36)."""
    problems = []
    for sentence in FULL_SENTENCE.split(normalize_dates(text)):
        ranges, counts = MONTH_RANGE.findall(sentence), MONTH_COUNT.findall(sentence)
        years = YEAR_RANGE.findall(sentence)
        if not ranges and len(years) == 1 and len(counts) == 1:
            # A range of whole years holds at most its months (a count may be fewer: months
            # with data).
            y1, y2 = int(years[0][0]), int(years[0][1])
            count, unit = int(counts[0][0]), counts[0][1].lower()
            months = count * 12 if unit.startswith("year") else count
            if y2 > y1 and months > (y2 - y1 + 1) * 12:
                problems.append(
                    f"{label}: {y1} to {y2} is at most {(y2 - y1 + 1) * 12} months, but the "
                    f"sentence says {count} {unit}."
                )
            continue
        if len(ranges) != 1 or len(counts) != 1:
            continue
        y1, m1, y2, m2 = (int(v) for v in ranges[0])
        span = (y2 - y1) * 12 + (m2 - m1) + 1
        count, unit = int(counts[0][0]), counts[0][1].lower()
        months = count * 12 if unit.startswith("year") else count
        if abs(span - months) > (6 if unit.startswith("year") else 1):
            problems.append(
                f"{label}: {y1}-{m1:02d} to {y2}-{m2:02d} is {span} months, but the sentence "
                f"says {count} {unit}."
            )
    return problems


# The plan as the subject and a well as what it forecast: "the plan forecast that well F-12
# would ...", "the PUD expected F-12 to ...". Not a well in another part of the sentence
# ("the PUD predicted this: '...' - F-12's breakthrough is consistent with it").
PLAN_ABOUT_WELL = re.compile(
    rf"\b(?:plan|PUD|report)\w*\b[^,;.:–—]{{0,60}}?\b(?:forecast|expect|predict|assum|said|"
    rf"state|planned|project|anticipat)\w*\s+(?:that\s+)?(?:for\s+)?(?:the\s+)?(?:well\s+)?"
    rf"(?P<well>{WELL.pattern})",
    re.IGNORECASE,
)
PLAN_YEAR = re.compile(
    r"\b(?P<a>(?:19|20)\d{2})(?:['’]s)?\s+(?:development\s+)?(?:plan|PUD)\b"
    r"|\b(?:plan|PUD)\s+(?:of|from|dated)\s+(?P<b>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


# Words that tell the reader a figure reading was not confirmed by the measurement of it.
DISPUTE = re.compile(
    r"\bSAM\b|measur\w*|disagree\w*|differ\w*|does not match|not confirmed|unconfirmed",
    re.IGNORECASE,
)
FIGURE_WORDS = re.compile(
    r"\b(?:figure|fig\.|figur|chart|curve|plot\w*|drawn|graph|picture|visual\w*)\b", re.I
)
LENGTH_UNITS = {"years": r"(?:years?|yrs?|år)", "months": r"(?:months?|måneder|mnd)"}


def _disputed(
    label: str, text: str, numbers: list[DeclaredNumber] | None, looks: list[Evidence]
) -> list[str]:
    """A number the model read off a figure that SAM 3.1's measurement of the same figure
    disagrees with, written without saying so: the reader would take one model's estimate
    for the page's. Declared from the page reading, or (a small length needs no declaring)
    written as that value and unit in a sentence about the figure."""
    problems = []
    for item in looks:
        result = item.result if isinstance(item.result, dict) else {}
        for check in result.get("cross_check") or []:
            value = check.get("muse")
            if check.get("agree") is not False or not isinstance(value, int | float):
                continue
            declared = [
                n for n in numbers or [] if n.evidence_id == item.id and abs(n.value - value) < 1e-9
            ]
            units = LENGTH_UNITS.get(str(check.get("unit", "")).lower().rstrip("s") + "s")
            written = (
                [
                    m
                    for sentence in FULL_SENTENCE.split(text)
                    if FIGURE_WORDS.search(sentence)
                    for m in re.finditer(
                        rf"(?<![\d.,]){re.escape(f'{value:g}')}(?:\.0)?(?:\s*(?:-|to)\s*[\d.]+)?\s*"
                        rf"{units}\b",
                        sentence,
                        re.I,
                    )
                    if not DISPUTE.search(sentence)
                ]
                if units
                else []
            )
            shown = [
                n.text
                for n in declared
                if not any(DISPUTE.search(s) for s in FULL_SENTENCE.split(text) if n.text in s)
            ] + [m.group(0) for m in written]
            if shown:
                unit = check.get("sam_unit") or check.get("unit") or ""
                # The hint gives SAM's own value, which the evidence holds, and an example
                # with no semicolon: the model copies both.
                problems.append(
                    f"{label}: '{shown[0]}' is {item.id}'s reading of the figure "
                    f"({check.get('label')}), and SAM 3.1's measurement of the same figure "
                    f"disagrees with it (SAM: {check.get('sam')} {unit}). Say so in the same "
                    'sentence ("the model read about 2 years, but SAM\'s measurement differs"), '
                    "or leave the reading out."
                )
    return problems


def _report_claims(label: str, text: str, cited: list[Evidence]) -> list[str]:
    """What a sentence gives the plan that its reports do not hold: a well the cited report
    results never name ("the plan forecast that F-12 would ...": the plan is field-level),
    or a year that is not the plan's ("the 2007 plan" for the February 2005 PUD)."""
    reports = [e for e in cited if e.tool in REPORT_TOOLS]
    if not reports:
        return []
    problems = []
    said = json.dumps([e.result for e in reports], ensure_ascii=False)
    named = {_well_key(m.group(0)) for m in WELL.finditer(said)}
    for m in PLAN_ABOUT_WELL.finditer(text):
        well = m.group("well")
        if _well_key(well) not in named:
            problems.append(
                f"{label} gives the report a statement about {well}, but "
                f"{', '.join(e.id for e in reports)} never name {well}: the report speaks for "
                "the field."
            )
    docs = {str(d) for e in reports for d in (e.arguments.get("doc_id"), _get(e.result, "doc_id"))}
    docs |= {m for m in re.findall(r'"doc_id":\s*"([\w-]+)"', said)}
    years = {r.doc_date[:4] for r in REPORTS if r.doc_id in docs} or {
        r.doc_date[:4] for r in REPORTS
    }
    for m in PLAN_YEAR.finditer(text):
        year = m.group("a") or m.group("b")
        if year not in years:
            problems.append(
                f"{label} dates the plan to {year}, but the report it cites is from "
                f"{', '.join(sorted(years))}."
            )
    return problems


def _get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _detail_attribution(
    label: str, detail: str, cited: list[Evidence], leaves_of, free: set[float]
) -> list[str]:
    """The detail's numbers checked like a claim's for well, quantity, unit and direction:
    "water_cut 0.9846" when 0.9846 is uptime, F-14's 2 175 written as F-12's. Each number
    passes if any cited result holding it agrees."""
    text = _normalize(detail)
    blanked = _blank(
        QUOTED.sub(lambda q: " " * len(q.group(0)), text), WELL, DATE, CALENDAR_WINDOW, REFERENCE
    )
    for match in WRITTEN_NUMBER.finditer(blanked):
        written = match.group(0).strip()
        readings = _readings(written)
        if not readings or all(_free(*n, _Pool(cited), free) for n in _numbers_in_text(written)):
            continue
        verdicts = [
            _attribution(text, blanked, written, readings, item, leaves_of(item), own_part=True)
            for item in cited
            if any(_reads_as((n, 2 * u, sc), _Pool([item]).numbers) for n, u, sc in readings)
        ]
        if verdicts and all(verdicts):
            return [
                f"{label}: in its detail, '{written}' {verdicts[0]}. Write what the evidence says."
            ]
    return []


def _inherited(
    item: Evidence, leaves: list[_Leaf], earlier: list[Evidence], leaves_of
) -> list[_Leaf]:
    """A calculation's results as what they were computed from: each result takes the well
    its inputs share, the label they share ("water_share", "oil_rate"), and their records
    (`origin`), so "1888/4207" from F-12's change stays F-12's, and "0.011*100" stays the
    share below the depth of its row. Without that, citing a calculation launders a number
    past the well, quantity and depth checks."""
    result = item.result if isinstance(item.result, dict) else {}
    rows = result.get("results") if isinstance(result.get("results"), list) else None

    def expression_of(lf: _Leaf) -> str | None:
        if rows is not None and lf.record is not None and lf.record < len(rows):
            return str((rows[lf.record] or {}).get("expression") or "")
        return str(result.get("expression") or "") or None

    out = []
    for lf in leaves:
        expression = expression_of(lf) if lf.value is not None else None
        if not expression or not lf.labels.endswith("value"):
            out.append(lf)
            continue
        holders: list[tuple[Evidence, _Leaf]] = []
        for value, decimals, _ in _numbers_in_text(expression):
            if _is_constant(value) or (decimals <= 0 and value <= SMALL_INTEGER):
                continue
            tolerance = 0.5 * 10 ** -max(decimals, 0) + 1e-12
            holders += [
                (e, h)
                for e in earlier
                for h in leaves_of(e)
                if h.value is not None and abs(abs(h.value) - value) <= tolerance
            ]
        if not holders:
            out.append(lf)
            continue
        wells = {h.well for _, h in holders}
        own = {"|".join(h.labels.split("|")[:2]) for _, h in holders}
        out.append(
            replace(
                lf,
                well=next(iter(wells)) if len(wells) == 1 else lf.well,
                labels=f"{lf.labels}|{next(iter(own))}" if len(own) == 1 else lf.labels,
                origin=tuple(
                    dict.fromkeys((e.id, h.record) for e, h in holders if h.record is not None)
                ),
            )
        )
    return out


DEPTH = re.compile(
    r"(?<![\d.,])(\d{1,2}[,\u00a0\u202f ]?\d{3}(?:\.\d+)?)\s*(?:m|metres?|meters?)\b"
)
YEARISH = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?|(?:19|20)\d{2})\b")


def _depth_ties(label: str, text: str, declared, log: EvidenceLog, leaves_of) -> list[str]:
    """A share from a production log is the share below one depth in one survey: written
    next to another depth ("93% of the water" at 3,160 m when 0.934 is below 3,165 m), or in
    a sentence about another survey's year, it is read off the wrong row. Calculated shares
    keep their rows (`_Leaf.origin`)."""
    problems = []
    text = _normalize(text)
    blanked = _blank(text, WELL, DATE, CALENDAR_WINDOW, REFERENCE)

    def rows_of(eid: str, record: int) -> tuple[set[float], set[str], bool]:
        item = log.get(eid)
        if item is None:
            return set(), set(), False
        same = [lf for lf in leaves_of(item) if lf.record == record]
        # A log row's depth, or a blank-pipe gap's ends (at_blank_pipe: the shares below
        # its top and its bottom).
        depths = {
            lf.value
            for lf in same
            if lf.value is not None and ("depth_m" in lf.labels or "blank_pipe_m" in lf.labels)
        }
        days = {d for lf in same for d in lf.record_days}
        share = any("share" in lf.labels for lf in same)
        return depths, days, share

    for d in declared:
        item = log.get(d.evidence_id)
        spans = _standalone(blanked, d.text)
        if item is None or not spans:
            continue
        scales = [d.value, d.value * 100, d.value / 100]
        # A share as written ("93%", "0.934"), held as a share or calculated from shares.
        share_text = d.text.rstrip().endswith(("%", "percent")) or abs(d.value) <= 1
        held = [
            lf
            for lf in leaves_of(item)
            if lf.value is not None
            and ("share" in lf.labels or (lf.origin and share_text))
            and any(abs(abs(lf.value) - abs(x)) <= 1e-9 * max(1.0, abs(x)) for x in scales)
        ]
        depths: set[float] = set()
        days: set[str] = set()
        for lf in held:
            for eid, record in ((item.id, lf.record), *lf.origin):
                if record is None:
                    continue
                found, when, share = rows_of(eid, record)
                if share and found:
                    depths |= found
                    days |= when
        if not depths:
            continue
        start, end = spans[0]
        bounds = [0, *(m.end() for m in re.finditer(r"[.;](?=\s)", text)), len(text)]
        part_start = max(b for b in bounds if b <= start)
        part_end = min((b for b in bounds if b > start), default=len(text))
        # A depth in a clause about another plug is that plug's: "the deep plug would cut
        # 33% ..., while a plug at 3150 m would cut 97%" (not "..., while losing 1%").
        turns = [
            m.start()
            for m in re.finditer(
                r",?\s+\b(?:while|whereas|but|and|compared with)\s+(?:a|an|the|one)\s+"
                r"(?:[\w-]+\s+)?(?:plug|depth|log|survey|gap)\b",
                text,
            )
            if part_start <= m.start() < part_end
        ]
        clause_start = max([part_start, *(t for t in turns if t < start)])
        clause_end = min([part_end, *(t for t in turns if t > end)])
        # "cutting water cut from about 97% to about 31%": a water cut, not a share of the
        # flow below a depth.
        if re.search(
            r"\bwater[\s_-]?cuts?\b(?:[^\d.;,%]{0,25}|\s+(?:\w+\s+){0,2}from\s+"
            r"(?:about\s+|around\s+|~\s*)?[\d.,]+\s*%?\s+to\s+(?:about\s+|around\s+|~\s*)?)$",
            text[clause_start:start],
            re.I,
        ):
            continue
        before = [m for m in DEPTH.finditer(text, clause_start, start)]
        after = [m for m in DEPTH.finditer(text, end, clause_end)]
        # "1% entered below 3,180 m": a depth the number points to right after it is its own.
        pointed = after and re.fullmatch(
            r"[^,;\d]{0,30}\b(?:below|above|under|at|from)\s+(?:about\s+|around\s+|~\s*)?",
            text[end : after[0].start()],
            re.I,
        )
        written = after[0] if pointed else before[-1] if before else after[0] if after else None
        if written is not None:
            depth = float(re.sub(r"[^\d.]", "", written.group(1)))
            # A range ("in 3147 to 3161 m", "3,147-3,161 m") names both its ends.
            ends = [depth]
            if low := re.search(
                r"(\d{1,2}[,\u00a0\u202f ]?\d{3}(?:\.\d+)?)\s*(?:m\s*)?(?:to|-|–)\s*$",
                text[max(0, written.start() - 20) : written.start()],
            ):
                ends.append(float(re.sub(r"[^\d.]", "", low.group(1))))
            # Blank pipe has no inflow: the share below any depth in a gap is the share below
            # either end of it ("a plug at 3150 m" with the 3,147-3,161 m gap's 81%).
            ends += [
                end_
                for e in log
                if e.tool == "production_log" and isinstance(e.result, dict)
                for gap in e.result.get("blank_pipe_m") or []
                if len(gap) == 2 and gap[0] - 0.5 <= depth <= gap[1] + 0.5
                for end_ in gap
            ]
            if not any(abs(e - x) <= 1.5 for e in ends for x in depths):
                shown = ", ".join(f"{x:g}" for x in sorted(depths))
                problems.append(
                    f"{label}: '{d.text}' is the share below {shown} m in {item.id}, not below "
                    f"{depth:g} m. Give each share the depth of its own row."
                )
                continue
        # The survey by its year, where the text names it as the survey's: "In 2009 ...
        # 17%", "by September 2010 ...", "the 2010 log". A plug's year ("the 2014 plug ...
        # 97% of the water measured in 2010") is not the survey's.
        part = text[part_start:part_end]
        named = re.match(r"\s*(?:in|by|during|as of)\s+(?:\w+\s+)?((?:19|20)\d{2})\b", part, re.I)
        named = named or re.search(
            r"\b((?:19|20)\d{2})(?:-\d{2})?\s+(?:production\s+)?(?:log|survey|PLT)\b|"
            r"\b(?:log|survey|PLT|measured)\s+(?:of|from|in)\s+(?:\w+\s+)?((?:19|20)\d{2})\b",
            part,
            re.I,
        )
        years = [y for y in (named.groups() if named else ()) if y]
        if years and days and not any(day.startswith(years[-1]) for day in days):
            problems.append(
                f"{label}: '{d.text}' is from the survey of {', '.join(sorted(days))} in "
                f"{item.id}, not from {years[-1]}. Give each share its own survey."
            )
    return list(dict.fromkeys(problems))


SLOW_DECLINE = re.compile(
    r"\b(?P<word>slow(?:ly)?|gentl[ey]|mild(?:ly)?|modest(?:ly)?|moderate(?:ly)?|gradual(?:ly)?|"
    r"persist\w*|sustain\w*|held up|lasting|not (?:a |just a )?transient)\b",
    re.IGNORECASE,
)
FAST_DECLINE = re.compile(
    r"\b(?P<word>steep(?:ly)?|rapid(?:ly)?|fast|sharp(?:ly)?)\s+decline", re.I
)


def _decline_words(label: str, text: str, cited: list[Evidence]) -> list[str]:
    """Words about a decline against the decline rate the cited fit gives: "a moderate
    decline" or "the gain persisted" for a well falling 0.96 a year (62% a year), "a steep
    decline" for one falling 0.05 a year."""
    rates = [
        float(r)
        for e in cited
        if e.tool in ("fit_decline", "forecast_production")
        for r in [
            _get(_get(e.result, "best"), "di_per_year")
            or _get(_get(e.result, "fit"), "di_per_year")
            or _get(e.result, "di_per_year")
        ]
        if isinstance(r, int | float)
    ]
    if len(rates) != 1:
        return []
    rate = rates[0]
    slow = SLOW_DECLINE.search(text)
    if slow and rate > 0.5 and re.search(r"declin|gain|rate|oil", text, re.I):
        return [
            f"{label} calls the decline '{slow['word']}', but the cited fit falls {rate:.2f} a "
            f"year ({1 - math.exp(-rate):.0%} a year). Describe it by its rate."
        ]
    fast = FAST_DECLINE.search(text)
    if fast and rate < 0.1:
        return [
            f"{label} calls the decline '{fast['word']}', but the cited fit falls only "
            f"{rate:.2f} a year. Describe it by its rate."
        ]
    return []


WINDOW_WORD = re.compile(
    r"\b(?P<word>before|until|up to|after|since|from)\s+"
    r"(?P<when>\d{4}-\d{2}(?:-\d{2})?|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _fit_windows(label: str, text: str, declared, log: EvidenceLog) -> list[str]:
    """A fit's decline rate given to a period its window is not in: "the decline before July
    2010 fits 0.45 a year" when 0.4461 is the fit from 2010-07-18."""
    text = _normalize(text)
    problems = []
    for d in declared:
        item = log.get(d.evidence_id)
        if item is None or item.tool not in ("fit_decline", "forecast_production"):
            continue
        best = _get(item.result, "best") or _get(item.result, "fit") or {}
        rate = _get(best, "di_per_year")
        if not isinstance(rate, int | float) or abs(abs(rate) - abs(d.value)) > 0.005 * max(
            1, abs(rate)
        ):
            continue
        spans = _standalone(_blank(text, WELL, DATE, CALENDAR_WINDOW), d.text)
        if not spans:
            continue
        clause_start = max([0, *(m.end() for m in re.finditer(r"[.;](?=\s)", text[: spans[0][0]]))])
        start, end = str(_get(best, "window_start") or ""), str(_get(best, "window_end") or "")
        for m in WINDOW_WORD.finditer(text, clause_start, spans[0][0]):
            when, word = m["when"], m["word"].lower()
            # "after/since/from" date a fit only right after its name ("the decline after
            # July 2010"); "forecast from 2011-01-01" is the forecast's period.
            if word in ("after", "since", "from") and not re.search(
                r"\b(?:declin\w*|fit\w*|window|trend)\s+(?:\w+\s+){0,2}$",
                text[clause_start : m.start()],
            ):
                continue
            early = word in ("before", "until", "up to") and start and start[: len(when)] >= when
            late = word in ("after", "since", "from") and end and end[: len(when)] < when
            if early or late:
                problems.append(
                    f"{label}: '{d.text}' is the fit from {start} to {end} in {item.id}, not one "
                    f"{word} {when}."
                )
    return list(dict.fromkeys(problems))


def _hyphenated(submission: Submission, log: EvidenceLog) -> Submission:
    """Well names written without their hyphen ("Well F14") as the evidence writes them
    (F-14), so a value given to them is judged for that well."""
    known = {str(e.arguments["well"]) for e in log if isinstance(e.arguments.get("well"), str)}
    known |= {m.group(0) for e in log for m in WELL.finditer(json.dumps(e.result, default=str))}
    forms = {}
    for well in known:
        base = re.fullmatch(r"(?:15/9-)?([A-Z]{1,3})-(\d+)(?:\s?([A-Z]))?", well)
        if base:
            letters, digits, side = base.groups()
            tail = rf"\s?{side}\b" if side else r"\b(?!\s?[A-Z]\b)"
            forms[re.compile(rf"\b{letters}{digits}{tail}")] = well
    if not forms:
        return submission

    def fix(text: str | None) -> str | None:
        for pattern, well in forms.items():
            text = pattern.sub(well, text) if text else text
        return text

    return submission.model_copy(
        update={
            "summary": fix(submission.summary),
            "findings": [
                f.model_copy(update={"claim": fix(f.claim), "detail": fix(f.detail)})
                for f in submission.findings
            ],
        }
    )


def _canonical_submission(submission: Submission) -> Submission:
    """Every text the checker reads, in plain typography (positions unchanged)."""

    def declared(numbers: list[DeclaredNumber] | None) -> list[DeclaredNumber] | None:
        if numbers is None:
            return None
        return [d.model_copy(update={"text": canonical(d.text)}) for d in numbers]

    return submission.model_copy(
        update={
            "summary": canonical(submission.summary),
            "summary_numbers": declared(submission.summary_numbers),
            "findings": [
                f.model_copy(
                    update={
                        "claim": canonical(f.claim),
                        "detail": canonical(f.detail) if f.detail else f.detail,
                        "numbers": declared(f.numbers),
                    }
                )
                for f in submission.findings
            ],
        }
    )


def check_submission(submission: Submission, log: EvidenceLog, question: str = "") -> list[str]:
    """Return problems as sentences for the model; an empty list means accepted."""
    submission = _hyphenated(_canonical_submission(submission), log)
    question = canonical(question)
    free = _question_numbers(question)
    free_dates = REPORT_DATES | set(DATE.findall(_normalize(question)))
    # "Over 2011 to 2013" bounds a window: its years' first and last days need no source.
    # They still belong to a well where the evidence dates them (F-12's 2011-01 is no
    # F-14 month), so they are not added to the dates no well owns below.
    bounds = {
        f"{y}-{md}"
        for y in re.findall(r"\b((?:19|20)\d{2})\b", question)
        for md in ("01-01", "12-31", "01", "12")
    }
    calcs = _calculations(log)
    sound = [e for e in log if not (e.id in calcs and calcs[e.id].unsupported)]
    walked: dict[str, list[_Leaf]] = {}
    order = {e.id: n for n, e in enumerate(log)}

    def leaves_of(item: Evidence) -> list[_Leaf]:
        if item.id not in walked:
            leaves = _leaves(item)
            if item.tool == "calculate":
                earlier = [e for e in log if order[e.id] < order.get(item.id, len(order))]
                leaves = _inherited(item, leaves, earlier, leaves_of)
            walked[item.id] = leaves
        return walked[item.id]

    days = sorted({lf.day for e in log for lf in leaves_of(e) if lf.day and len(lf.day) == 10})
    unowned = free_dates | ({days[0], days[-1]} if days else set())
    free_dates |= bounds
    failures = _gauge_failures(log, leaves_of)
    dead = _dead_gauges(log, leaves_of)

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
        elif misevent := _date_events(_normalize(finding.claim), cited, leaves_of) + (
            _gauge_dated_first(_normalize(finding.claim), cited, leaves_of)
        ):
            problems.append(f"Finding {n}: {'; '.join(misevent)}. Give each event its own date.")
        elif misperiod := _periods(_normalize(finding.claim), cited, leaves_of):
            problems.append(f"Finding {n}: {'; '.join(misperiod)}. Give each event its own date.")
        missing_quotes = _missing_quotes(text, pool)
        if missing_quotes:
            listed = "; ".join(f"'{q[:80]}'" for q in missing_quotes)
            problems.append(
                f"Finding {n}: quoted text not found in {ids}: {listed}. Quote the evidence "
                "word for word (the original quote or its statement_en) or drop the quotes."
            )
        # A cause inside a quote is the report's claim, which the finding only reports.
        cause = CAUSAL.search(_own_words(text, pool)) or CONSEQUENCE.search(
            _own_words(finding.claim, pool)
        )
        if finding.confidence == "fact" and cause:
            problems.append(
                f"Finding {n} states a cause ('{cause.group(0)}') but is marked fact. The data "
                "shows timing and size, not causes: mark it inference."
            )
        own = _own_words(_normalize(finding.claim), pool)
        if finding.numbers is not None:
            problems += _depth_ties(f"Finding {n}", finding.claim, finding.numbers, log, leaves_of)
            problems += _fit_windows(f"Finding {n}", finding.claim, finding.numbers, log)
        problems += _decline_words(f"Finding {n}", own, cited)
        about = {_well_key(str(e.arguments["well"])) for e in cited if e.arguments.get("well")}
        problems += _pressure_misread(f"Finding {n}", own, failures, about)
        problems += _within_limits(f"Finding {n}", own, cited)
        problems += _month_spans(f"Finding {n}", own)
        problems += _quote_neighbours(f"Finding {n}", finding.claim)
        problems += _choke_still(f"Finding {n}", own, cited)
        problems += _report_claims(f"Finding {n}", own, cited)
        looks = [e for e in cited if e.tool == "look_at_page"]
        problems += _disputed(f"Finding {n}", finding.claim, finding.numbers, looks)
        problems += _disputed(f"Finding {n} (its detail)", finding.detail or "", None, looks)
        problems += _trends(f"Finding {n}", own, cited, leaves_of)
        problems += _model_names(f"Finding {n}", own, cited)
        problems += _gauge_claims(f"Finding {n}", own, dead)
        problems += _durations(
            f"Finding {n}",
            _blank(own, WELL, DATE, CALENDAR_WINDOW, REFERENCE),
            cited,
            leaves_of,
            free,
        )
        if finding.detail and finding.numbers is not None:
            problems += _detail_attribution(f"Finding {n}", finding.detail, cited, leaves_of, free)
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
        elif finding.confidence == "fact" and (projected := _projections(finding, log, leaves_of)):
            problems.append(
                f"Finding {n} is marked fact but rests on a projection ({projected}): what "
                "has not happened yet is a model's estimate. Mark it inference."
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
        problems += _depth_ties(
            "Summary", submission.summary, submission.summary_numbers, log, leaves_of
        )
        problems += _fit_windows("Summary", submission.summary, submission.summary_numbers, log)
        numbers = []
    missing = (
        numbers + dates + [f"'{q[:80]}'" for q in _missing_quotes(submission.summary, everything)]
    )
    if missing:
        problems.append(f"Summary: {', '.join(missing)} not found in any evidence.")
    own = _normalize(_own_words(submission.summary, everything))  # "May 2010" is a date
    problems += _pressure_misread("Summary", own, failures)
    problems += _within_limits("Summary", own, list(log))
    # The headline gets every check a finding gets, against all the run's evidence.
    misdated = [
        d
        for part in FULL_SENTENCE.split(own)
        for d in _date_attribution(part, sound, leaves_of, unowned)
    ]
    if misdated:
        problems.append(
            f"Summary gives the well its sentence names date(s) {', '.join(misdated)}. Name "
            "the well the date belongs to."
        )
    # Against all the run's evidence a short shut-in no detector listed would read as
    # another's: the summary's event dates are judged for the gauge failure only.
    if misevent := _date_events(own, sound, leaves_of, ("gauge failure",)):
        problems.append(f"Summary: {'; '.join(misevent)}. Give each event its own date.")
    if misperiod := _periods(own, sound, leaves_of, ("gauge failure",)):
        problems.append(f"Summary: {'; '.join(misperiod)}. Give each event its own date.")
    problems += _month_spans("Summary", own)
    problems += _quote_neighbours("Summary", submission.summary)
    problems += _report_claims("Summary", own, sound)
    problems += _disputed(
        "Summary",
        submission.summary,
        submission.summary_numbers,
        [e for e in log if e.tool == "look_at_page"],
    )
    problems += _trends("Summary", own, sound, leaves_of)
    problems += _gauge_claims("Summary", own, dead)
    problems += _durations(
        "Summary", _blank(own, WELL, DATE, CALENDAR_WINDOW, REFERENCE), sound, leaves_of, free
    )
    for sentence in SENTENCE.split(own):
        causes = list(CAUSAL.finditer(sentence))
        # One hedged explanation covers the sentence's claim: "water-driven decline, most
        # likely because water is taking the well's capacity".
        explained = any(HEDGE.search(" ".join(sentence[: c.start()].split()[-3:])) for c in causes)
        cause = next((c for c in causes if not _hedged(sentence, c.start(), c.end())), None)
        if cause and not explained:
            problems.append(
                f"Summary states a cause as certain ('{cause.group(0)}'). The data shows "
                "timing and size, not causes: hedge it in the same sentence, e.g. 'most "
                "likely because' or 'consistent with'."
            )
            break
    return problems
