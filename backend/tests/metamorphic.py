"""Transforms of real accepted answers for tests/test_checker_metamorphic.py.

Meaning-preserving transforms rewrite an answer the checker accepted the way models vary
their writing (typography, dates in words, "about", the unit inside a declaration, the
thousands separator): the checker must still accept it. Meaning-breaking ones change one
thing the checker claims to judge (a number, a date, a cause in a fact): it must reject.
Each transform returns None when the answer has nothing it applies to.
"""

import copy
import re
from datetime import date, timedelta

from wellpilot.agent.checker import (
    CALENDAR_WINDOW,
    DATE,
    QUOTED,
    REFERENCE,
    WELL,
    _blank,
    _Pool,
    _standalone,
    _translations_outside,
)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
UNIT_AFTER = re.compile(r"\s?(?:Sm3/d|Sm3|months?|days?|years?|%|m\b|MSm3)")


def _texts(sub: dict):
    """(finding or None, field) for every text the checker reads."""
    yield None, "summary"
    for f in sub["findings"]:
        yield f, "claim"
        if f.get("detail"):
            yield f, "detail"


def _outside_quotes(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in QUOTED.finditer(_translations_outside(text))]


def _inside(spans, at: int) -> bool:
    return any(a <= at < b for a, b in spans)


def _map_text(sub: dict, fn) -> dict | None:
    """sub with fn applied to each text outside quotes; None when nothing changed."""
    out = copy.deepcopy(sub)
    changed = False
    for f, field in _texts(out):
        holder = out if f is None else f
        text = holder.get(field) or ""
        new = fn(text)
        if new != text:
            holder[field] = new
            changed = True
    return out if changed else None


def _declared(sub: dict):
    yield sub, sub.get("summary_numbers") or [], "summary"
    for f in sub["findings"]:
        yield f, f.get("numbers") or [], "claim"


# Meaning-preserving


def typography(sub: dict) -> dict | None:
    """Non-breaking hyphens in dates and well names, narrow no-break spaces in grouped
    numbers, a curly apostrophe in possessives."""

    def fn(text: str) -> str:
        text = DATE.sub(lambda m: m.group(0).replace("-", "‑"), text)
        text = WELL.sub(lambda m: m.group(0).replace("-", "‑"), text)
        text = re.sub(r"(?<=\d) (?=\d{3}\b)", " ", text)
        return text.replace("'s ", "’s ")

    out = _map_text(sub, fn)
    if out:
        for _, numbers, _ in _declared(out):
            for d in numbers:
                d["text"] = re.sub(r"(?<=\d) (?=\d{3}\b)", " ", d["text"])
    return out


def dates_in_words(sub: dict) -> dict | None:
    """ISO dates outside quotes written in words: 2010-07-18 as 18 July 2010, 2012-06 as
    June 2012."""

    def word(m: re.Match) -> str:
        parts = m.group(0).split("-")
        if len(parts) == 3:
            return f"{int(parts[2])} {MONTHS[int(parts[1]) - 1]} {parts[0]}"
        return f"{MONTHS[int(parts[1]) - 1]} {parts[0]}"

    def fn(text: str) -> str:
        quoted = _outside_quotes(text)
        return DATE.sub(lambda m: m.group(0) if _inside(quoted, m.start()) else word(m), text)

    return _map_text(sub, fn)


def about(sub: dict) -> dict | None:
    """'about' before each declared number in claims that has no approximation word yet."""
    out = copy.deepcopy(sub)
    changed = False
    for holder, numbers, field in _declared(out):
        text = holder.get(field) or ""
        for d in numbers:
            # Where the checker sees the number: not inside a date, a well name or a quote.
            blanked = _blank(text, DATE, WELL, CALENDAR_WINDOW, REFERENCE)
            at = [s for s, _ in _standalone(blanked, d["text"])]
            if not at or _inside(_outside_quotes(text), at[0]):
                continue
            if re.search(
                r"(?:about|around|roughly|approximately|nearly|~|near|some)\s*$", text[: at[0]]
            ):
                continue
            text = text[: at[0]] + "about " + text[at[0] :]
            changed = True
        holder[field] = text
    return out if changed else None


def unit_declared(sub: dict) -> dict | None:
    """A declared number's unit taken into its declared text: "21" before "months" is
    declared as "21 months"."""
    out = copy.deepcopy(sub)
    changed = False
    for holder, numbers, field in _declared(out):
        text = holder.get(field) or ""
        for d in numbers:
            if re.search(r"[A-Za-z%]", d["text"]):
                continue
            m = re.search(rf"(?<![\w.,]){re.escape(d['text'])}(?![\d.,]\d)", text)
            if not m:
                continue
            unit = UNIT_AFTER.match(text, m.end())
            if unit:
                d["text"] = text[m.start() : unit.end()]
                changed = True
    return out if changed else None


def commas(sub: dict) -> dict | None:
    """Thousands grouped with commas: "4 050" and "4050" written "4,050" (text and
    declaration)."""
    out = copy.deepcopy(sub)
    changed = False
    for holder, numbers, field in _declared(out):
        text = holder.get(field) or ""
        for d in numbers:
            plain = d["text"].replace(" ", "").replace(" ", "").replace(" ", "")
            if not re.fullmatch(r"\d{4,7}", plain) or abs(
                float(plain) - abs(d["value"])
            ) > 0.5 * abs(d["value"]):
                continue
            new = f"{int(plain):,}"
            m = re.search(rf"(?<![\w.,]){re.escape(d['text'])}(?![\d.,]\d)", text)
            if not m or new == d["text"]:
                continue
            text = text[: m.start()] + new + text[m.end() :]
            d["text"] = new
            changed = True
        holder[field] = text
    return out if changed else None


PRESERVING = {
    "typography": typography,
    "dates_in_words": dates_in_words,
    "about": about,
    "unit_declared": unit_declared,
    "commas": commas,
}


# Meaning-breaking: each changes the first finding it can and names it


def _cited_pool(case: dict, finding: dict) -> _Pool:
    from wellpilot.agent.evidence import EvidenceLog

    log = EvidenceLog.restore(case["evidence"])
    return _Pool([e for e in log if e.id in finding.get("evidence_ids", [])])


def changed_number(case: dict) -> tuple[dict, str] | None:
    """One declared number x1.17, written in the same style, where the cited evidence has
    no such value."""
    sub = copy.deepcopy(case["submission"])
    for n, f in enumerate(sub["findings"], start=1):
        pool = _cited_pool(case, f)
        for d in f.get("numbers") or []:
            m = re.fullmatch(r"(\d+)(?:\.(\d+))?(%?)", d["text"].strip())
            if not m or m.group(1) in {"0"} or float(d["text"].rstrip("%")) <= 10:
                continue
            decimals = len(m.group(2) or "")
            new_value = float(d["text"].rstrip("%")) * 1.17
            new_text = f"{new_value:.{decimals}f}{m.group(3)}"
            scale = d["value"] / float(d["text"].rstrip("%"))
            if pool.has_number(new_value * scale, decimals, "") or pool.has_number(
                new_value, decimals, ""
            ):
                continue
            at = re.search(rf"(?<![\w.,]){re.escape(d['text'])}(?![\d.,]\d)", f["claim"])
            if not at:
                continue
            f["claim"] = f["claim"][: at.start()] + new_text + f["claim"][at.end() :]
            d["text"], d["value"] = new_text, new_value * scale
            return sub, f"Finding {n}"
    return None


def shifted_date(case: dict) -> tuple[dict, str] | None:
    """One ISO day date in a claim, outside quotes, moved 3 days, to a day no cited
    evidence has."""
    sub = copy.deepcopy(case["submission"])
    for n, f in enumerate(sub["findings"], start=1):
        pool = _cited_pool(case, f)
        quoted = _outside_quotes(f["claim"])
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", f["claim"]):
            if _inside(quoted, m.start()):
                continue
            new = (date.fromisoformat(m.group(0)) + timedelta(days=3)).isoformat()
            if pool.has_date(new):
                continue
            f["claim"] = f["claim"][: m.start()] + new + f["claim"][m.end() :]
            return sub, f"Finding {n}"
    return None


def cause_in_fact(case: dict) -> tuple[dict, str] | None:
    """A cause added to a fact."""
    sub = copy.deepcopy(case["submission"])
    for n, f in enumerate(sub["findings"], start=1):
        if f.get("confidence") == "fact":
            f["claim"] = (
                f["claim"].rstrip(". ") + ", because the injectors pushed water into the well."
            )
            return sub, f"Finding {n}"
    return None


BREAKING = {
    "changed_number": changed_number,
    "shifted_date": shifted_date,
    "cause_in_fact": cause_in_fact,
}
