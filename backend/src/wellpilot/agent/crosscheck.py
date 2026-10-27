"""A number read off a picture, checked by a second Meta model.

look_at_page asks Muse Spark Open what a page shows; its readings are estimates, and they
carry a warning, so the checker lets an answer state them only as inferences. Where SAM 3.1
has measured the same figure at ingest (the plan's forecast curve, PUD page 23, Figure 4.2,
checked against a colour trace of the same pixels), each reading Muse gives of that curve
is set against SAM's measurement: a rate at a date against SAM's curve that month, the
plateau's rate and length against SAM's plateau. A reading the two agree on is no longer an
estimate of one model; if every reading on the page agrees, the warning goes.

The tolerances come from the figure, not from the readings: a quarter of the space between
two gridlines on each axis, the closest a careful reader reads a chart by eye. Gridlines on
Figure 4.2 are 1000 Sm3/d and one year apart: 250 Sm3/d and 3 months.

LLM looks, SAM measures, deterministic code judges.
"""

import json
import re
from pathlib import Path

import numpy as np

from wellpilot.ingest.figure import CURVE_FILE

# Figures SAM has measured, by report page.
FIGURES = {("volve_pud", 23): CURVE_FILE}
RATE_TOLERANCE = 250.0  # Sm3/d: a quarter of the 1000 Sm3/d between gridlines
MONTHS_TOLERANCE = 3  # a quarter of the year between gridlines
PLATEAU_SHARE = 0.98  # months within 2% of the top are the plateau (as the Plan tab draws)

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
YEAR = re.compile(r"\b(20\d{2})\b")
RATE_UNIT = re.compile(r"m3\s*/\s*d|m³\s*/\s*d|m3/day|m³/day", re.IGNORECASE)
# The model writes units and labels in the report's language too: "måneder", "år", "platå".
MONTHS_UNIT = re.compile(r"^\s*(?:months?|måneder|mnd)\b", re.IGNORECASE)
YEARS_UNIT = re.compile(r"^\s*(?:years?|år)\b", re.IGNORECASE)
PLATEAU = re.compile(r"plateau|plat[åa]|peak|max", re.IGNORECASE)
FIGURE = re.compile(r"^\s*fig", re.IGNORECASE)


def plateau(months: list[str], rates: list[float]) -> dict:
    """The drawn plateau: the months within 2% of the curve's top."""
    top = max(rates)
    flat = [(m, r) for m, r in zip(months, rates, strict=True) if r >= PLATEAU_SHARE * top]
    return {
        "sm3_per_day": round(float(np.median([r for _, r in flat]))),
        "first": flat[0][0],
        "last": flat[-1][0],
        "months": len(flat),
    }


def measured(processed_dir: Path, doc_id: str, page: int) -> dict | None:
    """SAM's measurement of the figure on a page, or None if it measured none there."""
    file = FIGURES.get((doc_id, page))
    if file is None or not (processed_dir / file).exists():
        return None
    figure = json.loads((processed_dir / file).read_text())
    if figure.get("read_by") != "sam-3.1":
        return None
    curve = figure["curve"]
    return {
        "by": "sam-3.1",
        "what": "the plan's forecast oil rate curve, monthly (Sm3/d)",
        "curve": dict(zip(curve["month"], curve["rate"], strict=True)),
        "plateau": plateau(curve["month"], curve["rate"]),
        "colour_check": figure.get("agreement_with_colour"),
    }


def _when(label: str) -> list[str] | None:
    """The months a reading's label points at: 'start of 2010' is 2010-01, 'end of 2010'
    2010-12, 'mid 2010' 2010-07, 'March 2010' 2010-03, '2010' alone every month of it."""
    year = YEAR.search(label)
    if not year:
        return None
    y, text = year.group(1), label.lower()
    named = next((n for word, n in MONTHS.items() if re.search(rf"\b{word}[a-z]*\b", text)), None)
    if named:
        return [f"{y}-{named:02d}"]
    if re.search(r"\b(?:start|beginning|early|jan)", text):
        return [f"{y}-01"]
    if re.search(r"\b(?:end|late)\b", text):
        return [f"{y}-12"]
    if re.search(r"\bmid", text):
        return [f"{y}-07"]
    return [f"{y}-{m:02d}" for m in range(1, 13)]


def cross_check(readings: list[dict], sam: dict) -> list[dict]:
    """Each reading against SAM's measurement: agree, disagree, or nothing to set it
    against (None)."""
    checks = []
    for reading in readings:
        label, unit, value = (
            str(reading.get("label", "")),
            str(reading.get("unit", "")),
            reading.get("value"),
        )
        check = {"label": label, "muse": value, "unit": unit, "sam": None, "agree": None}
        where = str(reading.get("where", ""))
        if not FIGURE.match(where):
            # SAM measured the drawn curve; a value from the page's text or a table is not
            # a reading of it.
            check["note"] = (
                f"read from the {where.split()[0].lower() if where else 'page'}, not the figure"
            )
            checks.append(check)
            continue
        if not isinstance(value, int | float):
            checks.append(check)
            continue
        years = YEARS_UNIT.match(unit)
        if (years or MONTHS_UNIT.match(unit)) and PLATEAU.search(label):
            months = value * 12 if years else value
            sam_months = sam["plateau"]["months"]
            check.update(
                {
                    "sam": sam_months,
                    "sam_unit": "months",
                    "difference": round(abs(months - sam_months), 1),
                    "tolerance": f"{MONTHS_TOLERANCE} months",
                    "agree": abs(months - sam_months) <= MONTHS_TOLERANCE,
                }
            )
        elif RATE_UNIT.search(unit):
            when = _when(label)
            at = [sam["curve"][m] for m in when or [] if m in sam["curve"]]
            if at:
                sam_value = round(float(np.mean(at)))
                where = when[0] if len(when) == 1 else f"{when[0][:4]} (mean of its months)"
            elif PLATEAU.search(label):
                sam_value, where = sam["plateau"]["sm3_per_day"], "the plateau"
            else:
                checks.append(check)
                continue
            check.update(
                {
                    "sam": sam_value,
                    "sam_unit": "Sm3/d",
                    "at": where,
                    "difference": round(abs(value - sam_value)),
                    "tolerance": f"{RATE_TOLERANCE:.0f} Sm3/d",
                    "agree": abs(value - sam_value) <= RATE_TOLERANCE,
                }
            )
        checks.append(check)
    return checks
