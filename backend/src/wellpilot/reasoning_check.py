"""The time machine's reasoning, read for the hidden period.

A run asked as of a past date sees no data after it. Muse Spark Open returns its
reasoning; this reads it for two things the hidden period holds and the run should not:
- dates after the as-of date, outside a sentence about forecasting (the year being
  forecast is the question, not a leak) and not in the run's own tool results (the
  forecast window the tool returned);
- numbers that match what the data has only after the date (each producer's oil for each
  later year, within 1%) and that the run's own tool results do not hold.

What it finds is listed for a person to read. Finding nothing means only that the
reasoning holds no date or value from the hidden period: a model could use what it
remembers without writing it down (memory_probe measures what it remembers).
"""

import re

import pandas as pd

from wellpilot.agent.checker import DATE, _numbers_in_text, _Pool, normalize_dates
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.data.store import ProductionStore

FORECAST = re.compile(
    r"forecast|predict|project|next|ahead|horizon|expect|outlook|estimate|will\b|would\b",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
MATCH = 0.01  # a number within 1% of a hidden value matches it
SMALLEST = 10_000  # smaller numbers match by chance too often


def hidden_values(full: ProductionStore, cut: pd.Timestamp) -> list[tuple[float, str]]:
    """Each producer's oil for each calendar year after the date, as the data has it."""
    rows = full.query(
        "SELECT well, year(date) AS year, sum(oil_sm3) AS oil FROM production_daily "
        "WHERE date > ? GROUP BY well, year(date) HAVING sum(oil_sm3) >= ? ORDER BY well, year",
        [cut, SMALLEST],
    ).rows
    return [(float(r.oil), f"{r.well}'s oil in {r.year}") for r in rows.itertuples()]


def reasoning_check(run: dict, full: ProductionStore) -> dict:
    cut = pd.Timestamp(run["as_of"])
    steps = [e for e in run.get("events", []) if e.get("type") == "step"]
    texts = [(s.get("index"), s.get("reasoning") or "") for s in steps]
    if not any(text for _, text in texts):
        return {"available": False, "note": "the model's provider returned no reasoning to read"}
    # What the run's own tool results hold, the model saw rather than remembered: the
    # forecast window's dates, values in its tables.
    pool = _Pool(EvidenceLog.restore(run["evidence"].values()))
    seen, seen_dates = pool.numbers, pool.dates
    hidden = hidden_values(full, cut)
    last_year = int(full.query("SELECT max(year(date)) AS y FROM production_daily").rows.iloc[0, 0])
    dates, values, forecasts = [], [], 0
    for step, text in texts:
        for sentence in SENTENCE.split(text):
            plain = normalize_dates(sentence)
            later = [d for d in DATE.findall(plain) if d > cut.date().isoformat()[: len(d)]]
            # A bare number is read as a year only inside the hidden period: "2082 -> 1142"
            # is a rate.
            later += [
                y for y in YEAR.findall(DATE.sub(" ", plain)) if cut.year < int(y) <= last_year
            ]
            if later and FORECAST.search(sentence):
                forecasts += len(later)
            elif later := [d for d in later if not any(sd.startswith(d) for sd in seen_dates)]:
                dates.append({"step": step, "dates": later, "sentence": sentence.strip()[:240]})
            for number, _, _ in _numbers_in_text(DATE.sub(" ", plain)):
                if number < SMALLEST or any(abs(number - v) <= MATCH * v for v in seen):
                    continue
                for value, what in hidden:
                    if abs(number - value) <= MATCH * value:
                        values.append(
                            {
                                "step": step,
                                "number": number,
                                "matches": f"{what}: {value:,.0f} Sm3",
                                "sentence": sentence.strip()[:240],
                            }
                        )
    return {
        "available": True,
        "steps": len(texts),
        "characters": sum(len(t) for _, t in texts),
        "later_dates": dates,
        "hidden_values": values,
        "forecast_mentions": forecasts,
    }
