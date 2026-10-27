"""A quality gate for the report facts: what must hold before they ship.

The extraction reviews each page as it goes (report_facts.review); this checks the result
as a whole, deterministically and for free, so a worse extraction (another model, another
prompt) is caught before the agent or the demo depends on it:

- the facts the product relies on are there: every row of the plan's production table,
  the four design limits, the key quotes;
- no quote is unverified, and few match only as reordered words;
- almost every exact quote was located on its page image;
- no page was left with problems after the review rounds (the cache records them).

Run: uv run python -m wellpilot.ingest.facts_gate       (exits 1 if the gate fails)
"""

import json
import sys
from pathlib import Path

import pandas as pd

from wellpilot.agent.tools import NoArgs, field_vs_design, plan_vs_actual
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.report_facts import FACTS_FILE, PageExtraction, review
from wellpilot.ingest.reports import PAGES_FILE

PLAN_YEARS = range(2008, 2013)  # the plan's P50 forecast years (PUD p.25, Table 4.5)
DESIGN_LIMITS = {
    "liquid_sm3_per_day": 13000,
    "oil_sm3_per_day": 9000,
    "water_cut": 0.8,
    "water_inj_sm3_per_day": 16000,
}
KEY_QUOTES = (  # quoted by the demo answers; each must stand word for word on its page
    "bryter gjennom i brønnene etter omlag 2 år",
    "Maksimum vannkutt 0,80",
    "Platåraten på 9.000",
)
MAX_TOKENS_SHARE = 0.02  # quotes matching only as reordered words
MIN_LOCATED_SHARE = 0.95  # exact quotes found on the page image


def check(processed_dir: Path) -> list[str]:
    """Every way the facts fall short, as sentences; [] means the gate passes."""
    facts_path = processed_dir / FACTS_FILE
    if not facts_path.exists():
        return [f"no {FACTS_FILE}; run python -m wellpilot.ingest.report_facts"]
    facts = pd.read_parquet(facts_path)
    store = ProductionStore(processed_dir)
    problems = []

    try:
        plan = plan_vs_actual(store, NoArgs())
        cols = plan["years"]["columns"]
        p50 = {r[cols.index("year")]: r[cols.index("plan_p50_msm3")] for r in plan["years"]["rows"]}
        missing = [y for y in PLAN_YEARS if not p50.get(y)]
        if missing:
            problems.append(f"the plan's P50 rows are missing for {missing}")
    except ValueError as e:
        problems.append(f"plan_vs_actual fails: {e}")

    try:
        limits = field_vs_design(store, NoArgs())["design_limits"]
        for key, value in DESIGN_LIMITS.items():
            got = limits.get(key, {}).get("value")
            if got != value:
                problems.append(f"design limit {key} is {got}, expected {value}")
    except ValueError as e:
        problems.append(f"field_vs_design fails: {e}")

    exact = facts[facts["quote_check"] == "exact"]
    for quote in KEY_QUOTES:
        if not exact["quote"].str.contains(quote, regex=False).any():
            problems.append(f"no exact quote holds {quote!r}")

    checks = facts["quote_check"].value_counts()
    if checks.get("unverified", 0):
        problems.append(f"{checks['unverified']} quotes are not on their page")
    if checks.get("tokens", 0) > MAX_TOKENS_SHARE * len(facts):
        problems.append(f"{checks['tokens']} quotes match only as reordered words")

    if "quote_boxes" in exact:
        located = (exact["quote_boxes"] != "[]").mean() if len(exact) else 1.0
        if located < MIN_LOCATED_SHARE:
            problems.append(f"only {located:.0%} of exact quotes were located on their page")

    # Each cached page reviewed again by today's rules; a page that still has no facts
    # after being asked is the model's judgement, not a problem.
    pages = pd.read_parquet(processed_dir / PAGES_FILE).set_index(["doc_id", "page"])
    for path in sorted((processed_dir / "reports").glob("*/facts/p*.json")):
        record = json.loads(path.read_text())
        key = (record["doc_id"], record["page"])
        if key not in pages.index or not pages.loc[key, "has_text"]:
            continue
        extraction = PageExtraction.model_validate(record["extraction"])
        image_page = bool(pages.loc[key].get("image_page", False))
        left = review(extraction, pages.loc[key, "text"], record["page"], image_page)
        left = [p for p in left if "no facts" not in p]
        if left:
            problems.append(f"{record['doc_id']} page {record['page']}: {left[0][:160]}")
    return problems


def main() -> None:
    processed = get_settings().processed_dir
    problems = check(processed)
    facts = pd.read_parquet(processed / FACTS_FILE) if (processed / FACTS_FILE).exists() else []
    print(f"report facts: {len(facts)} in {processed / FACTS_FILE}")
    if problems:
        print("GATE FAILED:")
        for p in problems:
            print(f"- {p}")
        sys.exit(1)
    print("gate passed: plan rows, design limits, key quotes, quote checks, locations, reviews")


if __name__ == "__main__":
    main()
