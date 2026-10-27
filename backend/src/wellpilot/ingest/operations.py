"""Daily drilling and well-operation reports (Volve's `Well_technical_data`, HTML version)
into two tables, with no model involved.

Each report covers 24 hours of one wellbore: a summary, the plan for the next day and an
operations table (start, end, depth, "main -- sub" activity, state, remark), plus the
perforations made. Besides drilling, they log the wireline work done on producing wells:
plugs set to shut off water, new zones perforated, production logs. Those explain changes
in the production data that no decline curve can (F-12's 2014 jump).

- `well_operations.parquet`: one row per report day (the period's first day, when the work
  was done), with the well as the production data names it.
- `well_operation_steps.parquet`: one row per line of the operations table.

Download (needs the Databricks profile in CLAUDE.local.md): the reports named
`15_9_<well>_<yyyy>_<mm>_<dd>.html` for the wells in the production data, into
`data/raw/ddr/`. Run: `uv run python -m wellpilot.ingest.operations`.
"""

import html
import json
import re
from pathlib import Path

import pandas as pd

from wellpilot.config import get_settings

OPERATIONS_FILE = "well_operations.parquet"
STEPS_FILE = "well_operation_steps.parquet"

# Wellbore as the reports name it -> the well as the production data names it. F-11's
# producing wellbore is F-11 B; its pilot and sidetracks (F-11, F-11 A, F-11 T2) are not.
WELLS = {
    "15/9-F-12": "F-12",
    "15/9-F-14": "F-14",
    "15/9-F-11 B": "F-11",
    "15/9-F-15 D": "F-15 D",
    "15/9-F-1 C": "F-1 C",
    "15/9-F-4": "F-4",
    "15/9-F-5": "F-5",
}

_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)


def _text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _after_header(page: str, header: str) -> str:
    """The text block under a TextHeader paragraph ("Summary of activities (24 Hours)")."""
    m = re.search(
        rf'class="TextHeader">\s*{re.escape(header)}.*?class="TextData">(.*?)</p>', page, re.S
    )
    return _text(m.group(1)) if m else ""


def _info(page: str, label: str) -> str:
    m = re.search(
        rf'class="ColHead">\s*{re.escape(label)}\s*</td>\s*<td[^>]*>(.*?)</td>', page, re.S
    )
    return _text(m.group(1)) if m else ""


def _table(page: str, table_id: str) -> list[list[str]]:
    m = re.search(rf'<table id="{table_id}".*?<tbody>(.*?)</tbody>', page, re.S)
    if not m:
        return []
    return [[_text(c) for c in _CELL.findall(row)] for row in _ROW.findall(m.group(1))]


def _number(value: str) -> float | None:
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return None
    return None if number <= -999 else number  # -999.99 means not recorded


def parse_report(path: Path) -> tuple[dict, list[dict]] | None:
    """One report: the day's row and its operation lines, or None for another wellbore."""
    # Empty cells are written self-closing (<td class="ColValue"/>): open and close them so
    # a cell never swallows the next one.
    page = re.sub(
        r"<(td|th)([^>]*?)/>", r"<\1\2></\1>", path.read_text(encoding="utf-8", errors="replace")
    )
    wellbore = _text(
        re.search(r"<b>Wellbore: </b>(.*?)</p>", page, re.S).group(1)
        if "<b>Wellbore: </b>" in page
        else ""
    )
    well = WELLS.get(wellbore)
    if well is None:
        return None
    period = re.search(r"<b>Period: </b>\s*(\d{4}-\d\d-\d\d) [\d:]+ - (\d{4}-\d\d-\d\d)", page)
    if not period:
        return None
    day, report_day = period.group(1), period.group(2)
    steps = []
    for cells in _table(page, "operationsInfoTable"):
        if len(cells) < 6:
            continue
        start, end, depth, activity, state, remark = cells[:6]
        main, _, sub = activity.partition(" -- ")
        steps.append(
            {
                "well": well,
                "date": day,
                "start": start,
                "end": end,
                "depth_mmd": _number(depth),
                "activity": main.strip(),
                "sub_activity": sub.strip(),
                "state": state,
                "remark": remark,
            }
        )
    perforations = [
        f"{top}-{bottom} mMD"
        for row in _table(page, "perforationInfoTable")
        if len(row) >= 4
        for top, bottom in [(row[2], row[3])]
        if _number(top) is not None and _number(bottom) is not None
    ]
    activities = list(dict.fromkeys(s["activity"] for s in steps if s["activity"]))
    return (
        {
            "well": well,
            "wellbore": wellbore,
            "date": day,
            "report_date": report_day,
            "contractor": _info(page, "Drilling contractor:"),
            "summary": _after_header(page, "Summary of activities (24 Hours)"),
            "planned": _after_header(page, "Summary of planned activities (24 Hours)"),
            "activities": ", ".join(activities),
            "perforations": "; ".join(perforations),
            "source": path.name,
        },
        steps,
    )


def ingest(raw_dir: Path, out_dir: Path) -> dict:
    days, steps = [], []
    for path in sorted(raw_dir.glob("15_9_*.html")):
        parsed = parse_report(path)
        if parsed is None:
            continue
        day, lines = parsed
        days.append(day)
        steps.extend(lines)
    frame = pd.DataFrame(days)
    lines = pd.DataFrame(steps)
    for table in (frame, lines):
        table["date"] = pd.to_datetime(table["date"])
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame = frame.sort_values(["well", "date"]).drop_duplicates(["well", "date"], keep="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / OPERATIONS_FILE, index=False)
    lines.to_parquet(out_dir / STEPS_FILE, index=False)
    return {
        "days": len(frame),
        "steps": len(lines),
        "wells": {w: int(n) for w, n in frame["well"].value_counts().sort_index().items()},
    }


def main() -> None:
    settings = get_settings()
    print(json.dumps(ingest(settings.raw_dir / "ddr", settings.processed_dir), indent=2))


if __name__ == "__main__":
    main()
