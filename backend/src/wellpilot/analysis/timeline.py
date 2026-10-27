"""The field's history as the screen would have told it: the field screen replayed at every
month end, each time with the data up to that day only, so an alarm appears when the data
first showed it, not with hindsight. With each producer's monthly oil rate and the work
done on its wells (from the daily reports), on one time axis.

No model is involved. Replaying about 100 month ends takes some 30 s, so the result is built
once and kept beside the processed data:
    uv run python -m wellpilot.analysis.timeline
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from wellpilot.agent.tools import OperationsArgs, well_operations
from wellpilot.analysis.decline import producing_day_rate
from wellpilot.analysis.screen import LEVELS, screen
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore

TIMELINE_FILE = "field_timeline.json"


def _intervals(flags: list[tuple[str, dict]], months: list[str]) -> list[dict]:
    """(month end, issue) pairs of one well and kind into stretches of consecutive months:
    when it was first flagged, the day the detector dates it from, when it cleared, its
    worst level and the text it was first flagged with."""
    by_month = {m: i for m, i in flags}
    out: list[dict] = []
    current: dict | None = None
    for n, month in enumerate(months):
        issue = by_month.get(month)
        if issue is None:
            # One month without it between two with it (a shut-in month, say) does not
            # clear it: the same stretch goes on.
            if current and not (n + 1 < len(months) and months[n + 1] in by_month):
                current["cleared"] = month
                current = None
            continue
        if current is None:
            current = {
                "first_flagged": month,
                "since": issue.get("since"),
                "cleared": None,
                "level": issue["level"],
                "text": issue["text"],
            }
            out.append(current)
        elif LEVELS[issue["level"]] > LEVELS[current["level"]]:
            current["level"] = issue["level"]
    return out


def build(processed_dir: Path, months: list[pd.Timestamp] | None = None) -> dict:
    full = ProductionStore(processed_dir)
    days = full.query(
        "SELECT min(date) FILTER (WHERE oil_sm3 > 0) AS first, "
        "max(date) FILTER (WHERE oil_sm3 > 0) AS last FROM production_daily"
    ).rows.iloc[0]
    if months is None:
        months = list(pd.date_range(days["first"], days["last"], freq="ME"))
    ends = [m.date().isoformat() for m in months]
    flags: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # curve fits on short windows
        for month, end in zip(months, ends, strict=True):
            for row in screen(ProductionStore(processed_dir, as_of=month.date()))["wells"]:
                for issue in row["issues"]:
                    flags.setdefault((row["well"], issue["kind"]), []).append((end, issue))

    producers = full.query(
        "SELECT well, min(date) FILTER (WHERE oil_sm3 > 0) AS first_oil, sum(oil_sm3) AS oil "
        "FROM production_daily GROUP BY well HAVING sum(oil_sm3) > 0 ORDER BY first_oil, well"
    ).rows
    wells = []
    for w in producers.itertuples():
        daily = full.well_daily(w.well)
        rate = producing_day_rate(daily[daily["flow_kind"] == "production"])
        monthly = rate.groupby(rate.index.to_period("M")).mean()
        work = []
        if "well_operations" in full.tables:
            jobs = well_operations(full, OperationsArgs(well=w.well))["jobs"]
            work = [
                {
                    "start": j["start"],
                    "end": j["end"],
                    "perforations": j["perforations"],
                    "summary": " ".join(s["summary"] for s in j["summaries"])[:600],
                }
                for j in jobs
                if j["kind"] == "workover"
            ]
        wells.append(
            {
                "well": w.well,
                "first_oil": str(pd.Timestamp(w.first_oil).date()),
                "oil_sm3": round(float(w.oil)),
                "oil_rate": {
                    "month": [str(p) for p in monthly.index],
                    "rate": [round(float(v), 1) for v in monthly.to_numpy()],
                },
                "issues": [
                    {"kind": kind, **stretch}
                    for (well, kind), found in sorted(flags.items())
                    if well == w.well
                    for stretch in _intervals(found, ends)
                ],
                "work": work,
            }
        )
    return {
        "months": [ends[0], ends[-1]],
        "replayed": len(ends),
        "how": "The field screen run at every month end with the data up to that day only.",
        "wells": wells,
    }


def main() -> None:
    settings = get_settings()
    out = build(settings.processed_dir)
    (settings.processed_dir / TIMELINE_FILE).write_text(json.dumps(out, indent=1))
    issues = sum(len(w["issues"]) for w in out["wells"])
    work = sum(len(w["work"]) for w in out["wells"])
    print(
        f"{out['replayed']} month ends, {len(out['wells'])} producers, {issues} issues, {work} jobs"
    )


if __name__ == "__main__":
    main()
