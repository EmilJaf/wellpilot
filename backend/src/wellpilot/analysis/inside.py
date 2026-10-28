"""Inside a well: where its oil and water came in, and what was done about it.

One picture of a well along its depth, from three sources that were never drawn together:
- its production logs (wellpilot.ingest.production_logs), each survey's flowing passes as the
  share of each phase coming from below each depth, with the water cut at the top, so water
  and oil can be stacked as parts of the whole flow;
- its perforated zones, as the 2009 plots draw them, and the zones opened later;
- the work done on it from the daily reports (plugs and perforations, by depth), with what
  the well produced before and after each job.
No model involved: the logs read off pictures say so, with their checks.
"""

import json
import re

import pandas as pd

from wellpilot.agent.tools import LogArgs, OperationsArgs, production_log, well_operations
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production_logs import PLOTS_FILE, inflow

STEP_M = 2.0
# "Installed 5,72" EVO bridge plug at 3210 m", 'Installed 7" EV0 plug with mid element at
# 3150 m', "Installed plug @ 3592 mMD"
PLUG = re.compile(r"plug[^.]*?(?:at|@)\s*(\d{4}(?:[.,]\d+)?)\s*m", re.IGNORECASE)
WINDOW_DAYS = 60  # production compared before and after a job


def _production(store: ProductionStore, well: str, start, end) -> dict | None:
    rows = store.query(
        "SELECT median(water_cut) AS water_cut, median(oil_sm3 * 24 / on_stream_hrs) AS oil, "
        "count(*) AS days, min(date) AS first, max(date) AS last FROM production_daily "
        "WHERE well = ? AND date BETWEEN ? AND ? AND on_stream_hrs >= 12 AND oil_sm3 > 0",
        [well, start, end],
    ).rows.iloc[0]
    if not rows["days"]:
        return None
    return {
        "water_cut": round(float(rows["water_cut"]), 3),
        "oil_sm3_per_day": round(float(rows["oil"])),
        "days": int(rows["days"]),
        "from": rows["first"].date().isoformat(),
        "to": rows["last"].date().isoformat(),
    }


def _back_on(store: ProductionStore, well: str, after: pd.Timestamp) -> pd.Timestamp | None:
    """The first day the well flowed again after a job (it often stays shut in for weeks)."""
    first = store.query(
        "SELECT min(date) AS day FROM production_daily "
        "WHERE well = ? AND date > ? AND on_stream_hrs >= 12 AND oil_sm3 > 0",
        [well, after],
    ).rows.iloc[0]["day"]
    return None if pd.isna(first) else pd.Timestamp(first)


def inside_well(store: ProductionStore, well: str) -> dict:
    logs = production_log(store, LogArgs(well=well))
    path = store.processed_dir / PLOTS_FILE
    plots = [
        {
            "id": p["id"],
            "survey": p["survey"],
            "pass": p["pass"],
            "page": p["source"]["page"],
            "read_by": {phase: r["read_by"] for phase, r in p["phases"].items()},
            "prompts": {phase: r["prompt"] for phase, r in p["phases"].items() if "prompt" in r},
        }
        for p in (json.loads(path.read_text()) if path.exists() else [])
        if p["well"] == well
    ]
    surveys = []
    rows = store.query(
        "SELECT survey, pass, depth_m, oil, water FROM production_logs "
        "WHERE well = ? AND pass != 'shut-in' ORDER BY survey, pass, depth_m",
        [well],
    ).rows
    for survey in logs["surveys"]:
        passes = []
        for p in survey["passes"]:
            log = rows[(rows["survey"] == survey["survey"]) & (rows["pass"] == p["pass"])]
            shares = inflow(log, STEP_M)["from_below"]
            columns = {c: [r[i] for r in shares["rows"]] for i, c in enumerate(shares["columns"])}
            passes.append(
                {
                    "pass": p["pass"],
                    "water_cut_at_top": p.get("water_cut_at_top", 0.0),
                    "depth_m": columns["depth_m"],
                    "oil_share": columns["oil_share"],
                    "water_share": columns.get("water_share"),
                }
            )
        surveys.append(
            {
                k: survey[k]
                for k in (
                    "survey",
                    "date",
                    "read_from",
                    "rate_unit",
                    "check",
                    "production_30_days_before",
                )
                if k in survey
            }
            | {"passes": passes, "plots": [p for p in plots if p["survey"] == survey["survey"]]}
        )
    jobs = []
    logged = {pd.Timestamp(s["date"]) for s in surveys}
    for job in well_operations(store, OperationsArgs(well=well))["jobs"]:
        if job["kind"] in ("drilling", "abandonment"):
            continue
        text = " ".join(d["summary"] for d in job["summaries"])
        plugs = [float(p.replace(",", ".")) for p in PLUG.findall(text)]
        perforations = [
            [float(a), float(b)]
            for interval in job["perforations"]
            for a, b in re.findall(r"(\d{4}(?:\.\d+)?)-(\d{4}(?:\.\d+)?)", interval)
        ]
        start, end = pd.Timestamp(job["start"]), pd.Timestamp(job["end"])
        survey = next((d for d in logged if start <= d <= end), None)
        if not (plugs or perforations or survey):  # nothing done along the well's depth
            continue
        back = _back_on(store, well, end)
        jobs.append(
            {
                "start": job["start"],
                "end": job["end"],
                "kind": job["kind"],
                "production_log": survey is not None,
                "plugs_m": plugs,
                "perforations_m": perforations,
                "summaries": job["summaries"],
                # The same windows as the agent's production_log (plugs_set) and the reveal,
                # so the tab, the answer and the README give one number per job.
                "before": _production(store, well, start - pd.Timedelta(days=WINDOW_DAYS), start),
                "after": back
                and _production(store, well, back, back + pd.Timedelta(days=WINDOW_DAYS)),
            }
        )
    zones = next(
        (
            s["check"]["perforated_zones_m"]
            for s in logs["surveys"]
            if "perforated_zones_m" in s.get("check", {})
        ),
        [],
    )
    return {"well": well, "surveys": surveys, "zones_m": zones, "jobs": jobs}
