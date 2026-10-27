"""Production logs (PLT): where along the well the oil and the water come in.

A production logging tool is run down a flowing well and measures, depth by depth, the
rate passing it; the rate grows going up past each zone that produces, so the log shows
which zones give oil and which give water. Volve has F-12's survey of September 2010 as
interpreted LAS files (`Well_logs_pr_WELL/15_9-F-12/10.PRODUCTION LOGS/PLT Sept 2010/`):
a shut-in pass and two flowing passes, the plot's "low" and "medium" downhole rates. The
daily reports give the rest: shut-in passes on 2010-09-21, flowing passes on 2010-09-22 at
about 1500 and 3000 Sm3/d of liquid at surface (5050 Sm3/d was tried, but the tool could not
go down against it), and a bridge plug set at 3210 m the next day.

The files give rates without units; what the log says is in the shares (how much of the
oil and of the water enters below a depth), which need none. No model involved.

    uv run python -m wellpilot.ingest.production_logs
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wellpilot.config import get_settings

LOGS_FILE = "production_logs.parquet"


@dataclass(frozen=True)
class Survey:
    well: str
    survey: str
    date: str  # the day it was logged, from the daily reports
    directory: str  # under data/raw
    passes: dict[str, str]  # pass name -> LAS file
    surface_sm3_per_day: dict[str, int]  # flowing pass -> liquid rate at surface, daily report
    plot: str  # the interpreted plot, a picture (no text layer)
    source: str


SURVEYS = [
    Survey(
        well="F-12",
        survey="PLT 2010-09",
        date="2010-09-22",
        directory="plt/2010",
        passes={
            "shut-in": "F12_PLT_302S.LAS",
            "low rate": "F12_PLT_302A.LAS",
            "medium rate": "F12_PLT_302B.LAS",
        },
        surface_sm3_per_day={"low rate": 1500, "medium rate": 3000},
        plot="plot_volve-f12-plt-0910.pdf",
        source="Well_logs_pr_WELL/15_9-F-12/10.PRODUCTION LOGS/PLT Sept 2010",
    )
]
SURVEYS_BY_WELL = {s.well: s for s in SURVEYS}

# LAS mnemonic suffix -> column
CURVES = {
    "QOIL": "oil",
    "QWAT": "water",
    "QGAS": "gas",
    "QTOT": "total",
    "YW": "water_holdup",
    "WPRE": "pressure_bar",
    "WTEP": "temperature_c",
}


def read_las(path: Path) -> pd.DataFrame:
    """A LAS 2.0 file (one line per depth): its curves by suffix, the null value as NaN."""
    lines = path.read_text(errors="replace").splitlines()
    null = -999.25
    for line in lines:
        if line.strip().startswith("NULL."):
            null = float(line.split(".", 1)[1].split(":")[0])
    start = next(i for i, line in enumerate(lines) if line.startswith("~A"))
    names = lines[start].split()[1:]
    rows = [[float(v) for v in line.split()] for line in lines[start + 1 :] if line.strip()]
    frame = pd.DataFrame(rows, columns=names).replace(null, np.nan)
    frame.columns = ["depth_m" if c == "DEPTH" else c.split("-")[-1] for c in frame.columns]
    return frame.rename(columns=CURVES)


def ingest(raw_dir: Path, out_dir: Path) -> dict:
    frames = []
    for survey in SURVEYS:
        for name, file in survey.passes.items():
            frame = read_las(raw_dir / survey.directory / file)
            frame.insert(0, "pass", name)
            frame.insert(0, "date", pd.Timestamp(survey.date))
            frame.insert(0, "survey", survey.survey)
            frame.insert(0, "well", survey.well)
            frames.append(frame)
    logs = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(out_dir / LOGS_FILE, index=False)
    return {"rows": len(logs), "surveys": [s.survey for s in SURVEYS]}


REFERENCE_M = 30.0  # the top of the log, above every zone: the well's whole flow
SMOOTH_M = 3.0  # a running median over this many metres takes out the tool's noise


def shares(log: pd.DataFrame) -> pd.DataFrame:
    """A flowing pass as shares: at each logged depth, the part of the pass's oil and of its
    water that comes from below it (the flow passing the depth over the flow at the top)."""
    flowing = log.dropna(subset=["oil", "water"]).sort_values("depth_m").reset_index(drop=True)
    depth = flowing["depth_m"]
    top = flowing[depth <= depth.iloc[0] + REFERENCE_M]
    points = max(1, round(SMOOTH_M / float(depth.diff().median())))
    out = flowing[["depth_m"]].copy()
    for phase in ("oil", "water"):
        smooth = flowing[phase].rolling(points, center=True, min_periods=1).median()
        out[phase] = (smooth / top[phase].median()).clip(0, 1)
    out.attrs["water_cut_at_top"] = float(
        top["water"].median() / (top["water"].median() + top["oil"].median())
    )
    return out


def inflow(log: pd.DataFrame, step_m: float = 5.0) -> dict:
    """Where each phase comes in (above the deepest depth with 95% of it from below, below the
    shallowest with 5%, half of it below half_below_m) and the shares every step_m."""
    s = shares(log)
    depths = np.arange(np.ceil(s["depth_m"].min() / step_m) * step_m, s["depth_m"].max(), step_m)
    index = [int(np.abs(s["depth_m"].to_numpy() - d).argmin()) for d in depths]

    def band(share: pd.Series) -> dict:
        d = s["depth_m"]
        return {
            "comes_in_from_m": round(float(d[share >= 0.95].max()), 1),
            "to_m": round(float(d[share >= 0.05].max()), 1),
            "half_below_m": round(float(d[share >= 0.5].max()), 1),
        }

    return {
        "top_depth_m": round(float(s["depth_m"].iloc[0]), 1),
        "water_cut_at_top": round(s.attrs["water_cut_at_top"], 3),
        "oil_comes_in": band(s["oil"]),
        "water_comes_in": band(s["water"]),
        "from_below": {
            "columns": ["depth_m", "oil_share", "water_share"],
            "rows": [
                [
                    round(float(d), 1),
                    round(float(s["oil"].iloc[i]), 3),
                    round(float(s["water"].iloc[i]), 3),
                ]
                for d, i in zip(depths, index, strict=True)
            ],
        },
    }


def inflow_below(log: pd.DataFrame, depth_m: float) -> dict | None:
    """The share of a flowing pass's oil and water entering below one depth, read at the
    logged depth nearest to it; None when the depth is outside the log."""
    s = shares(log)
    d = s["depth_m"].to_numpy()
    if not d[0] <= depth_m <= d[-1]:
        return None
    at = s.iloc[int(np.abs(d - depth_m).argmin())]
    return {"oil_share": round(float(at["oil"]), 3), "water_share": round(float(at["water"]), 3)}


def main() -> None:
    settings = get_settings()
    print(json.dumps(ingest(settings.raw_dir, settings.processed_dir), indent=1))


if __name__ == "__main__":
    main()
