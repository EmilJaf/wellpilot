"""Draw frontend/public/og.png, the link preview and the README's first picture.

The time machine's featured run: data cut at 2010-12-31, the agent dates F-12's shift
itself, fits after it and forecasts 2011; the shaded year is what it could not see. The
numbers come from the run's hindsight on the processed data, drawn as SVG in an HTML card
and captured by headless Chrome at 1200x630.

    cd backend && uv run python ../scripts/og_card.py [run id]
"""

import json
import math
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from wellpilot.data.store import ProductionStore
from wellpilot.hindsight import hindsight

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = ROOT / "frontend/node_modules/@fontsource-variable/archivo/files"

run_id = sys.argv[1] if len(sys.argv) > 1 else "9291bac14418"
saved = json.loads(
    next(
        path
        for folder in ("runs", "retired_runs")
        if (path := ROOT / f"backend/tests/fixtures/{folder}/{run_id}.json").exists()
    ).read_text()
)
cut = date.fromisoformat(saved["as_of"])
processed = ROOT / "data/processed"
h = hindsight(
    {**saved, "verified": True}, ProductionStore(processed), ProductionStore(processed, cut)
)
f = h["forecasts"][0]
agent = f["methods"][0]
fixed = [w["error"] for w in f["fixed_windows"] if w["error"] is not None]
shift = (f.get("window_from_change") or {}).get("date") or agent["window_start"]

# Plot geometry: 2008 to 2012 across, 500 to 6000 Sm3/d up a log scale.
W, H = 600, 336
L, R, T, B = 52, 92, 34, 58
T0, T1 = date(2008, 1, 1).toordinal(), date(2012, 1, 1).toordinal()
LO, HI = math.log10(500), math.log10(6000)


def x(day: str) -> float:
    return L + (W - L - R) * (date.fromisoformat(day[:10]).toordinal() - T0) / (T1 - T0)


def y(rate: float) -> float:
    v = math.log10(min(max(rate, 500), 6000))
    return T + (H - T - B) * (1 - (v - LO) / (HI - LO))


def line(dates: list[str], rates: list[float]) -> str:
    return " ".join(
        f"{x(d):.1f},{y(r):.1f}" for d, r in zip(dates, rates, strict=True) if r and r > 0
    )


actual = line(f["rate"]["date"], f["rate"]["rate"])
fit = line(agent["fitted"]["date"], agent["fitted"]["rate"])
ahead = line(agent["forecast"]["date"], agent["forecast"]["rate"])
end_x, end_y = x(agent["forecast"]["date"][-1]), y(agent["forecast"]["rate"][-1])
hidden = x(f["forecast_from"])
pct = round(agent["error"] * 100)
low, high = round(min(fixed) * 100), round(max(fixed) * 100)

grid = "".join(
    f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
    f'<text x="{L - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v // 1000}k</text>'
    for v in (1000, 2000, 5000)
)
years = "".join(
    f'<text x="{x(f"{yr}-07-01"):.1f}" y="{H - B + 18}" class="tick" '
    f'text-anchor="middle">{yr}</text>'
    for yr in range(2008, 2012)
)
svg = f"""
<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect x="{hidden:.1f}" y="{T - 18}" width="{W - R - hidden:.1f}" height="{H - B - T + 18}"
        class="hidden"/>
  <text x="{hidden + 6:.1f}" y="{T - 5}" class="note">hidden from the agent</text>
  {grid}{years}
  <polyline points="{actual}" class="oil"/>
  <line x1="{x(shift):.1f}" x2="{x(shift):.1f}" y1="{T - 18}" y2="{H - B}" class="shift"/>
  <text x="{x(shift) - 6:.1f}" y="{T - 5}" class="note strong" text-anchor="end">
    the shift it found, {shift}</text>
  <polyline points="{fit}" class="agent"/>
  <polyline points="{ahead}" class="agent"/>
  <text x="{end_x + 6:.1f}" y="{end_y - 2:.1f}" class="label">agent +{pct}%</text>
  <text x="{end_x + 6:.1f}" y="{end_y + 15:.1f}" class="small">fixed windows</text>
  <text x="{end_x + 6:.1f}" y="{end_y + 29:.1f}" class="small">+{low}% to +{high}%</text>
  <line x1="{x(agent["window_start"]):.1f}" x2="{x(agent["window_end"]):.1f}"
        y1="{H - B + 30}" y2="{H - B + 30}" class="window"/>
  <text x="{x(agent["window_end"]) + 6:.1f}" y="{H - B + 34}" class="small">
    the agent's window</text>
</svg>"""

page = f"""<!doctype html><meta charset="utf-8">
<style>
@font-face {{ font-family: Archivo; font-weight: 100 900; font-stretch: 62% 125%;
  src: url("file://{FONT}/archivo-latin-wdth-normal.woff2") format("woff2"),
       url("file://{FONT}/archivo-latin-standard-normal.woff2") format("woff2"); }}
* {{ margin: 0; box-sizing: border-box; }}
body {{ width: 1200px; height: 630px; background: #e6e9eb; color: #1f262d;
  font-family: Archivo, system-ui, sans-serif; display: grid;
  grid-template-columns: 440px 1fr; gap: 40px; padding: 58px 50px 46px 60px; }}
.brand {{ font-size: 34px; font-weight: 800; font-stretch: 112%; letter-spacing: -0.5px; }}
.tag {{ font-size: 16px; color: #4c5661; margin-top: 6px; }}
h1 {{ font-size: 43px; line-height: 1.08; font-weight: 750; margin-top: 54px;
  letter-spacing: -0.6px; }}
.sub {{ font-size: 21px; line-height: 1.4; color: #4c5661; margin-top: 20px; }}
.sub b {{ color: #1f262d; }}
.credit {{ font-size: 13.5px; color: #626b76; position: absolute; left: 60px; bottom: 40px;
  width: 400px; line-height: 1.35; }}
.panel {{ background: #f6f7f7; border: 1px solid #d3d8dc; margin-top: 46px; height: 424px;
  padding: 20px 24px; }}
.title {{ font-size: 15px; color: #4c5661; }} .title b {{ color: #1f262d; }}
svg {{ display: block; margin-top: 18px; overflow: visible; }}
.grid {{ stroke: #d3d8dc; }} .tick {{ font-size: 12.5px; fill: #626b76; }}
.hidden {{ fill: rgba(31, 38, 45, 0.07); }}
.oil {{ fill: none; stroke: #008300; stroke-width: 1.6; }}
.agent {{ fill: none; stroke: #1f262d; stroke-width: 2.4; }}
.shift {{ stroke: #1f262d; stroke-width: 1.2; stroke-dasharray: 3 3; }}
.window {{ stroke: #1f262d; stroke-width: 4; }}
.note {{ font-size: 12.5px; fill: #4c5661; }} .strong {{ fill: #1f262d; font-weight: 650; }}
.label {{ font-size: 15px; font-weight: 750; fill: #1f262d; }}
.small {{ font-size: 12.5px; fill: #4c5661; }}
.key {{ font-size: 13px; color: #4c5661; margin-top: 8px; display: flex; gap: 22px; }}
.key i {{ display: inline-block; width: 18px; height: 0; border-top: 2px solid; margin-right: 7px;
  vertical-align: middle; }}
</style>
<div>
  <div class="brand">WellPilot</div>
  <div class="tag">Well analysis on Meta's open-weight Muse Spark Open</div>
  <h1>It found where the decline changed, and forecast from there.</h1>
  <p class="sub">Given data only up to {cut.year}, its forecast for {cut.year + 1} came within
    <b>{pct}%</b>. Fixed windows said +{low}% to +{high}%, and which was right showed only
    afterwards. Every number checked against the data.</p>
  <p class="credit">Volve field data, made open by Equinor and the former Volve licence
    partners.</p>
</div>
<div class="panel">
  <div class="title"><b>Well {f["well"]}, oil rate (Sm³/d).</b> Shaded: the year the agent could
    not see.</div>
  {svg}
  <div class="key"><span><i style="border-color:#008300"></i>actual oil, from the data</span>
    <span><i style="border-color:#1f262d;border-top-width:2.5px"></i>agent's fit and
    forecast</span></div>
</div>"""

with tempfile.TemporaryDirectory() as tmp:
    html = Path(tmp) / "card.html"
    html.write_text(page)
    out = ROOT / "frontend/public/og.png"
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--force-device-scale-factor=1",
            "--window-size=1200,630",
            f"--screenshot={out}",
            html.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
print(f"wrote {out} ({run_id}: agent {pct:+d}%, fixed windows {low:+d}% to {high:+d}%)")
