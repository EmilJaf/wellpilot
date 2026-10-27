"""Production logs kept only as pictures, read by SAM 3.1.

F-12 was logged twice while it produced: in March 2009, a year after first oil, and in
September 2010, after the water had come. The 2010 survey survives as LAS files (digital
curves), and as a plot. The 2009 one survives only in its interpretation report, as plots:
pictures in a PDF. Each plot's rate track shows the flow passing each depth as a filled area;
the area's edge at a depth is the rate there. SAM 3.1 (Meta's segmentation model, over the
Meta Model API) finds the area from a few words ("green area": 2009's oil; "cyan area" and
"black area": 2010's water and the oil drawn beside it) and returns its mask.

Turning pixels into depths and rates takes the plot's own marks, found in the image: the
track's edges, whose values are printed in its header, and two labelled depths (2009: the
underlined 3100 and 3200 m; 2010: the grid's top and bottom, 3100 and 3500 m). The values
are entered by hand, as a person digitising a log would.

Each reading carries its checks:
- the same edge traced by its colour, no model;
- 2010: the LAS files, at the same depths (as shares: the plot and the files use other units);
- 2009: the report's own words ("~83% of oil produced from the upper 2 completions") and
  its rate table: the plot's rate at the top over the surface rate the report simulated is
  the same at all three rates (the oil's volume factor), so the scale holds.

Runs once (seven SAM calls on the Meta Model API, MODEL_API_KEY):
    uv run python -m wellpilot.ingest.log_plots            # SAM, then the checks
    uv run python -m wellpilot.ingest.log_plots --colour   # colour traces only, free
"""

import argparse
import asyncio
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster
from PIL import Image, ImageDraw

from wellpilot.config import get_settings
from wellpilot.ingest.figure import MODEL, parse_masks, segment
from wellpilot.ingest.production_logs import PLOTS_FILE, SURVEYS_BY_WELL, inflow_below, read_las

PLOTS_DIR = "production_log_plots"  # SAM's answers as they came, and the overlays
STEP_M = 1.0  # the profile's depth step


def _green(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 1] > 180) & (rgb[..., 0] < 100) & (rgb[..., 2] < 100)


def _cyan(rgb: np.ndarray) -> np.ndarray:
    """Cyan fill, with the dark blue line drawn along its edge."""
    fill = (rgb[..., 0] < 80) & (rgb[..., 1] > 200) & (rgb[..., 2] > 200)
    edge = (rgb[..., 2] > 150) & (rgb[..., 0] < 120) & (rgb[..., 1] < 120)
    return fill | edge


def _black(rgb: np.ndarray) -> np.ndarray:
    return rgb.sum(axis=-1) < 150


@dataclass(frozen=True)
class Layer:
    phase: str  # oil | water
    prompt: str  # what SAM is asked for
    colour: Callable[[np.ndarray], np.ndarray]
    beside: str | None = None  # drawn to the right of this phase's area (2010's oil)


@dataclass(frozen=True)
class Plot:
    id: str
    well: str
    survey: str
    date: str  # the day the flowing passes were logged
    pass_name: str
    file: str  # under data/raw
    page: int  # 1-based
    track_px: tuple[int, int]  # the rate track's edges near here, snapped to its lines
    rate_axis: tuple[float, float]  # the values at its edges, from the header
    depth_marks: tuple[float, float]  # the two labelled depths found in the image
    depth_style: str  # "labels" (underlined depths) | "grid" (the grid's top and bottom)
    rate_unit: str
    layers: tuple[Layer, ...]
    scale: float = 0.0  # render the page at this scale; 0: read its one raster as stored
    grid_px: tuple[int, int] | None = None  # a track with no fill, where the grid shows whole
    simulated_sm3_per_day: float | None = None  # the report's simulated surface oil rate
    separator_sm3_per_day: float | None = None  # and the oil rate measured at the separator
    source: dict = field(default_factory=dict)


REPORT_2009 = "plt/2009/F-12 PLT Volve March 2009.pdf"
PLOT_2010 = "plt/2010/plot_volve-f12-plt-0910.pdf"
OIL_2009 = Layer("oil", "green area", _green)
WATER_2010 = Layer("water", "cyan area", _cyan)
OIL_2010 = Layer("oil", "black area", _black, beside="water")


def _plot_2009(pass_name: str, page: int, top: float, simulated: float, measured: float) -> Plot:
    return Plot(
        id=f"f12-2009-{pass_name.split()[0]}",
        well="F-12",
        survey="PLT 2009-03",
        date="2009-03-14",  # "logged ... during the time period 14th - 16th March 2009"
        pass_name=pass_name,
        file=REPORT_2009,
        page=page,
        track_px=(523, 708),
        rate_axis=(-500.0 if top > 2000 else -200.0, top),
        depth_marks=(3100.0, 3200.0),
        depth_style="labels",
        rate_unit="m3/d downhole",
        layers=(OIL_2009,),
        simulated_sm3_per_day=simulated,
        separator_sm3_per_day=measured,
        source={
            "document": "Volve 15/9-F-12 Production Logging Interpretation Results, March 2009",
            "page": page,
        },
    )


PLOTS = [
    # Appendices B-D of the 2009 report; the surface oil rates simulated and measured at the
    # separator are its Tables 2 and 1.
    _plot_2009("high rate", 14, 6000.0, 3850.0, 3700.0),
    _plot_2009("medium rate", 15, 4500.0, 2574.0, 2800.0),
    _plot_2009("low rate", 16, 2000.0, 1286.0, 1400.0),
    *[
        Plot(
            id=f"f12-2010-{name.split()[0]}",
            well="F-12",
            survey="PLT 2010-09",
            date="2010-09-22",
            pass_name=name,
            file=PLOT_2010,
            page=1,
            track_px=track,
            rate_axis=(0.0, 5000.0),
            depth_marks=(3100.0, 3500.0),
            depth_style="grid",
            rate_unit="not stated",
            layers=(WATER_2010, OIL_2010),
            scale=5.0,
            grid_px=(1545, 1850),  # the shut-in rate track
            source={"document": "15/9-F-12 PLT & CPI data September 2010", "page": 1},
        )
        for name, track in [("low rate", (1854, 2169)), ("medium rate", (2169, 2485))]
    ],
]
PLOTS_BY_ID = {p.id: p for p in PLOTS}
# The 2009 report's conclusion the reading is checked against (page 4).
UPPER_TWO_QUOTE = "~83% of oil produced from the upper 2 completions, the rest from the lower 3"
UPPER_TWO_SHARE = 0.83


def page_image(raw_dir: Path, plot: Plot) -> Image.Image:
    page = pdfium.PdfDocument(raw_dir / plot.file)[plot.page - 1]
    if plot.scale:
        return page.render(scale=plot.scale).to_pil().convert("RGB")
    images = [o for o in page.get_objects() if o.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE]
    largest = max(images, key=lambda o: o.get_px_size()[0] * o.get_px_size()[1])
    return largest.get_bitmap(render=False).to_pil().convert("RGB")


def _runs(flags: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for i in np.nonzero(flags)[0]:
        if out and i <= out[-1][1] + gap:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return [(a, b) for a, b in out]


@dataclass(frozen=True)
class Calibration:
    left: int  # the track's edge lines, in image columns
    right: int
    rows: tuple[float, float]  # the image rows of the two labelled depths
    body: tuple[int, int]  # the plot's first and last rows of data
    plot: Plot

    def rate(self, x: float) -> float:
        lo, hi = self.plot.rate_axis
        return lo + (hi - lo) * (x - self.left) / (self.right - self.left)

    def depth(self, y: float) -> float:
        (r0, r1), (d0, d1) = self.rows, self.plot.depth_marks
        return d0 + (d1 - d0) * (y - r0) / (r1 - r0)

    def row(self, depth: float) -> float:
        (r0, r1), (d0, d1) = self.rows, self.plot.depth_marks
        return r0 + (r1 - r0) * (depth - d0) / (d1 - d0)

    def zero(self) -> float:
        """The column of a zero rate."""
        lo, hi = self.plot.rate_axis
        return self.left + (self.right - self.left) * (0 - lo) / (hi - lo)


def _snap(lines: np.ndarray, guess: int, reach: int = 8) -> int:
    """The track's edge: the column nearest the guess that is a line most of the way down."""
    near = range(max(0, guess - reach), min(len(lines), guess + reach + 1))
    return max(near, key=lambda x: (lines[x], -abs(x - guess)))


def calibrate(rgb: np.ndarray, plot: Plot) -> Calibration:
    dark = rgb.sum(axis=-1) < 300
    lum = rgb.mean(axis=-1)
    height = rgb.shape[0]
    body = slice(int(height * 0.25), int(height * 0.9))
    lines = (lum[body] < 200).mean(axis=0)
    left, right = (_snap(lines, x) for x in plot.track_px)
    if plot.depth_style == "labels":
        # The depth column, just left of the first track: its underlined depths span it;
        # the plot's header and footer lines span everything, so the first and last go.
        # The depth column, just left of the first track: its labelled depths are underlined
        # across it. Lines across the whole plot are its frame: the header's bottom and the
        # footer's top bound the rows of data.
        frame = [(a + b) / 2 for a, b in _runs((lum[:, 80:880] < 200).mean(axis=1) > 0.9)]
        column = [(a + b) / 2 for a, b in _runs((lum[:, 118:152] < 200).mean(axis=1) > 0.95)]
        found = [r for r in column if min(abs(r - f) for f in frame) > 3]
        if found:
            body = (
                int(max(f for f in frame if f < found[0])) + 2,
                int(min(f for f in frame if f > found[-1])) - 1,
            )
    else:
        # The grid's lines, evenly spaced, where no fill hides them: the longest chain without
        # a gap of more than three steps (the header box above the grid is farther away).
        assert plot.grid_px is not None
        across = dark[:, slice(*plot.grid_px)].mean(axis=1) > 0.9
        rows = [(a + b) / 2 for a, b in _runs(across)]
        step = float(np.median(np.diff(rows)))
        chains: list[list[float]] = [[rows[0]]]
        for r in rows[1:]:
            if r - chains[-1][-1] <= 3.5 * step:
                chains[-1].append(r)
            else:
                chains.append([r])
        grid = max(chains, key=len)
        found, body = [grid[0], grid[-1]], (int(grid[0]) + 2, int(grid[-1]) - 1)
    if len(found) != 2:
        raise ValueError(f"{plot.id}: expected two labelled depths, found {len(found)}")
    return Calibration(left, right, (found[0], found[1]), body, plot)


def edges(hit: np.ndarray, cal: Calibration, under: np.ndarray | None = None) -> dict[int, float]:
    """Per image row of the plot, how wide an area is, in columns, from a grid of the track's
    pixels it covers (rows as the page image's, columns as the track's): an area drawn from
    zero runs from the zero line to its right edge; one drawn beside another (`under`) runs
    from that one's right edge to its own."""
    widths: dict[int, float] = {}
    zero = cal.zero() - cal.left - 1
    for y in range(*cal.body):
        if under is None:
            xs = np.nonzero(hit[y])[0]
            if len(xs):
                widths[y] = float(xs.max() + 1 - zero)
            continue
        xs = np.nonzero(under[y])[0]
        if not len(xs):
            continue
        run, end = 0, hit.shape[1]
        for x in range(xs.max() + 1, hit.shape[1]):
            if hit[y, x]:
                run += 1
            elif run:
                end = x
                break
        # A run to the track's edge is a gridline across it; no run, nothing beside.
        if run == 0 or end < hit.shape[1]:
            widths[y] = float(run)
    return widths


def colour_hits(rgb: np.ndarray, cal: Calibration, layer: Layer) -> np.ndarray:
    """The track's pixels of the layer's colour. No model."""
    return layer.colour(rgb[:, cal.left + 1 : cal.right])


CROP_MARGIN_M = 10.0


def crop(image: Image.Image, cal: Calibration) -> tuple[Image.Image, tuple[int, int]]:
    """The rate track alone, as SAM is shown it: over the rows its first area (the one drawn
    from zero) covers, and a margin, within the plot's rows of data; and its top-left corner
    in the page image."""
    hit = colour_hits(np.asarray(image).astype(int), cal, cal.plot.layers[0])
    rows = np.nonzero(hit[cal.body[0] : cal.body[1]].any(axis=1))[0] + cal.body[0]
    margin = CROP_MARGIN_M * abs(cal.row(1) - cal.row(0))
    top = max(cal.body[0], int(rows.min() - margin)) if len(rows) else cal.body[0]
    bottom = min(cal.body[1], int(rows.max() + margin)) if len(rows) else cal.body[1]
    box = (cal.left + 1, top, cal.right, bottom)
    return image.crop(box), (box[0], box[1])


def mask_hits(
    records: list[SegmentationMaskRecord], shape: tuple[int, int], origin: tuple[int, int]
) -> np.ndarray:
    """SAM's masks as a grid like colour_hits: each mask's raster spans its box in the crop
    SAM saw, whose top-left corner is `origin` in the page image."""
    hit = np.zeros(shape, dtype=bool)
    for record in records:
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
            record.mask.height, record.mask.width
        )
        b = record.bounds
        ys = np.arange(int(np.floor(b.top)), int(np.ceil(b.bottom)))
        xs = np.arange(int(np.floor(b.left)), int(np.ceil(b.right)))
        rows = ((ys + 0.5 - b.top) * record.mask.height / (b.bottom - b.top)).astype(int)
        cols = ((xs + 0.5 - b.left) * record.mask.width / (b.right - b.left)).astype(int)
        inside = raster[np.clip(rows, 0, record.mask.height - 1)][
            :, np.clip(cols, 0, record.mask.width - 1)
        ].astype(bool)
        page_rows = ys + origin[1]
        keep_r = (page_rows >= 0) & (page_rows < shape[0])
        keep_c = (xs >= 0) & (xs < shape[1])  # the crop's columns are the track's
        hit[np.ix_(page_rows[keep_r], xs[keep_c])] |= inside[np.ix_(keep_r, keep_c)]
    return hit


def profile(widths: dict[int, float], cal: Calibration, line_px: float = 0.0) -> dict[float, float]:
    """Rates by depth (median of the image rows in each STEP_M), in the plot's units. A width
    `line_px` is the line drawn along an area's edge, taken off."""
    unit = (cal.plot.rate_axis[1] - cal.plot.rate_axis[0]) / (cal.right - cal.left)
    by_depth: dict[float, list[float]] = {}
    for y, w in widths.items():
        depth = round(cal.depth(y + 0.5) / STEP_M) * STEP_M
        by_depth.setdefault(depth, []).append(max(w - line_px, 0.0) * unit)
    return {d: float(np.median(v)) for d, v in sorted(by_depth.items())}


def _line_px(widths: dict[int, float]) -> float:
    """The width of the line along an area drawn beside another: what the area measures
    where it holds nothing (the median of its narrow rows). A reading that leaves the line
    out (SAM's mask of the area) has rows of 0 there, and nothing to take off."""
    values = np.array(list(widths.values()))
    if not len(values):
        return 0.0
    return float(np.median(values[values <= 3 * np.percentile(values, 5)]))


def shares(rates: dict[float, float], top_m: float = 30.0) -> dict[float, float]:
    """Each depth's rate over the rate at the top of the profile (its first top_m metres): the
    share of the phase entering below that depth."""
    depths = sorted(rates)
    top = [rates[d] for d in depths if d <= depths[0] + top_m]
    reference = float(np.median(top))
    if reference <= 0:
        return {}
    return {d: float(np.clip(rates[d] / reference, 0, 1)) for d in depths}


def _compare(a: dict[float, float], b: dict[float, float]) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"depths": 0}
    gap = np.array([abs(a[d] - b[d]) for d in shared])
    return {
        "depths": len(shared),
        "median_share_gap": round(float(np.median(gap)), 3),
        "p90_share_gap": round(float(np.percentile(gap, 90)), 3),
    }


def zones_2009(rgb: np.ndarray, cal: Calibration) -> list[list[float]]:
    """The perforated zones the 2009 plots draw as red bars left of the depth column."""
    red = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    bars = red[:, 80:96].mean(axis=1) > 0.5
    return [[round(cal.depth(a), 1), round(cal.depth(b + 1), 1)] for a, b in _runs(bars, gap=2)]


def read(
    image: Image.Image,
    plot: Plot,
    sam_outputs: dict[str, str] | None,
    las: pd.DataFrame | None = None,
) -> tuple[dict, Image.Image]:
    """One plot read by colour, and by SAM when its outputs are given (phase -> output text).
    `las` is the same pass from the digital files, when they exist. Returns the result and
    an overlay of the edges read, drawn on the track."""
    rgb = np.asarray(image).astype(int)
    cal = calibrate(rgb, plot)
    track, origin = crop(image, cal)
    result: dict = {
        "id": plot.id,
        "well": plot.well,
        "survey": plot.survey,
        "date": plot.date,
        "pass": plot.pass_name,
        "source": plot.source,
        "rate_unit": plot.rate_unit,
        "separator_oil_sm3_per_day": plot.separator_sm3_per_day,
        "calibration": {
            "track_px": [cal.left, cal.right],
            "rate_axis": list(plot.rate_axis),
            "depth_rows_px": [round(r, 1) for r in cal.rows],
            "depth_marks_m": list(plot.depth_marks),
            "body_rows_px": list(cal.body),
            "crop_origin_px": list(origin),
            "image_height_px": rgb.shape[0],
            "how": "track edges and labelled depths found in the image; values from the "
            "header and the depth labels",
        },
        "phases": {},
    }
    shape = (rgb.shape[0], cal.right - cal.left - 1)
    hits: dict[str, np.ndarray] = {}
    read_edges: dict[str, dict[int, float]] = {}
    for layer in plot.layers:
        under = layer.beside
        colour = colour_hits(rgb, cal, layer)
        by_colour_px = edges(
            colour, cal, colour_hits(rgb, cal, _layer(plot, under)) if under else None
        )
        by_colour = profile(by_colour_px, cal, _line_px(by_colour_px) if under else 0.0)
        phase: dict = {"colour_trace": by_colour}
        if sam_outputs is None:
            phase.update({"read_by": "colour trace (no model)", "rate": by_colour})
            hits[layer.phase], read_edges[layer.phase] = colour, by_colour_px
        else:
            masks, sam, by_sam_px, by_sam = sam_reading(
                sam_outputs[layer.phase], cal, origin, shape, layer, hits.get(under or "")
            )
            phase.update(
                {
                    "read_by": MODEL,
                    "prompt": layer.prompt,
                    "objects_found": len(masks),
                    "confidence": max((m.confidence or 0.0) for m in masks) or None,
                    "rate": by_sam,
                    "agreement_with_colour": _compare(shares(by_sam), shares(by_colour)),
                }
            )
            hits[layer.phase], read_edges[layer.phase] = sam, by_sam_px
        result["phases"][layer.phase] = phase
    # The edges read, in red, on the track as SAM saw it.
    overlay = track.copy()
    draw = ImageDraw.Draw(overlay)
    zero = cal.zero() - origin[0]
    for layer in plot.layers:
        for y, w in read_edges[layer.phase].items():
            base = read_edges[layer.beside].get(y, 0.0) if layer.beside else 0.0
            draw.line(
                [(zero + base + w - 1, y - origin[1]), (zero + base + w + 1, y - origin[1])],
                fill=(220, 0, 0),
            )
    _checks(plot, rgb, cal, result, las)
    return result, overlay


def sam_reading(
    output: str,
    cal: Calibration,
    origin: tuple[int, int],
    shape: tuple[int, int],
    layer: Layer,
    under: np.ndarray | None,
) -> tuple[list[SegmentationMaskRecord], np.ndarray, dict[int, float], dict[float, float]]:
    """One layer read from SAM's answer: its masks, the track pixels they cover, the width
    per image row, and the rates by depth. Needs no image, only the calibration, so a saved
    answer rebuilds its reading anywhere."""
    masks = parse_masks(output)
    if not masks:
        raise ValueError(f"{cal.plot.id}: SAM found no {layer.prompt!r}")
    hit = mask_hits(masks, shape, origin)
    widths = edges(hit, cal, under)
    return masks, hit, widths, profile(widths, cal, _line_px(widths) if layer.beside else 0.0)


def calibration_from(result: dict) -> tuple[Calibration, tuple[int, int], tuple[int, int]]:
    """A saved reading's calibration, crop corner and the page's shape in track columns."""
    c = result["calibration"]
    plot = PLOTS_BY_ID[result["id"]]
    cal = Calibration(
        c["track_px"][0],
        c["track_px"][1],
        tuple(c["depth_rows_px"]),
        tuple(c["body_rows_px"]),
        plot,
    )
    shape = (c["image_height_px"], cal.right - cal.left - 1)
    return cal, tuple(c["crop_origin_px"]), shape


def _layer(plot: Plot, phase: str) -> Layer:
    return next(layer for layer in plot.layers if layer.phase == phase)


def _checks(
    plot: Plot, rgb: np.ndarray, cal: Calibration, result: dict, las: pd.DataFrame | None
) -> None:
    oil = result["phases"]["oil"]["rate"]
    depths = sorted(oil)
    top = float(np.median([oil[d] for d in depths if d <= depths[0] + 30]))
    result["oil_rate_at_top"] = round(top)
    if plot.simulated_sm3_per_day is not None:  # kept only as pictures: its report checks it
        zones = zones_2009(rgb, cal)
        # Below the gap between the second and third zones: the oil from the lower three.
        gap = (zones[1][1] + zones[2][0]) / 2
        below = float(np.median([oil[d] for d in depths if abs(d - gap) <= 5]))
        result["checks"] = {
            "zones_m": zones,
            "upper_two_zones_share": round(1 - below / top, 3),
            "report_says": {"quote": UPPER_TWO_QUOTE, "share": UPPER_TWO_SHARE, "page": 4},
            "top_rate_over_simulated_surface_rate": round(top / plot.simulated_sm3_per_day, 3),
        }
        return
    checks = {}
    for phase, rates in ((p, result["phases"][p]["rate"]) for p in ("oil", "water")):
        if las is None:
            break
        from_plot = shares(rates)
        from_las = {}
        for d in from_plot:
            share = inflow_below(las, d)
            if share is not None:
                from_las[d] = share[f"{phase}_share"]
        checks[f"{phase}_against_las"] = _compare(from_plot, from_las)
    water = result["phases"]["water"]["rate"]
    w_top = float(np.median([water[d] for d in sorted(water) if d <= depths[0] + 30]))
    checks["water_cut_at_top"] = round(w_top / (w_top + top), 3)
    result["checks"] = checks


def build(raw_dir: Path, out_dir: Path, sam: dict[str, dict[str, str]] | None) -> list[dict]:
    """Read every plot (by SAM where its outputs are given); write the results and overlays."""
    results = []
    (out_dir / PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    for plot in PLOTS:
        outputs = sam.get(plot.id) if sam else None
        survey = SURVEYS_BY_WELL.get(plot.well)
        las = None
        if survey and survey.survey == plot.survey:
            las = read_las(raw_dir / survey.directory / survey.passes[plot.pass_name])
        result, overlay = read(page_image(raw_dir, plot), plot, outputs, las)
        overlay.save(out_dir / PLOTS_DIR / f"{plot.id}.png", optimize=True)
        results.append(result)
    (out_dir / PLOTS_FILE).write_text(json.dumps(results, indent=1))
    return results


def track_png(raw_dir: Path, plot: Plot) -> bytes:
    image = page_image(raw_dir, plot)
    cal = calibrate(np.asarray(image).astype(int), plot)
    buffer = io.BytesIO()
    crop(image, cal)[0].save(buffer, format="PNG")
    return buffer.getvalue()


def saved_outputs(out_dir: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for plot in PLOTS:
        for layer in plot.layers:
            path = out_dir / PLOTS_DIR / f"{plot.id}-{layer.phase}.txt"
            if path.exists():
                found.setdefault(plot.id, {})[layer.phase] = path.read_text()
    return {k: v for k, v in found.items() if len(v) == len(PLOTS_BY_ID[k].layers)}


async def _segment_all(api_key: str, base_url: str, raw_dir: Path, out_dir: Path) -> None:
    for plot in PLOTS:
        png = track_png(raw_dir, plot)
        for layer in plot.layers:
            path = out_dir / PLOTS_DIR / f"{plot.id}-{layer.phase}.txt"
            if path.exists():
                continue
            text = await segment(api_key, base_url, png, layer.prompt)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{plot.id} {layer.prompt!r}: {len(parse_masks(text))} object(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--colour", action="store_true", help="colour traces only, no SAM call")
    args = parser.parse_args()
    settings = get_settings()
    out = settings.processed_dir
    if not args.colour:
        if not settings.model_api_key:
            raise SystemExit("MODEL_API_KEY is not set (SAM 3.1 is on the Meta Model API)")
        asyncio.run(
            _segment_all(settings.model_api_key, settings.meta_base_url, settings.raw_dir, out)
        )
    results = build(settings.raw_dir, out, None if args.colour else saved_outputs(out))
    for r in results:
        phases = {
            p: {k: v for k, v in r["phases"][p].items() if k not in ("rate", "colour_trace")}
            for p in r["phases"]
        }
        print(
            json.dumps(
                {
                    "id": r["id"],
                    "oil_rate_at_top": r["oil_rate_at_top"],
                    "phases": phases,
                    "checks": r["checks"],
                }
            )
        )


if __name__ == "__main__":
    main()
