"""The plan's forecast curve, read off its figure by SAM 3.1.

The 2005 development plan (PUD) gives its oil forecast twice: as yearly volumes in Table 4.5
and as a drawn curve in Figure 4.2 (page 23), the only place it shows the rate it expected
day by day (the plateau, when it would end, how steeply it would fall). The figure is a
picture: no text layer holds the curve.

SAM 3.1 (Meta's segmentation model, over the Meta Model API) finds the curve from a text
prompt ("magenta line") and returns its mask. Turning pixels into Sm3/d needs the axes:
- the plot area is the figure's grey fill, found in the image;
- its gridlines (found in the image, every 1000 Sm3/d) and edges calibrate the rate axis;
  the edge values, 0 and 9000 Sm3/d, and the date axis' first and last labels, 2007 and
  2017, are read off the figure's labels, as a person digitising a chart would enter them.
The same curve is also traced by its colour, with no model, and the two are compared, so
the reading carries its own check.

Runs once (one SAM call, on the Meta Model API with MODEL_API_KEY):
    uv run python -m wellpilot.ingest.figure            # SAM, then the colour check
    uv run python -m wellpilot.ingest.figure --colour   # the colour trace only, free
"""

import argparse
import asyncio
import base64
import io
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)
from openai import AsyncOpenAI
from PIL import Image

from wellpilot.config import get_settings

CURVE_FILE = "plan_figure_curve.json"
OVERLAY_FILE = "plan_figure_overlay.png"
SAM_OUTPUT_FILE = "plan_figure_sam_output.txt"  # SAM's answer as it came, kept to rebuild
MODEL = "sam-3.1"
PROMPT = "magenta line"
SOURCE = {
    "doc_id": "volve_pud",
    "page": 23,
    "figure": "Figur 4.2 - Volve oljerate og kumulativ produksjon",
}
PAGE_WIDTH_PX = 3720  # three times the page images the app serves: thin lines stay solid
# Where Figure 4.2 sits on page 23, as fractions of the page (left, top, right, bottom).
FIGURE_BOX = (0.115, 0.49, 0.71, 0.765)
# The axis labels at the plot area's edges: 0 and 9000 Sm3/d, 2007 and 2017.
RATE_AXIS = (0.0, 9000.0)
YEAR_AXIS = (2007, 2017)
GRID_STEP = 1000.0  # Sm3/d between gridlines


@dataclass(frozen=True)
class Axes:
    left: int
    right: int
    top: int
    bottom: int
    gridlines: list[float]  # pixel rows, bottom first

    def rate(self, row: float) -> float:
        """Sm3/d at a pixel row: the straight line through the gridlines (1000, 2000, ...
        from the bottom)."""
        values = [RATE_AXIS[0] + GRID_STEP * (k + 1) for k in range(len(self.gridlines))]
        slope, intercept = np.polyfit(self.gridlines, values, 1)
        return float(slope * row + intercept)

    def day(self, column: float) -> date:
        years = YEAR_AXIS[1] - YEAR_AXIS[0]
        frac = (column - self.left) / (self.right - self.left)
        start = date(YEAR_AXIS[0], 1, 1)
        return start + timedelta(days=frac * years * 365.25)


def axes_from(calibration: dict) -> Axes:
    """The axes a saved curve file was read with."""
    left, top, right, bottom = calibration["plot_area_px"]
    return Axes(left, right, top, bottom, list(calibration["gridline_rows"]))


def render_figure(pdf_path: Path) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[SOURCE["page"] - 1]
    width, _ = page.get_size()
    image = page.render(scale=PAGE_WIDTH_PX / width).to_pil().convert("RGB")
    w, h = image.size
    left, top, right, bottom = FIGURE_BOX
    return image.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """[start, end) of the longest run of True."""
    best, start = (0, 0), None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_axes(pixels: np.ndarray) -> Axes:
    """The plot area and its gridlines, checked to be evenly spaced. The figure is a pasted
    JPEG: the fill is a grey near 185 and the gridlines are blurred dark bands. The plot
    area is the longest unbroken run of columns, and of rows, that are mostly neutral grey
    darker than the page (the chart's outer frame is single grey lines); a gridline is a
    band of rows much darker than the fill, its centre the darkest row."""
    rgb = pixels.astype(int)
    neutral = (rgb.max(axis=2) - rgb.min(axis=2) < 20) & (rgb.mean(axis=2) < 205)
    # Rows first, then columns counted within those rows: where the curve rises steeply it
    # covers half a column, so a column needs a quarter of the plot's height in grey.
    top, bottom = _longest_run(neutral.sum(axis=1) > 0.3 * neutral.sum(axis=1).max())
    left, right = _longest_run(neutral[top:bottom].sum(axis=0) > 0.25 * (bottom - top))
    # The median over the row, so the curves crossing it do not pull it.
    luma = np.median(rgb[top:bottom, left:right].mean(axis=2), axis=1)
    fill = float(np.median(luma))
    bands: list[list[int]] = []
    for r in np.flatnonzero(luma < fill - 40):
        if bands and r - bands[-1][-1] <= 2:
            bands[-1].append(int(r))
        else:
            bands.append([int(r)])
    centres = [top + float(b[int(np.argmin(luma[b]))]) for b in bands]
    # Bands touching the edges are the plot's own border.
    centres = [c for c in centres if top + 8 < c < bottom - 8]
    expected = int(round((RATE_AXIS[1] - RATE_AXIS[0]) / GRID_STEP)) - 1
    if len(centres) != expected:
        raise ValueError(f"found {len(centres)} gridlines, expected {expected}")
    # The gridlines calibrate the rate axis; the blurred edges only bound the plot. The
    # chart is a small raster scaled up, so its lines sit on whole source pixels and the
    # spacing varies by a few percent; the straight line through all of them evens it out.
    spacing = np.diff(sorted(centres))
    if np.ptp(spacing) > 0.10 * spacing.mean():
        raise ValueError(f"the gridlines are not evenly spaced: {spacing.tolist()}")
    return Axes(left, right, top, bottom, sorted(centres, reverse=True))


def colour_trace(pixels: np.ndarray, axes: Axes) -> dict[int, float]:
    """The magenta curve by its colour: for each column of the plot area, the middle row of
    its magenta pixels. No model involved."""
    rgb = pixels.astype(int)
    magenta = (rgb[..., 0] > 170) & (rgb[..., 1] < 140) & (rgb[..., 2] > 170)
    trace = {}
    for x in range(axes.left, axes.right):
        ys = np.flatnonzero(magenta[axes.top : axes.bottom, x])
        if len(ys):
            trace[x] = axes.top + float(np.median(ys))
    return trace


def mask_trace(
    record: SegmentationMaskRecord, axes: Axes, origin: tuple[int, int]
) -> dict[int, float]:
    """A SAM mask as a curve: the mask raster spans its box in the image SAM saw (whose
    top-left corner is `origin` in the figure); for each figure column, the middle row of
    the mask's pixels."""
    raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
        record.mask.height, record.mask.width
    )
    b = record.bounds
    sx = (b.right - b.left) / record.mask.width
    sy = (b.bottom - b.top) / record.mask.height
    rows, cols = np.nonzero(raster)
    xs = origin[0] + b.left + (cols + 0.5) * sx
    ys = origin[1] + b.top + (rows + 0.5) * sy
    trace = {}
    for x in range(axes.left, axes.right):
        hit = ys[np.abs(xs - (x + 0.5)) <= max(sx, 1.0) / 2]
        if len(hit):
            trace[x] = float(np.median(hit))
    return trace


def parse_masks(output_text: str) -> list[SegmentationMaskRecord]:
    """The masks in SAM's output text, by Meta's own parser (meta-sam-parser)."""
    parser = image_segmentation_format().create_parser()
    parser.push(output_text if output_text.endswith("\n") else output_text + "\n")
    result = parser.finish(CompletedOutcome()).result
    return [r for r in result.records if isinstance(r, SegmentationMaskRecord)]


async def segment(api_key: str, base_url: str, png: bytes, prompt: str = PROMPT) -> str:
    """One SAM 3.1 call over the Responses API; returns its output text."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    image = base64.b64encode(png).decode()
    response = await client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image}"},
                ],
            }
        ],
        metadata={"include_confidence": "true"},
    )
    return response.output_text


def monthly(trace: dict[int, float], axes: Axes) -> dict:
    """Mid-month readings of a trace (the median of the columns in each month)."""
    by_month: dict[str, list[float]] = {}
    for x, row in trace.items():
        by_month.setdefault(axes.day(x + 0.5).strftime("%Y-%m"), []).append(axes.rate(row))
    months = sorted(by_month)
    return {"month": months, "rate": [round(float(np.median(by_month[m]))) for m in months]}


def compare(a: dict[int, float], b: dict[int, float], axes: Axes) -> dict:
    """How far apart two traces of the same curve are, in Sm3/d, over the columns both have."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"columns": 0}
    gap = np.array([abs(axes.rate(a[x]) - axes.rate(b[x])) for x in shared])
    return {
        "columns": len(shared),
        "median_sm3_per_day": round(float(np.median(gap))),
        "p90_sm3_per_day": round(float(np.percentile(gap, 90))),
        "within_200": round(float((gap <= 200).mean()), 3),
    }


def overlay(figure: Image.Image, trace: dict[int, float]) -> Image.Image:
    """The figure with the read curve drawn over it in ink, to show what was read."""
    out = figure.convert("RGB").copy()
    pixels = out.load()
    for x, row in trace.items():
        for dy in range(-3, 4):
            y = int(round(row)) + dy
            if 0 <= y < out.size[1]:
                pixels[x, y] = (31, 38, 45)
    return out


def digitize(figure: Image.Image, sam_output: str | None) -> tuple[dict, dict[int, float]]:
    """The curve read off a figure: by SAM's mask when its output is given (with the colour
    trace as the check), else by colour alone. Returns the result and the trace drawn."""
    pixels = np.asarray(figure.convert("RGB"))
    axes = find_axes(pixels)
    colour = colour_trace(pixels, axes)
    result: dict = {
        "source": SOURCE,
        "calibration": {
            "plot_area_px": [axes.left, axes.top, axes.right, axes.bottom],
            "gridlines": len(axes.gridlines),
            "gridline_rows": [round(r, 1) for r in axes.gridlines],
            "rate_axis_sm3_per_day": list(RATE_AXIS),
            "year_axis": list(YEAR_AXIS),
            "how": "plot area and gridlines found in the image; edge values from the axis labels",
        },
        "colour_trace": monthly(colour, axes),
    }
    if sam_output is None:
        result.update({"read_by": "colour trace (no model)", "curve": result["colour_trace"]})
        return result, colour
    masks = parse_masks(sam_output)
    if not masks:
        raise ValueError("SAM returned no mask for the prompt")
    # SAM saw the plot area alone; the curve is the mask that spans most of its width.
    traces = [(m, mask_trace(m, axes, (axes.left, axes.top))) for m in masks]
    best, sam = max(traces, key=lambda t: len(t[1]))
    result.update(
        {
            "read_by": MODEL,
            "prompt": PROMPT,
            "objects_found": len(masks),
            "confidence": best.confidence,
            "curve": monthly(sam, axes),
            "agreement_with_colour": compare(sam, colour, axes),
        }
    )
    return result, sam


def build(pdf_path: Path, out_dir: Path, sam_output: str | None) -> dict:
    """The curve file and an overlay image, from SAM's output text (None: colour only)."""
    figure = render_figure(pdf_path)
    result, shown = digitize(figure, sam_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CURVE_FILE).write_text(json.dumps(result, indent=1))
    overlay(figure, shown).save(out_dir / OVERLAY_FILE, optimize=True)
    return result


def plot_png(pdf_path: Path) -> bytes:
    """The plot area alone, as SAM is shown it: no legend, whose sample line is magenta too."""
    figure = render_figure(pdf_path)
    axes = find_axes(np.asarray(figure))
    buffer = io.BytesIO()
    figure.crop((axes.left, axes.top, axes.right, axes.bottom)).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--colour", action="store_true", help="colour trace only, no SAM call")
    parser.add_argument(
        "--from-output", type=Path, help="rebuild from SAM output text saved earlier, no call"
    )
    args = parser.parse_args()
    settings = get_settings()
    pdf = settings.raw_dir / "reports" / "Volve_PUD.pdf"
    output = None
    if args.from_output:
        output = args.from_output.read_text()
    elif not args.colour:
        if not settings.model_api_key:
            raise SystemExit("MODEL_API_KEY is not set (SAM 3.1 is on the Meta Model API)")
        output = asyncio.run(segment(settings.model_api_key, settings.meta_base_url, plot_png(pdf)))
        (settings.processed_dir / SAM_OUTPUT_FILE).write_text(output)
    result = build(pdf, settings.processed_dir, output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("curve", "colour_trace")}, indent=1
        )
    )
    curve = result["curve"]
    print(f"{len(curve['month'])} months, {curve['month'][0]} to {curve['month'][-1]}")


if __name__ == "__main__":
    main()
