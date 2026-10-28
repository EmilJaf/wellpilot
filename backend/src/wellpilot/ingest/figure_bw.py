"""A black curve among black text: where tracing a colour fails and SAM 3.1 does not.

The plan's forecast curve (figure.py) is magenta on grey, so its colour alone finds it, and
SAM's reading agrees with that colour trace to 10 Sm3/d: there SAM adds a second, independent
reading, not something colour could not do. Here it does. F-11 B's completion report (page
47) draws its time-against-depth chart with the operations curve, "Ops (f)", in black,
on a grid labelled in black, with black operation names written up the plot area and black
markers along its foot. A trace of dark pixels per column cannot tell the curve from the
words; SAM is asked for "black plot line" and returns the curve alone.

Both are traced in the same plot area and compared: where they differ by more than JUMP_PX,
the dark-pixel trace has left the curve for a label or a marker. The overlay draws both.

Runs once (one SAM call, on the Meta Model API with MODEL_API_KEY):
    uv run python -m wellpilot.ingest.figure_bw            # SAM, then the comparison
    uv run python -m wellpilot.ingest.figure_bw --from-output <file>   # rebuild, no call
"""

import argparse
import asyncio
import io
import json
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from wellpilot.config import get_settings
from wellpilot.ingest.figure import _longest_run, parse_masks, segment

SOURCE = {"doc_id": "f11b_completion", "page": 47, "figure": "Time vs depth curve, NO 15/9-F-11 B"}
PDF = "F11B_Completion_FWR.pdf"
PAGE_WIDTH_PX = 2480
# Tried in turn (Oct 28): "black line with x markers", "black curve" and "line with
# crosses" returned nothing; "black line" and "thin black line" only the steep start;
# "black plot line" the whole operations curve.
PROMPT = "black plot line"
JUMP_PX = 12  # a trace this far from the other has left the curve (the line is ~4 px thick)
RESULT_FILE = "figure_bw.json"
OVERLAY_FILE = "figure_bw_overlay.png"
SAM_OUTPUT_FILE = "figure_bw_sam_output.txt"


def render(pdf_path: Path) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[SOURCE["page"] - 1]
    width, _ = page.get_size()
    return page.render(scale=PAGE_WIDTH_PX / width).to_pil().convert("RGB")


def plot_area(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """The chart's pale yellow fill (255, 255, ~245): the longest run of rows, then of
    columns within them, that are mostly that colour (gridlines cross it, so 'mostly')."""
    rgb = pixels.astype(int)
    fill = (rgb[..., 0] > 245) & (rgb[..., 1] > 245) & (rgb[..., 2] < 253) & (rgb[..., 2] > 225)
    top, bottom = _longest_run(_bridged(fill.mean(axis=1) > 0.3))
    left, right = _longest_run(_bridged(fill[top:bottom].mean(axis=0) > 0.3))
    return left, top, right, bottom


def _bridged(flags: np.ndarray, gap: int = 8) -> np.ndarray:
    """Flags with short gaps filled: gridlines cross the fill every few dozen pixels."""
    out = flags.copy()
    idx = np.flatnonzero(flags)
    for a, b in zip(idx, idx[1:], strict=False):
        if 1 < b - a <= gap:
            out[a:b] = True
    return out


def dark_trace(pixels: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, float]:
    """No model: for each column, the middle row of its dark, unsaturated pixels (black)."""
    left, top, right, bottom = box
    rgb = pixels.astype(int)
    dark = (rgb.max(axis=-1) < 110) & (rgb.max(axis=-1) - rgb.min(axis=-1) < 40)
    trace = {}
    for x in range(left, right):
        ys = np.flatnonzero(dark[top:bottom, x])
        if len(ys):
            trace[x] = top + float(np.median(ys))
    return trace


def sam_trace(output: str, box: tuple[int, int, int, int]) -> tuple[dict[int, float], float]:
    """SAM's widest mask as a curve, per column the middle row of its pixels, in page pixels
    (SAM saw the plot area alone, so its coordinates start at the box's corner)."""
    from wellpilot.ingest.figure import Axes, mask_trace

    records = parse_masks(output)
    if not records:
        raise ValueError("SAM returned no mask")
    left, top, right, bottom = box
    axes = Axes(left, right, top, bottom, [])
    # Each mask is a piece of the curve (the steep start, the long rise); per column the
    # widest mask that covers it.
    trace: dict[int, float] = {}
    for record in sorted(records, key=lambda r: r.bounds.right - r.bounds.left):
        trace |= mask_trace(record, axes, (left, top))
    widest = max(records, key=lambda r: r.bounds.right - r.bounds.left)
    return trace, float(getattr(widest, "confidence", None) or 0.0)


def compare(dark: dict[int, float], sam: dict[int, float]) -> dict:
    both = sorted(set(dark) & set(sam))
    gaps = np.array([abs(dark[x] - sam[x]) for x in both])
    off = gaps > JUMP_PX
    return {
        "columns_compared": len(both),
        "dark_trace_off_curve_share": round(float(off.mean()), 3) if len(both) else None,
        "median_gap_px": round(float(np.median(gaps)), 1) if len(both) else None,
        "dark_trace_jumps": int(np.sum(np.abs(np.diff([dark[x] for x in sorted(dark)])) > JUMP_PX)),
        "sam_trace_jumps": int(np.sum(np.abs(np.diff([sam[x] for x in sorted(sam)])) > JUMP_PX)),
    }


def overlay(figure: Image.Image, box, dark: dict[int, float], sam: dict[int, float]) -> Image.Image:
    left, top, right, bottom = box
    crop = figure.crop((left - 20, top - 20, right + 20, bottom + 20)).copy()
    draw = ImageDraw.Draw(crop)
    for trace, colour, width in ((dark, (230, 120, 0), 3), (sam, (0, 110, 220), 4)):
        points = [(x - left + 20, y - top + 20) for x, y in sorted(trace.items())]
        for a, b in zip(points, points[1:], strict=False):
            if b[0] - a[0] <= 2:
                draw.line([a, b], fill=colour, width=width)
    return crop


def build(pdf_path: Path, out_dir: Path, sam_output: str) -> dict:
    figure = render(pdf_path)
    pixels = np.asarray(figure)
    box = plot_area(pixels)
    dark = dark_trace(pixels, box)
    sam, confidence = sam_trace(sam_output, box)
    result = {
        "source": SOURCE,
        "prompt": PROMPT,
        "plot_area_px": list(box),
        "sam_confidence": confidence,
        **compare(dark, sam),
        "read": "dark-pixel trace (no model) against SAM 3.1's mask, per column of the plot area",
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    overlay(figure, box, dark, sam).save(out_dir / OVERLAY_FILE)
    return result


def plot_png(pdf_path: Path) -> bytes:
    figure = render(pdf_path)
    left, top, right, bottom = plot_area(np.asarray(figure))
    buffer = io.BytesIO()
    figure.crop((left, top, right, bottom)).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-output", type=Path, help="SAM output saved earlier, no call")
    args = parser.parse_args()
    settings = get_settings()
    pdf = settings.raw_dir / "reports" / PDF
    if args.from_output:
        output = args.from_output.read_text()
    else:
        if not settings.model_api_key:
            raise SystemExit("MODEL_API_KEY is not set (SAM 3.1 is on the Meta Model API)")
        output = asyncio.run(
            segment(settings.model_api_key, settings.meta_base_url, plot_png(pdf), PROMPT)
        )
        (settings.processed_dir / SAM_OUTPUT_FILE).write_text(output)
    print(json.dumps(build(pdf, settings.processed_dir, output), indent=1))


if __name__ == "__main__":
    main()
