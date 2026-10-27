"""Reading the plan's forecast curve off its figure: the axes found in the image, the curve
traced by colour, and SAM 3.1's mask read through Meta's parser (a mask made with the same
codec, so no API call). On a chart drawn here, like the PUD's: grey fill, gridlines every
1000 Sm3/d, a magenta curve."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster  # the codec SAM's masks use
from PIL import Image

from wellpilot.ingest.figure import (
    axes_from,
    compare,
    digitize,
    find_axes,
    mask_trace,
    monthly,
    parse_masks,
)

LEFT, TOP, RIGHT, BOTTOM = 60, 30, 660, 480  # plot area in the drawn chart


def plan_rate(year: float) -> float:
    """A plan-like curve: ramp, plateau at 8500, decline."""
    if year < 2008:
        return 2500 + (year - 2007) * 6000
    if year < 2009.7:
        return 8500.0
    return 8500 * np.exp(-0.45 * (year - 2009.7)) + 600


def row_of(rate: float) -> float:
    return BOTTOM - (BOTTOM - TOP) * rate / 9000


def year_of(column: float) -> float:
    return 2007 + 10 * (column - LEFT) / (RIGHT - LEFT)


def chart() -> Image.Image:
    img = np.full((520, 700, 3), 255, dtype=np.uint8)
    img[TOP:BOTTOM, LEFT:RIGHT] = 187
    img[0, :] = img[-1, :] = 190  # an outer frame in the same grey, as in the PUD
    for k in range(1, 9):
        r = int(round(row_of(1000 * k)))
        img[r - 1 : r + 2, LEFT:RIGHT] = 80
    for x in range(LEFT, RIGHT):
        r = int(round(row_of(plan_rate(year_of(x + 0.5)))))
        img[r - 2 : r + 3, x] = (230, 40, 230)
    return Image.fromarray(img)


def test_the_axes_are_found_and_the_colour_trace_reads_the_curve() -> None:
    figure = chart()
    axes = find_axes(np.asarray(figure))
    result, _ = digitize(figure, None)

    assert (axes.left, axes.top, axes.right, axes.bottom) == (LEFT, TOP, RIGHT, BOTTOM)
    assert len(axes.gridlines) == 8
    assert axes.rate(row_of(4500)) == pytest.approx(4500, abs=25)
    assert axes.day(LEFT + (RIGHT - LEFT) / 2) == date(2012, 1, 1)
    curve = dict(zip(result["curve"]["month"], result["curve"]["rate"], strict=True))
    assert curve["2008-06"] == pytest.approx(8500, abs=60)
    assert curve["2012-06"] == pytest.approx(plan_rate(2012.45), abs=120)
    assert result["read_by"] == "colour trace (no model)"


def sam_output(figure: Image.Image) -> str:
    """What SAM 3.1 returns for "magenta line" on the plot area: one record, a box and its
    box-local mask, encoded with the codec of Meta's parser."""
    plot = np.asarray(figure)[TOP:BOTTOM, LEFT:RIGHT].astype(int)
    magenta = (plot[..., 0] > 170) & (plot[..., 1] < 140) & (plot[..., 2] > 170)
    rows, cols = np.nonzero(magenta)
    x1, y1, x2, y2 = cols.min(), rows.min(), cols.max(), rows.max()
    box = magenta[y1 : y2 + 1, x1 : x2 + 1]
    # SAM's rasters are coarser than the image: every other pixel.
    small = box[::2, ::2]
    payload = _encode_raster(small.astype(int).ravel().tolist(), small.shape[1], small.shape[0])
    h, w = plot.shape[:2]
    return (
        f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c=0.94|>"
        f"<|mask;x=0;y=0;data={small.shape[0]},{small.shape[1]},{payload};c=0.92|>\n"
    )


def test_sams_mask_is_read_and_checked_against_the_colour() -> None:
    figure = chart()

    result, trace = digitize(figure, sam_output(figure))

    assert result["read_by"] == "sam-3.1" and result["objects_found"] == 1
    assert result["confidence"] == pytest.approx(0.92)
    agreement = result["agreement_with_colour"]
    assert agreement["within_200"] > 0.95 and agreement["median_sm3_per_day"] < 60
    curve = dict(zip(result["curve"]["month"], result["curve"]["rate"], strict=True))
    assert curve["2008-06"] == pytest.approx(8500, abs=80)
    assert len(trace) > 0.9 * (RIGHT - LEFT)


def test_no_mask_is_an_error_and_disagreement_is_measured() -> None:
    figure = chart()
    axes = find_axes(np.asarray(figure))

    with pytest.raises(ValueError, match="no mask"):
        digitize(figure, "<0f>\n")
    far = compare({100: row_of(5000)}, {100: row_of(4000)}, axes)
    assert far["median_sm3_per_day"] == pytest.approx(1000, abs=5) and far["within_200"] == 0


FIXTURE = Path(__file__).parent / "fixtures" / "sam"


def test_sams_real_answer_rebuilds_the_saved_curve() -> None:
    """SAM 3.1's answer for the PUD's Figure 4.2, as it came: Meta's parser reads one mask,
    and the curve read from it is the one saved, month by month. The plateau is the plan's
    19 months, near its 9,000 Sm3/d capacity at 94% regularity."""
    saved = json.loads((FIXTURE / "plan_figure_curve.json").read_text())
    axes = axes_from(saved["calibration"])

    masks = parse_masks((FIXTURE / "plan_figure_sam_output.txt").read_text())

    assert len(masks) == 1 and masks[0].confidence == pytest.approx(0.91, abs=0.005)
    trace = mask_trace(masks[0], axes, (axes.left, axes.top))
    assert monthly(trace, axes) == saved["curve"]
    curve = dict(zip(saved["curve"]["month"], saved["curve"]["rate"], strict=True))
    plateau = [m for m, r in curve.items() if r >= 0.98 * max(curve.values())]
    assert (plateau[0], plateau[-1], len(plateau)) == ("2008-03", "2009-09", 19)
    assert max(curve.values()) == pytest.approx(9000 * 0.94, abs=40)
    assert saved["agreement_with_colour"]["median_sm3_per_day"] <= 15
