"""Production logs read off their plots: the track and the labelled depths found in the
image, the rate areas traced by colour, and SAM 3.1's masks read through Meta's parser
(masks made with the same codec, so no API call). On plots drawn here in the two styles of
F-12's surveys: 2009's (an oil area from a zero line, depths underlined) and 2010's (water
in cyan, the oil drawn beside it in black, on a grid)."""

from dataclasses import replace

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster  # the codec SAM's masks use
from PIL import Image

from wellpilot.ingest.log_plots import (
    OIL_2009,
    OIL_2010,
    PLOTS_BY_ID,
    WATER_2010,
    calibrate,
    colour_hits,
    read,
)

# 2009 style: depths underlined at these rows; the track from -500 to 6000 m3/d.
ROW_3100, ROW_3200 = 461, 980
PX_PER_M = (ROW_3200 - ROW_3100) / 100
LEFT, RIGHT = 523, 708
ZONES = [(3131, 3147), (3161, 3180), (3220, 3231)]


def y_of(depth: float) -> int:
    return int(round(ROW_3100 + (depth - 3100) * PX_PER_M))


def oil_2009(depth: float) -> float:
    """5000 m3/d at the top: 1000 from the first zone, 3200 from the second, 800 below."""
    if depth < 3131:
        return 5000.0
    if depth < 3147:
        return 5000 - 1000 * (depth - 3131) / 16
    if depth < 3161:
        return 4000.0
    if depth < 3180:
        return 4000 - 3200 * (depth - 3161) / 19
    if depth < 3220:
        return 800.0
    return max(800 - 800 * (depth - 3220) / 11, 0.0)


def plot_2009() -> Image.Image:
    img = np.full((1400, 960, 3), 255, dtype=np.uint8)
    img[[303, 1274], 72:897] = 128  # the frame: header bottom, footer top
    img[ROW_3100 : ROW_3100 + 2, 117:154] = 0  # "3100" underlined
    img[ROW_3200 : ROW_3200 + 2, 117:154] = 0
    img[303:1274, [LEFT, RIGHT]] = 150  # the track's edges
    zero = LEFT + (RIGHT - LEFT) * 500 / 6500
    for y in range(305, 1273):
        depth = 3100 + (y - ROW_3100) / PX_PER_M
        rate = oil_2009(depth)
        if rate > 0:
            img[y, int(round(zero)) : int(round(zero + rate * (RIGHT - LEFT) / 6500))] = (0, 255, 0)
    for top, bottom in ZONES:
        img[y_of(top) : y_of(bottom), 80:96] = (255, 0, 0)
    return Image.fromarray(img)


def sam_output(hit: np.ndarray, origin_row: int) -> str:
    """What SAM 3.1 returns for an area on the crop it was shown (the track over the plot's
    rows of data): a box and its box-local mask at half resolution."""
    crop = hit[origin_row:]
    rows, cols = np.nonzero(crop)
    x1, y1, x2, y2 = cols.min(), rows.min(), cols.max(), rows.max()
    small = crop[y1 : y2 + 1, x1 : x2 + 1][::2, ::2]
    payload = _encode_raster(small.astype(int).ravel().tolist(), small.shape[1], small.shape[0])
    h, w = crop.shape
    return (
        f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c=0.95|>"
        f"<|mask;x=0;y=0;data={small.shape[0]},{small.shape[1]},{payload};c=0.93|>\n"
    )


PLOT_2009 = replace(PLOTS_BY_ID["f12-2009-high"], rate_axis=(-500.0, 6000.0))


def test_a_2009_plot_is_calibrated_and_read_by_colour() -> None:
    image = plot_2009()
    cal = calibrate(np.asarray(image).astype(int), PLOT_2009)

    result, _ = read(image, PLOT_2009, None)

    assert (cal.left, cal.right) == (LEFT, RIGHT)
    assert cal.depth(ROW_3100 + 0.5) == pytest.approx(3100, abs=0.5)
    assert cal.body == (305, 1273)
    oil = result["phases"]["oil"]["rate"]
    assert oil[3100.0] == pytest.approx(5000, abs=60)
    assert oil[3170.0] == pytest.approx(oil_2009(3170), abs=120)
    checks = result["checks"]
    # The upper two zones gave 4200 of 5000: 84%.
    assert checks["upper_two_zones_share"] == pytest.approx(0.84, abs=0.02)
    zones = [d for zone in checks["zones_m"][:3] for d in zone]
    assert zones == pytest.approx([d for zone in ZONES for d in zone], abs=0.6)
    assert checks["top_rate_over_simulated_surface_rate"] == pytest.approx(5000 / 3850, abs=0.02)
    assert result["phases"]["oil"]["read_by"] == "colour trace (no model)"


def test_sams_mask_of_a_2009_plot_reads_the_same_rates() -> None:
    image = plot_2009()
    rgb = np.asarray(image).astype(int)
    cal = calibrate(rgb, PLOT_2009)
    output = sam_output(colour_hits(rgb, cal, OIL_2009), cal.body[0])

    result, overlay = read(image, PLOT_2009, {"oil": output})

    oil = result["phases"]["oil"]
    assert oil["read_by"] == "sam-3.1" and oil["prompt"] == "green area"
    assert oil["objects_found"] == 1 and oil["confidence"] == pytest.approx(0.93)
    assert oil["agreement_with_colour"]["median_share_gap"] <= 0.01
    assert result["checks"]["upper_two_zones_share"] == pytest.approx(0.84, abs=0.02)
    assert overlay.size == (RIGHT - LEFT - 1, cal.body[1] - cal.body[0])


def test_no_mask_is_an_error() -> None:
    with pytest.raises(ValueError, match="SAM found no 'green area'"):
        read(plot_2009(), PLOT_2009, {"oil": "<0f>\n"})


# 2010 style: a grid every 5 m from 3100 to 3500 m, 10 px apart; water from zero in cyan
# with a dark blue line along it; the oil beside it in black, with a 3 px black line where
# there is none. The grid shows whole in a track with no fill.
GRID_TOP, STEP = 100, 10.0
TRACK, CLEAR = (200, 400), (40, 160)


def depth_2010(y: float) -> float:
    return 3100 + 5 * (y - GRID_TOP) / STEP


def plot_2010() -> Image.Image:
    img = np.full((1000, 460, 3), 255, dtype=np.uint8)
    for k in range(81):
        img[GRID_TOP + int(k * STEP), CLEAR[0] : TRACK[1]] = 0
    img[GRID_TOP - 60, CLEAR[0] : TRACK[1]] = 0  # the header box, farther up
    img[GRID_TOP : GRID_TOP + 801, [TRACK[0], TRACK[1]]] = 0
    unit = (TRACK[1] - TRACK[0]) / 5000
    for y in range(GRID_TOP + 1, GRID_TOP + 800):
        d = depth_2010(y)
        water = 2000.0 if d < 3165 else (1000.0 if d < 3230 else 0.0)
        oil = 1200.0 if d < 3132 else (1200 * (3165 - d) / 33 if d < 3165 else 0.0)
        if water == 0:
            continue
        x0 = TRACK[0] + 1
        x1 = x0 + int(round(water * unit))
        img[y, x0:x1] = (0, 255, 255)
        img[y, x1 : x1 + 2] = (0, 0, 200)
        img[y, x1 + 2 : x1 + 2 + 3 + int(round(oil * unit))] = 0
    return Image.fromarray(img)


PLOT_2010 = replace(
    PLOTS_BY_ID["f12-2010-medium"], track_px=TRACK, grid_px=(CLEAR[0] + 2, CLEAR[1] - 2)
)


def test_a_2010_plot_reads_water_and_the_oil_beside_it() -> None:
    image = plot_2010()
    rgb = np.asarray(image).astype(int)
    cal = calibrate(rgb, PLOT_2010)

    by_colour, _ = read(image, PLOT_2010, None)
    outputs = {
        "water": sam_output(colour_hits(rgb, cal, WATER_2010), cal.body[0]),
        "oil": sam_output(colour_hits(rgb, cal, OIL_2010), cal.body[0]),
    }
    by_sam, _ = read(image, PLOT_2010, outputs)

    assert cal.rows == (GRID_TOP, GRID_TOP + 800)
    for result in (by_colour, by_sam):
        water, oil = result["phases"]["water"]["rate"], result["phases"]["oil"]["rate"]
        assert water[3120.0] == pytest.approx(2000 + 2 * 25, abs=60)  # the blue line is water's
        assert oil[3120.0] == pytest.approx(1200, abs=60)  # the black line taken off
        assert oil[3200.0] == pytest.approx(0, abs=30)
        assert water[3200.0] == pytest.approx(1050, abs=60)
    assert by_sam["phases"]["oil"]["agreement_with_colour"]["median_share_gap"] <= 0.02
