"""Production logs: F-12's survey of September 2010 read from its LAS files with no model,
served as shares by the production_log tool, cut at the store's as-of date, and citable in
an answer the checker accepts."""

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.tools import execute
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production_logs import PLOTS_FILE, ingest, read_las

FIXTURES = Path(__file__).parent / "fixtures"


def test_a_las_file_gives_its_curves_by_name() -> None:
    log = read_las(FIXTURES / "plt" / "2010" / "F12_PLT_302B.LAS")

    assert {"depth_m", "oil", "water", "pressure_bar", "temperature_c"} <= set(log.columns)
    # Above the flowing interval the rates are the file's null value, read as missing.
    assert log["oil"].isna().iloc[0] and log["pressure_bar"].notna().all()
    assert log["depth_m"].is_monotonic_increasing


@pytest.fixture
def processed(tmp_path) -> Path:
    out = tmp_path / "processed"
    shutil.copytree(FIXTURES / "processed", out)
    # The 2009 survey, read off its report's plots (wellpilot.ingest.log_plots).
    shutil.copy(FIXTURES / "plt" / PLOTS_FILE, out / PLOTS_FILE)
    counts = ingest(FIXTURES, out)
    assert counts["surveys"] == ["PLT 2009-03", "PLT 2010-09"]
    return out


def call(processed: Path, as_of: date | None = None, **args) -> dict:
    return execute(ProductionStore(processed, as_of=as_of), "production_log", json.dumps(args))


def survey(result: dict, name: str) -> dict:
    return next(s for s in result["surveys"] if s["survey"] == name)


def test_the_log_shows_oil_from_the_top_zone_and_water_from_below(processed) -> None:
    result = survey(call(processed, well="F-12", below_m=[3150, 3210])["result"], "PLT 2010-09")

    assert result["date"] == "2010-09-22" and result["read_from"].startswith("LAS files")
    passes = {p["pass"]: p for p in result["passes"]}
    assert set(passes) == {"low rate", "medium rate"}
    for p in passes.values():
        # Oil comes in from about 3130 to 3165 m, the water below it.
        assert 3125 <= p["oil_comes_in"]["comes_in_from_m"] <= 3140
        assert 3160 <= p["oil_comes_in"]["to_m"] <= 3170
        assert 3160 <= p["water_comes_in"]["comes_in_from_m"] <= 3170
        assert 3160 <= p["water_comes_in"]["half_below_m"] <= 3180
        below_3150, below_3210 = p["below"]
        # The 2014 plug at 3150 m sits below nearly all the water; the 2010 one at 3210 m
        # below about a third of it and almost none of the oil.
        assert below_3150["water_share"] >= 0.95
        assert 0.28 <= below_3210["water_share"] <= 0.42 and below_3210["oil_share"] <= 0.03
    assert passes["medium rate"]["surface_liquid_sm3_per_day"] == 3000
    # The log's water cut at the top agrees with the well's production the month before.
    produced = result["production_30_days_before"]["water_cut"]
    assert all(abs(p["water_cut_at_top"] - produced) < 0.05 for p in passes.values())


def test_in_2009_the_lower_zones_gave_oil_and_by_2010_water(processed) -> None:
    """The 2009 survey exists only as its report's plots, read as pictures: then, the oil came
    from all five zones, about a sixth of it from below 3200 m, with no water. In 2010 almost
    no oil came from there, and about a third of the water did."""
    result = call(processed, well="F-12", below_m=[3200])["result"]
    before, after = survey(result, "PLT 2009-03"), survey(result, "PLT 2010-09")

    assert "pictures" in before["read_from"] and before["rate_unit"] == "m3/d downhole"
    assert before["production_30_days_before"]["water_cut"] == 0
    for p in before["passes"]:
        assert "water_comes_in" not in p
        assert 0.08 <= p["below"][0]["oil_share"] <= 0.2
    # The plots' reading agrees with the report's conclusion, "~83% ... the upper 2".
    check = before["check"]
    assert check["report_says"]["share"] == 0.83
    assert all(abs(v - 0.83) <= 0.08 for v in check["upper_two_zones_share"].values())
    assert len(check["perforated_zones_m"]) == 5
    for p in after["passes"]:
        assert p["below"][0]["oil_share"] <= 0.03 and p["below"][0]["water_share"] >= 0.25
    # The 2010 plot, read the same way, against the files.
    gaps = after["check"]["plot_against_files_median_share_gap"]
    assert all(g <= 0.03 for pass_ in gaps.values() for g in pass_.values())


def test_wells_without_a_log_and_dates_before_it_are_refused(processed) -> None:
    other = call(processed, well="F-14")
    early = call(processed, as_of=date(2009, 3, 1), well="F-12")
    between = call(processed, as_of=date(2010, 9, 1), well="F-12")["result"]

    assert "logs exist for: F-12" in other["error"]
    assert "logs exist for: none" in early["error"]
    assert [s["survey"] for s in between["surveys"]] == ["PLT 2009-03"]


def test_a_share_from_the_log_passes_the_checker(processed) -> None:
    log = EvidenceLog()
    payload = call(processed, well="F-12", below_m=[3210])
    eid = log.add("production_log", {"well": "F-12"}, payload["result"], lambda: None).id
    medium = next(
        p for p in survey(payload["result"], "PLT 2010-09")["passes"] if p["pass"] == "medium rate"
    )
    share = medium["below"][0]["water_share"]
    written = f"{round(share * 100)}%"

    def finding(text: str, value: float) -> Finding:
        return Finding(
            claim=f"F-12's production log of 2010-09-22 shows {text} of the water entering "
            "below 3210 m.",
            evidence_ids=[eid],
            confidence="fact",
            numbers=[
                DeclaredNumber(text=text, value=value, evidence_id=eid),
                DeclaredNumber(text="3210", value=3210, evidence_id=eid),
            ],
        )

    assert (
        check_submission(Submission(summary="Done.", findings=[finding(written, share)]), log) == []
    )
    assert check_submission(Submission(summary="Done.", findings=[finding("60%", 0.6)]), log)
