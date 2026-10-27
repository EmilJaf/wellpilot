"""The field timeline: the screen replayed with the data up to each month end, so an alarm
appears when the data first showed it (on the real Volve fixture)."""

import shutil
from pathlib import Path

import pandas as pd

from wellpilot.analysis.timeline import _intervals, build
from wellpilot.ingest.operations import ingest

FIXTURES = Path(__file__).parent / "fixtures"


def test_the_dead_gauge_is_flagged_the_month_it_died_and_work_is_on_the_line(tmp_path) -> None:
    processed = tmp_path / "processed"
    shutil.copytree(FIXTURES / "processed", processed)
    ingest(FIXTURES / "ddr", processed)
    months = list(pd.date_range("2010-08-31", "2010-11-30", freq="ME"))

    out = build(processed, months)

    f12 = next(w for w in out["wells"] if w["well"] == "F-12")
    gauge = next(i for i in f12["issues"] if i["kind"] == "dead_gauge")
    assert (gauge["first_flagged"], gauge["since"], gauge["level"]) == (
        "2010-10-31",
        "2010-10-12",
        "alarm",
    )
    assert gauge["cleared"] is None
    assert [j["start"] for j in f12["work"]] == ["2010-09-23", "2014-11-22"]
    assert "3150 m" in f12["work"][1]["summary"]
    assert out["replayed"] == 4 and f12["oil_rate"]["month"][0] == "2008-02"


def test_a_single_month_without_an_issue_does_not_clear_it() -> None:
    months = ["2014-10-31", "2014-11-30", "2014-12-31", "2015-01-31", "2015-02-28", "2015-03-31"]
    issue = {"level": "alarm", "since": "2010-10-12", "text": "dead"}
    found = [(m, issue) for m in months if m not in ("2014-12-31", "2015-02-28", "2015-03-31")]

    stretches = _intervals(found, months)

    assert [(s["first_flagged"], s["cleared"]) for s in stretches] == [("2014-10-31", "2015-02-28")]
