"""Daily operation reports: parsed with no model, served by the well_operations tool, cut at
the store's as-of date, and quotable in an answer the checker accepts."""

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from wellpilot.agent.checker import Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.tools import execute
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.operations import ingest, parse_report

FIXTURES = Path(__file__).parent / "fixtures"
REPORTS = FIXTURES / "ddr"


def test_a_report_is_dated_by_the_day_its_work_was_done() -> None:
    day, steps = parse_report(REPORTS / "15_9_F_12_2014_11_27.html")

    assert day["well"] == "F-12" and day["wellbore"] == "15/9-F-12"
    assert day["date"] == "2014-11-26" and day["report_date"] == "2014-11-27"
    assert day["summary"].startswith('Installed 7" EV0 plug with mid element at 3150 m.')
    assert day["perforations"] == "3126.2-3130.7 mMD"
    assert day["contractor"] == "Altus Intervention"
    assert "workover" in day["activities"]
    perforate = [s for s in steps if s["sub_activity"] == "perforate"]
    assert perforate and all(s["date"] == "2014-11-26" for s in perforate)
    # Empty cells are self-closing in the HTML; none swallows the next cell.
    assert all(s["state"] in ("ok", "fail", "") for s in steps)


@pytest.fixture
def store(tmp_path) -> ProductionStore:
    processed = tmp_path / "processed"
    shutil.copytree(FIXTURES / "processed", processed)
    counts = ingest(REPORTS, processed)
    assert counts["wells"] == {"F-12": 7, "F-14": 1}
    return processed


def call(processed: Path, as_of: date | None = None, **args) -> dict:
    return execute(ProductionStore(processed, as_of=as_of), "well_operations", json.dumps(args))


def test_jobs_are_the_report_days_grouped(store) -> None:
    result = call(store, well="F-12")["result"]

    assert [(j["start"], j["end"], j["kind"]) for j in result["jobs"]] == [
        ("2010-09-23", "2010-09-23", "workover"),
        ("2014-11-22", "2014-11-27", "workover"),
    ]
    november = result["jobs"][1]
    assert november["perforations"] == ["3126.2-3130.7 mMD"]
    assert any("Perforated new zone" in s["summary"] for s in november["summaries"])


def test_one_day_gives_every_operation_line(store) -> None:
    result = call(store, well="F-12", day="2014-11-26")["result"]

    assert result["summary"].startswith('Installed 7" EV0 plug')
    lines = result["operations"]
    remarks = lines["rows"]
    assert len(remarks) > 10 and "remark" in lines["columns"]


def test_the_time_machine_never_sees_later_work(store) -> None:
    before = call(store, as_of=date(2014, 11, 1), well="F-12")["result"]
    missing = call(store, as_of=date(2014, 11, 1), well="F-12", day="2014-11-26")

    assert [j["start"] for j in before["jobs"]] == ["2010-09-23"]
    assert "no report" in missing["error"]


def test_a_quote_from_a_report_passes_the_checker(store) -> None:
    log = EvidenceLog()
    payload = call(store, well="F-12", start="2014-11-01", end="2014-12-31")
    eid = log.add("well_operations", {"well": "F-12"}, payload["result"], lambda: None).id
    finding = Finding(
        claim='F-12\'s water cut fell after the job of 2014-11-26: "Installed 7" EV0 plug with '
        'mid element at 3150 m. Perforated new zone from 3126,2 -3139,7 mMD."',
        evidence_ids=[eid],
        confidence="fact",
        numbers=[],
    )
    invented = finding.model_copy(
        update={"claim": 'The report says "Pumped acid into the new zone at 3150 m."'}
    )

    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []
    assert check_submission(Submission(summary="Done.", findings=[invented]), log)
