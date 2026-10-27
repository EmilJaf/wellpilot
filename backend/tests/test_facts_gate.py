"""The report facts gate on the committed real-data fixture (and on local data if present)."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from wellpilot.config import get_settings
from wellpilot.ingest.facts_gate import check
from wellpilot.ingest.report_facts import FACTS_FILE, uncovered_numbers

FIXTURE = Path(__file__).parent / "fixtures" / "processed"


def test_the_gate_names_what_falls_short_in_the_fixture() -> None:
    """The fixture holds the hosted model's facts: two misquotes, everything else in place."""
    assert check(FIXTURE) == ["2 quotes are not on their page"]


def test_a_missing_plan_row_fails_the_gate(tmp_path) -> None:
    processed = tmp_path / "processed"
    shutil.copytree(FIXTURE, processed)
    facts = pd.read_parquet(processed / FACTS_FILE)
    facts = facts[~facts["quote"].str.startswith("2011 ")]
    facts.to_parquet(processed / FACTS_FILE, index=False)

    assert "the plan's P50 rows are missing for [2011]" in check(processed)


def test_contents_page_numbers_need_no_fact() -> None:
    contents = "INNHOLD Innledning.............6 1.1 Beliggenhet..........7 Hydrater.....28"
    assert uncovered_numbers(contents, [], page=2) == []


LIVE = get_settings().processed_dir


@pytest.mark.skipif(not (LIVE / FACTS_FILE).exists(), reason="no local processed data")
def test_the_local_facts_pass_the_gate() -> None:
    assert check(LIVE) == []
