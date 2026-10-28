"""Two Meta models on one picture: Muse Spark Open reads the plan's forecast figure, SAM 3.1
measured the same pixels at ingest, and the cross-check judges each reading within
tolerances fixed by the figure's grid (a quarter of a gridline: 250 Sm3/d, 3 months). On
SAM's real reading of PUD Figure 4.2 (plateau 8 464 Sm3/d for 19 months)."""

import json
import shutil
from pathlib import Path

import pytest
from test_vision import ASK, Client, processed  # noqa: F401  (a fixture)

from wellpilot.agent.crosscheck import cross_check, measured
from wellpilot.agent.vision import WARNING, look
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.figure import CURVE_FILE
from wellpilot.llm import LLM

SAM = Path(__file__).parent / "fixtures" / "sam" / CURVE_FILE


@pytest.fixture
def sam(tmp_path) -> dict:
    shutil.copy(SAM, tmp_path / CURVE_FILE)
    return measured(tmp_path, "volve_pud", 23)


def reading(label: str, value: float, unit: str = "Sm3/d") -> dict:
    return {"label": label, "value": value, "unit": unit, "where": "figure"}


def test_the_plateau_rate_agrees_and_its_length_does_not(sam) -> None:
    """The featured run's reading: about 8 400 Sm3/d for about 2 years. SAM: 8 464 for 19
    months. The rate is 64 apart (within 250); the length 5 months (over 3): the text's
    "om lag 19 måneder" sides with SAM."""
    rate, length = cross_check(
        [reading("plateau oil rate", 8400), reading("plateau length", 2, "years")], sam
    )

    assert sam["plateau"] == {
        "sm3_per_day": 8464,
        "first": "2008-03",
        "last": "2009-09",
        "months": 19,
    }
    assert rate["agree"] and rate["sam"] == 8464 and rate["difference"] == 64
    assert length["agree"] is False and length["sam"] == 19 and length["difference"] == 5


def test_a_rate_at_a_date_is_set_against_the_curve_that_month(sam) -> None:
    start, year, capacity = cross_check(
        [
            reading("oil rate at the start of 2010", 7000),
            reading("oil rate in 2012", 2500),
            reading("liquid capacity", 13000),
        ],
        sam,
    )

    assert start["at"] == "2010-01" and start["sam"] == 7244 and start["agree"]
    assert year["at"].startswith("2012") and year["agree"] is False
    # Nothing SAM measured to set it against: neither agreed nor not.
    assert capacity["sam"] is None and capacity["agree"] is None


def test_no_measurement_for_a_page_sam_did_not_read(tmp_path) -> None:
    shutil.copy(SAM, tmp_path / CURVE_FILE)

    assert measured(tmp_path, "volve_pud", 25) is None
    assert measured(tmp_path / "elsewhere", "volve_pud", 23) is None


async def test_a_reading_both_models_agree_on_loses_its_warning(processed) -> None:  # noqa: F811
    shutil.copy(SAM, processed / CURVE_FILE)
    agree = Client(
        [], reading={"answer": "About 8 400.", "readings": [reading("plateau oil rate", 8400)]}
    )
    differ = Client(
        [],
        reading={
            "answer": "About 8 400 for two years.",
            "readings": [reading("plateau oil rate", 8400), reading("plateau length", 2, "years")],
        },
    )
    store = ProductionStore(processed)

    agreed = (await look(store, LLM(client=agree, model="muse", provider="fake"), json.dumps(ASK)))[
        "result"
    ]
    split = (await look(store, LLM(client=differ, model="muse", provider="fake"), json.dumps(ASK)))[
        "result"
    ]

    assert agreed["warnings"] == [] and agreed["cross_check"][0]["agree"]
    assert agreed["measured_by_sam"]["plateau"]["months"] == 19
    assert split["warnings"][0] == WARNING and "plateau length" in split["warnings"][1]


async def test_only_the_measured_values_of_an_agreed_reading_back_a_fact(processed) -> None:  # noqa: F811
    """The outside review: with the readings agreeing with SAM, the warning went from the
    whole result, so a number the model only wrote in its answer ("a 9,500 Sm3/d
    plateau") passed as a fact. Only the values both models gave count as measured."""
    from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
    from wellpilot.agent.evidence import EvidenceLog
    from wellpilot.agent.vision import VISION_TOOL

    shutil.copy(SAM, processed / CURVE_FILE)
    client = Client(
        [],
        reading={
            "answer": "A plateau of about 8 400, though the text gives 9,500 Sm3/d.",
            "readings": [reading("plateau oil rate", 8400)],
        },
    )
    store = ProductionStore(processed)
    result = (
        await look(store, LLM(client=client, model="muse", provider="fake"), json.dumps(ASK))
    )["result"]
    assert result["warnings"] == []
    log = EvidenceLog()
    eid = log.add("look_at_page", ASK, result, VISION_TOOL.run).id

    def check(written: str, value: float) -> list[str]:
        finding = Finding(
            claim=f"The plan's figure shows a {written} Sm3/d plateau.",
            evidence_ids=[eid],
            confidence="fact",
            numbers=[DeclaredNumber(text=written, value=value, evidence_id=eid)],
        )
        return check_submission(Submission(summary="Done.", findings=[finding]), log)

    assert check("8,400", 8400) == []
    assert any("not in" in p for p in check("9,500", 9500))
