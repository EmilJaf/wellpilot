"""Report rendering, fact extraction (scripted fake LLM, no tokens) and report tools."""

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.tools import execute
from wellpilot.config import get_settings
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production import DAILY_FILE, ISSUE_COLUMNS, ISSUES_FILE
from wellpilot.ingest.report_facts import (
    FACTS_FILE,
    SUMMARIES_FILE,
    cache_path,
    combine,
    extract_page,
    parse_pages,
    quote_check,
    quote_in_text,
)
from wellpilot.ingest.reports import PAGES_FILE, REPORTS_BY_ID, normalize_text, render_report
from wellpilot.llm import LLM

PAGE_23 = (
    "Tabell 4.3 - Felt begrensninger Maksimum vannkutt 0,80 4.5.2 Prognose for vannproduksjon "
    "Vannproduksjonen kommer først og fremst fra injeksjonsvannet som bryter gjennom i "
    "brønnene etter omlag 2 år."
)
EXTRACTION = {
    "page_summary_en": "Field limits and the water production forecast.",
    "facts": [
        {
            "topic": "water",
            "statement_en": "Injection water is expected to break through after about 2 years.",
            "quote": "injeksjonsvannet som bryter gjennom i brønnene etter omlag 2 år",
            "source": "text",
            "wells": [],
        },
        {
            "topic": "facility_limits",
            "statement_en": "The maximum water cut is 0.80.",
            "quote": "Maksimum vannkutt 0,80",
            "source": "table",
            "wells": [],
        },
        {
            "topic": "wells",
            "statement_en": "F-12 will be drilled first.",
            "quote": "F-12 bores først",
            "source": "text",
            "wells": ["F-12"],
        },
    ],
}


@pytest.fixture
def processed_dir(tmp_path):
    pages = pd.DataFrame(
        {
            "doc_id": "volve_pud",
            "page": [22, 23],
            "pages": 53,
            "text": ["Innhold", PAGE_23],
            "has_text": True,
            "image": ["reports/volve_pud/p022.png", "reports/volve_pud/p023.png"],
            "width_px": 1240,
            "height_px": 1755,
        }
    )
    pages.to_parquet(tmp_path / PAGES_FILE, index=False)
    images = tmp_path / "reports" / "volve_pud"
    images.mkdir(parents=True)
    for name in ("p022.png", "p023.png"):
        (images / name).write_bytes(b"\x89PNG fake")
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=3),
            "well": "F-12",
            "oil_sm3": np.ones(3),
        }
    )
    daily.to_parquet(tmp_path / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        tmp_path / ISSUES_FILE, index=False
    )
    return tmp_path


def fake_llm(contents: list[str]) -> tuple[LLM, list[dict]]:
    requests: list[dict] = []

    async def create(**kwargs):
        requests.append(kwargs)
        content = contents[min(len(requests), len(contents)) - 1]
        return ChatCompletion.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 300, "total_tokens": 2300},
            }
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLM(client=client, model="fake", provider="fake"), requests


def test_normalize_text_joins_soft_hyphens_and_spaces() -> None:
    assert normalize_text("reser￾ voaret  er\n\ngodt") == "reservoaret er godt"
    assert normalize_text("vann-\nproduksjon") == "vannproduksjon"


def test_quote_check_tolerates_spacing_but_not_other_numbers() -> None:
    assert quote_in_text("Maksimum  vannkutt 0,80", PAGE_23)
    assert not quote_in_text("Maksimum vannkutt 0,90", PAGE_23)
    assert not quote_in_text("2 år", PAGE_23)  # too short to mean anything
    assert quote_in_text("8000 Sm³/d pr. vannprodusent", "rate på 8000 Sm/d pr. vannprodusent")
    assert quote_in_text("13000 Sm3/d", "Væskekapasitet 13000 Sm 3 /d")
    assert quote_in_text("2013 - - - 0,88", "2012 0,40 - 0,40 1,25 2013 - - - 0,88")


def test_quote_check_levels() -> None:
    page = (
        "Boringen er planlagt med oljebasert slam i 8 og 12 seksjonene. 3 produksjonsbrønner "
        "3 injeksjonsbrønner Borevæske 5673 4485 Kaks 1137 2115"
    )

    assert quote_check('oljebasert slam i 8 ½" og 12 ¼" seksjonene', page) == "exact"
    assert quote_check("3 injeksjonsbrønner Borevæske 4485 Kaks 2115", page) == "tokens"
    assert quote_check("3 injeksjonsbrønner Borevæske 9999", page) == "unverified"
    # Numbers count whole: digits that occur only inside other numbers are not on the page.
    assert quote_check("3 injeksjonsbrønner Borevæske 448", page) == "unverified"


def test_tampered_quotes_on_pud_page_23_are_not_exact() -> None:
    """The audit's swapped-number quotes. Two are caught as unverified; the other two still
    read as 'tokens' (each number is on the page, in another row), which is why only exact
    quotes reach the agent."""
    table = (
        "Tabell 4.3 - Felt begrensninger Antatt produksjonsstart mars.2007 Væskekapasitet 13000 "
        "Sm3 /d Oljekapasitet 9000 Sm3 /d Gass kapasitet 1,5 M Sm3 /d Maksimum vannkutt 0,80"
    )
    assert quote_check("Maksimum vannkutt 0,40", table) == "unverified"
    assert quote_check("bryter gjennom i brønnene etter omlag 5 år", PAGE_23) == "unverified"
    assert quote_check("Væskekapasitet 9000 Sm3/d", table) == "tokens"
    assert quote_check("Oljekapasitet 13000 Sm3/d", table) == "tokens"
    assert quote_check("Oljekapasitet 9000 Sm3/d", table) == "exact"


def test_parse_pages() -> None:
    assert parse_pages("23,25-27", 53) == [23, 25, 26, 27]
    assert parse_pages("all", 3) == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_pages("60", 53)


async def test_extract_page_sends_image_and_text_and_retries_bad_json(processed_dir) -> None:
    llm, requests = fake_llm(["not json", json.dumps(EXTRACTION)])
    page = pd.read_parquet(processed_dir / PAGES_FILE).iloc[1]

    record = await extract_page(llm, "volve_pud", page, processed_dir, reasoning_effort="low")

    assert len(requests) == 2
    assert requests[0]["reasoning_effort"] == "low"
    assert record["reasoning_effort"] == "low"
    content = requests[0]["messages"][1]["content"]
    assert "Page 23 of 53" in content[0]["text"]
    assert "Maksimum vannkutt" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert record["page"] == 23
    assert len(record["extraction"]["facts"]) == 3


def test_combine_verifies_quotes_against_the_page(processed_dir) -> None:
    path = cache_path(processed_dir, "volve_pud", 23)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"doc_id": "volve_pud", "page": 23, "model": "fake", "extraction": EXTRACTION})
    )

    facts, summaries = combine(processed_dir)

    assert facts["quote_check"].tolist() == ["exact", "exact", "unverified"]
    assert facts["quote_verified"].tolist() == [True, True, False]
    assert facts.loc[2, "wells"] == "F-12"
    assert summaries.loc[0, "facts"] == 3


def _with_facts(processed_dir):
    path = cache_path(processed_dir, "volve_pud", 23)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"doc_id": "volve_pud", "page": 23, "model": "fake", "extraction": EXTRACTION})
    )
    facts, summaries = combine(processed_dir)
    facts.to_parquet(processed_dir / FACTS_FILE, index=False)
    summaries.to_parquet(processed_dir / SUMMARIES_FILE, index=False)
    return processed_dir


def test_search_reports_ranks_facts_and_filters(processed_dir) -> None:
    store = ProductionStore(_with_facts(processed_dir))

    hits = execute(store, "search_reports", '{"query": "water break through"}')["result"]
    by_well = execute(store, "search_reports", '{"well": "F-12"}')["result"]
    by_topic = execute(store, "search_reports", '{"topic": "facility_limits"}')["result"]

    assert hits["facts"]["rows"][0][4].startswith("Injection water")
    assert "volve_pud" in hits["documents"]
    assert [r[4] for r in by_well["facts"]["rows"]] == ["F-12 will be drilled first."]
    assert by_topic["facts"]["rows"][0][5] == "Maksimum vannkutt 0,80"


def test_search_falls_back_to_page_text_before_extraction(processed_dir) -> None:
    store = ProductionStore(processed_dir)

    result = execute(store, "search_reports", '{"query": "vannkutt"}')["result"]

    assert result["pages"]["rows"][0][1] == 23
    assert "vannkutt" in result["pages"]["rows"][0][3]
    assert "error" in execute(store, "search_reports", '{"topic": "water"}')


def test_read_report_page(processed_dir) -> None:
    store = ProductionStore(_with_facts(processed_dir))

    page = execute(store, "read_report_page", '{"doc_id": "volve_pud", "page": 23}')["result"]

    assert page["summary_en"].startswith("Field limits")
    assert page["image"] == "reports/volve_pud/p023.png"
    assert len(page["facts"]["rows"]) == 3
    assert "error" in execute(store, "read_report_page", '{"doc_id": "nope", "page": 1}')


def test_a_long_report_page_reaches_the_agent_whole(processed_dir) -> None:
    """PUD p.28 runs past 2000 characters; the Ba/Sr and Utsira sentences come last."""
    pages = pd.read_parquet(processed_dir / PAGES_FILE)
    long_text = "Vanninjeksjon. " * 300 + "Barium og strontium i Utsira vannet."
    pages.loc[pages["page"] == 23, "text"] = long_text
    pages.to_parquet(processed_dir / PAGES_FILE, index=False)
    store = ProductionStore(processed_dir)

    page = execute(store, "read_report_page", '{"doc_id": "volve_pud", "page": 23}')["result"]

    assert page["text"] == long_text


def test_a_report_not_in_the_registry_is_skipped_not_fatal(processed_dir) -> None:
    pages = pd.read_parquet(processed_dir / PAGES_FILE)
    stray = pages.iloc[:1].assign(doc_id="f12_completion_report")
    pd.concat([pages, stray]).to_parquet(processed_dir / PAGES_FILE, index=False)

    store = ProductionStore(processed_dir, as_of=date(2012, 12, 31))

    ids = store.query("SELECT DISTINCT doc_id FROM report_pages").rows["doc_id"].tolist()
    assert ids == ["volve_pud"]


def test_reports_issued_after_as_of_are_hidden(processed_dir) -> None:
    store = ProductionStore(_with_facts(processed_dir), as_of=date(2004, 12, 31))

    assert store.query("SELECT count(*) AS n FROM report_facts").rows.loc[0, "n"] == 0


@pytest.mark.skipif(
    not (get_settings().raw_dir / "reports" / REPORTS_BY_ID["volve_pud"].file).exists(),
    reason="Volve PUD not downloaded",
)
def test_render_real_pud_page_text(tmp_path) -> None:
    settings = get_settings()

    pages = render_report(REPORTS_BY_ID["volve_pud"], settings.raw_dir, tmp_path)

    assert len(pages) == 53
    page_23 = pages.loc[pages["page"] == 23, "text"].iloc[0]
    assert quote_in_text("bryter gjennom i brønnene etter omlag 2 år", page_23)
    assert (tmp_path / "reports" / "volve_pud" / "p023.png").exists()
