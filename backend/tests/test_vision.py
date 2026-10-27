"""look_at_page: the agent shows a report page image to the model during a run (no tokens:
the model is scripted)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.agent.loop import SUBMIT, run_agent
from wellpilot.agent.replay import differences, replay
from wellpilot.agent.vision import LOOK, WARNING, look, look_at_page
from wellpilot.api.runs import RunRecord
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.production import DAILY_FILE, ISSUE_COLUMNS, ISSUES_FILE
from wellpilot.ingest.reports import PAGES_FILE
from wellpilot.llm import LLM

READING = {
    "answer": "The forecast figure shows a plateau of about 8500 Sm3/d until 2010.",
    "readings": [{"label": "plateau oil rate", "value": 8500, "unit": "Sm3/d", "where": "figure"}],
}


@pytest.fixture
def processed(tmp_path):
    days = 60
    pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=days, freq="D"),
            "well": "F-12",
            "well_type": "OP",
            "flow_kind": "production",
            "on_stream_hrs": 24.0,
            "oil_sm3": 2000.0 * np.exp(-0.002 * np.arange(days)),
            "water_sm3": 100.0,
            "gas_sm3": 300_000.0,
            "water_inj_sm3": np.nan,
            "water_cut": 0.05,
            "bhp_bar": 250.0,
            "whp_bar": 55.0,
            "choke_pct": 50.0,
        }
    ).to_parquet(tmp_path / DAILY_FILE, index=False)
    pd.DataFrame(columns=ISSUE_COLUMNS).astype({"date": "datetime64[us]"}).to_parquet(
        tmp_path / ISSUES_FILE, index=False
    )
    pd.DataFrame(
        {
            "doc_id": ["volve_pud"],
            "page": [23],
            "pages": [53],
            "text": ["Figur 4.2 - Volve oljerate og kumulativ produksjon"],
            "has_text": [True],
            "image": ["reports/volve_pud/p023.png"],
            "width_px": [1240],
            "height_px": [1755],
        }
    ).to_parquet(tmp_path / PAGES_FILE, index=False)
    (tmp_path / "reports" / "volve_pud").mkdir(parents=True)
    (tmp_path / "reports" / "volve_pud" / "p023.png").write_bytes(b"\x89PNG fake")
    return tmp_path


def reply(content=None, tool_calls=()) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


class Client:
    """Answers page readings (requests with a response_format) with READING, and the
    agent's turns from a script."""

    def __init__(self, script: list[ChatCompletion], reading=READING) -> None:
        self.script = iter(script)
        self.reading = reading
        self.vision: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs) -> ChatCompletion:
        if "response_format" in kwargs:
            self.vision.append(kwargs)
            text = self.reading if isinstance(self.reading, str) else json.dumps(self.reading)
            return reply(content=text)
        return next(self.script)


ASK = {"doc_id": "volve_pud", "page": 23, "question": "What plateau rate does the figure show?"}


async def test_a_page_reading_shows_the_image_and_carries_a_warning(processed) -> None:
    client = Client([])
    llm = LLM(client=client, model="muse-open", provider="fake")

    payload = await look(ProductionStore(processed), llm, json.dumps(ASK))

    request = client.vision[0]
    content = request["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "What plateau rate does the figure show?" in content[0]["text"]
    assert request["response_format"]["json_schema"]["name"] == "page_reading"
    assert request["model"] == "muse-open"
    result = payload["result"]
    assert result["readings"][0]["value"] == 8500
    assert result["read_by"] == "muse-open"
    assert result["warnings"] == [WARNING]  # a reading off a picture is never a fact


async def test_bad_pages_and_bad_readings_are_errors_for_the_model(processed) -> None:
    store = ProductionStore(processed)
    llm = LLM(client=Client([]), model="m", provider="fake")
    garbled = LLM(client=Client([], reading="not json"), model="m", provider="fake")

    unknown = await look(store, llm, json.dumps({**ASK, "doc_id": "other"}))
    missing = await look(store, llm, json.dumps({**ASK, "page": 99}))
    invalid = await look(store, garbled, json.dumps(ASK))
    typo = await look(store, llm, json.dumps({**ASK, "pg": 23}))

    assert "unknown document" in unknown["error"]
    assert "has no page 99" in missing["error"]
    assert "not valid" in invalid["error"]
    assert "invalid arguments" in typo["error"]


def _submit(confidence: str) -> dict:
    return {
        "summary": "The plan's figure suggests a plateau near 8500 Sm3/d.",
        "summary_numbers": [{"text": "8500", "value": 8500, "evidence_id": "E1"}],
        "findings": [
            {
                "claim": "The plan's forecast figure shows a plateau of about 8500 Sm3/d.",
                "evidence_ids": ["E1"],
                "confidence": confidence,
                "numbers": [{"text": "8500", "value": 8500, "evidence_id": "E1"}],
            }
        ],
    }


SCRIPT = [
    reply("1. Read the plan's figure.", [(LOOK, ASK)]),
    reply(tool_calls=[(SUBMIT, _submit("fact"))]),
    reply(tool_calls=[(SUBMIT, _submit("inference"))]),
]


async def test_a_finding_on_a_page_reading_is_an_inference(processed) -> None:
    llm = LLM(client=Client(SCRIPT), model="muse-open", provider="fake")

    run = await run_agent("Compare F-12 with the plan.", llm, ProductionStore(processed))

    assert run.evidence[0].tool == LOOK
    assert run.evidence[0].code.startswith(f"{look_at_page.__module__}.look_at_page@")
    assert "is marked fact but rests on evidence with warnings" in run.check_rounds[0][0]
    assert run.check_rounds[1] == [] and run.verified


async def test_a_run_with_a_page_reading_replays_without_asking_the_model(processed) -> None:
    client = Client(SCRIPT)
    record = RunRecord("0123456789ab", "Compare F-12 with the plan.", None)
    run = await run_agent(
        record.question,
        LLM(client=client, model="muse-open", provider="fake"),
        ProductionStore(processed),
        on_event=record.emit,
        evidence_log=record.log,
    )
    record.complete(run)
    saved = json.loads(json.dumps(record.to_file()))

    replayed = await replay(saved, processed)

    assert len(client.vision) == 1  # only the recording asked the model about the page
    assert differences(saved, replayed) == []
