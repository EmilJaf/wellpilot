"""Ask by voice: a recorded question transcribed by the model (scripted here, no tokens)."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from wellpilot.config import Settings
from wellpilot.llm import LLM
from wellpilot.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    """No data needed: transcription touches neither the store nor the runs."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        frontend_dir=tmp_path / "no-frontend",
        runs_per_client_per_hour=100,
        daily_token_budget=10**9,
    )


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


CLIP = "UklGRg" * 40  # a WAV header's base64 start, repeated: long enough for the body model
HEARD = {"text": "Why did F-12 decline faster after 2010?", "language": "en"}


def reply(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "muse-open",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
        }
    )


class Client:
    """Records every request; refuses the first `refuse` of them with a 400."""

    def __init__(self, refuse: int = 0, content: str = json.dumps(HEARD)) -> None:
        self.requests: list[dict] = []
        self.refuse = refuse
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(kwargs)
        if len(self.requests) <= self.refuse:
            response = httpx.Response(400, request=httpx.Request("POST", "http://baseten.test"))
            raise openai.BadRequestError("unknown content type", response=response, body=None)
        return reply(self.content)


def llm_with(client: Client, provider: str = "baseten") -> LLM:
    return LLM(client=client, model="muse-open", provider=provider)


async def test_the_clip_goes_to_the_model_as_a_wav_data_url(settings) -> None:
    client = Client()
    app = create_app(settings, llm_factory=lambda: llm_with(client))
    async with client_for(app) as http:
        response = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})
        health = await http.get("/api/health")

    assert response.status_code == 200, response.text
    assert response.json() == {"question": HEARD["text"], "language": "en", "tokens": 940}
    request = client.requests[0]
    content = request["messages"][1]["content"]
    assert content[1] == {
        "type": "audio_url",
        "audio_url": {"url": f"data:audio/wav;base64,{CLIP}"},
    }
    assert request["response_format"]["json_schema"]["name"] == "transcript"
    assert request["reasoning_effort"] == "low"
    assert health.json()["voice"] is False  # the test settings have no key


async def test_a_400_on_the_data_url_falls_back_to_the_input_audio_block(settings) -> None:
    client = Client(refuse=1)
    app = create_app(settings, llm_factory=lambda: llm_with(client))
    async with client_for(app) as http:
        response = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})

    assert response.status_code == 200
    assert client.requests[1]["messages"][1]["content"][1] == {
        "type": "input_audio",
        "input_audio": {"data": CLIP, "format": "wav"},
    }


async def test_voice_is_refused_without_a_key_or_off_baseten(settings) -> None:
    def no_key() -> LLM:
        raise RuntimeError("BASETEN_API_KEY is not set")

    async with client_for(create_app(settings, llm_factory=no_key)) as http:
        missing = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})
    meta = create_app(settings, llm_factory=lambda: llm_with(Client(), provider="meta"))
    async with client_for(meta) as http:
        other = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})

    assert missing.status_code == 503 and "BASETEN_API_KEY" in missing.json()["detail"]
    assert other.status_code == 503 and "Baseten" in other.json()["detail"]


async def test_voice_has_its_own_hourly_limit_and_the_operator_skips_it(settings) -> None:
    limited = settings.model_copy(
        update={
            "voice_per_client_per_hour": 1,
            "operator_key": "s3cret",
            "voice_max_audio_b64": 300,
        }
    )
    app = create_app(limited, llm_factory=lambda: llm_with(Client()))
    async with client_for(app) as http:
        first = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})
        second = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})
        operator = await http.post(
            "/api/transcribe", json={"audio_wav_b64": CLIP}, headers={"x-operator-key": "s3cret"}
        )
        long = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP * 3})

    assert first.status_code == 200
    assert second.status_code == 429 and "voice questions per hour" in second.json()["detail"]
    assert operator.status_code == 200
    assert long.status_code == 413


async def test_a_reply_that_is_not_a_transcript_is_a_502(settings) -> None:
    app = create_app(settings, llm_factory=lambda: llm_with(Client(content="not json")))
    async with client_for(app) as http:
        response = await http.post("/api/transcribe", json={"audio_wav_b64": CLIP})

    assert response.status_code == 502 and "transcript" in response.json()["detail"]
