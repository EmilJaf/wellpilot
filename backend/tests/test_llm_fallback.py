"""Muse Spark Open on Fireworks answers when Baseten refuses the key, wants payment or is down."""

import httpx
import openai
import pytest

from wellpilot import llm as L
from wellpilot.config import Settings

REQUEST = httpx.Request("POST", "https://example.test/v1/chat/completions")


def status_error(cls, status: int) -> Exception:
    return cls("refused", response=httpx.Response(status, request=REQUEST), body=None)


class FakeClient:
    """`chat.completions.create` that records its calls and raises or answers in turn."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def fresh_state():
    L._down_until.clear()
    yield
    L._down_until.clear()


async def ask(client) -> str:
    return await client.chat.completions.create(
        model="ignored", messages=[], reasoning_effort="low"
    )


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.AuthenticationError, 401),
        status_error(openai.PermissionDeniedError, 403),
        status_error(openai.APIStatusError, 402),
        status_error(openai.NotFoundError, 404),
        status_error(openai.InternalServerError, 503),
        openai.APIConnectionError(request=REQUEST),
    ],
)
async def test_falls_back_and_passes_baseten_over_for_a_while(error):
    baseten, fireworks = FakeClient(error), FakeClient("from fireworks", "again")
    client = L.FallbackClient(
        [("baseten", baseten, "b-model"), ("fireworks", fireworks, "f-model")]
    )
    assert await ask(client) == "from fireworks"
    assert fireworks.calls[0]["model"] == "f-model"
    assert fireworks.calls[0]["reasoning_effort"] == "low"
    assert await ask(client) == "again"
    assert len(baseten.calls) == 1  # passed over until DOWN_FOR_S has gone by


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.BadRequestError, 400),
        status_error(openai.RateLimitError, 429),
        openai.APITimeoutError(request=REQUEST),
    ],
)
async def test_a_bad_request_a_rate_limit_or_a_slow_reply_is_not_an_outage(error):
    baseten, fireworks = FakeClient(error), FakeClient()
    client = L.FallbackClient([("baseten", baseten, "b"), ("fireworks", fireworks, "f")])
    with pytest.raises(type(error)):
        await ask(client)
    assert not fireworks.calls
    assert "baseten" not in L._down_until


async def test_the_last_provider_raises():
    error = status_error(openai.AuthenticationError, 401)
    baseten = FakeClient(error)
    fireworks = FakeClient(status_error(openai.AuthenticationError, 401))
    client = L.FallbackClient([("baseten", baseten, "b"), ("fireworks", fireworks, "f")])
    with pytest.raises(openai.AuthenticationError):
        await ask(client)


async def test_baseten_is_tried_again_after_its_time_out(monkeypatch):
    baseten = FakeClient(status_error(openai.InternalServerError, 500), "back")
    fireworks = FakeClient("from fireworks")
    client = L.FallbackClient([("baseten", baseten, "b"), ("fireworks", fireworks, "f")])
    await ask(client)
    L._down_until["baseten"] = 0.0
    assert await ask(client) == "back"


def test_make_llm_wraps_baseten_only_with_a_fireworks_key():
    base = {"llm_provider": "baseten", "baseten_api_key": "b-key", "_env_file": None}
    plain = L.make_llm(Settings(**base))
    assert isinstance(plain.client, openai.AsyncOpenAI) and plain.fallback == ""
    wrapped = L.make_llm(Settings(**base, fireworks_api_key="f-key"))
    assert isinstance(wrapped.client, L.FallbackClient) and wrapped.fallback == "fireworks"
    assert wrapped.provider == "baseten"
    off = L.make_llm(Settings(**base, fireworks_api_key="f-key", llm_fallback=""))
    assert isinstance(off.client, openai.AsyncOpenAI)
    fireworks = L.make_llm(
        Settings(llm_provider="fireworks", fireworks_api_key="f-key", _env_file=None)
    )
    assert isinstance(fireworks.client, openai.AsyncOpenAI)
