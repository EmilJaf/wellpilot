"""One OpenAI-compatible client for every provider, plus JSON-schema helpers.

Baseten and Fireworks (Muse Spark Open) and the Meta Model API (Muse Spark) all speak Chat
Completions; only the base URL, key and model differ. Chat Completions is used for all so
the agent has one code path. Note: the Meta Model API accepts only tool_choice="auto", so
never send another value.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from wellpilot.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLM:
    client: AsyncOpenAI
    model: str
    provider: str
    default_reasoning: str = ""  # used when a caller asks for no particular effort
    fallback: str = ""  # the provider that answers when this one cannot


# A provider that refused the key (401, 403), wants payment (402) or no longer serves the
# model (404) is passed over at once for a while; one that is failing (5xx or no connection,
# after the client's own retries) only after OUTAGE_AFTER such failures in a row, so one bad
# request does not move everyone to the fallback (audit, Oct 31).
FALLBACK_STATUS = (401, 402, 403, 404)
OUTAGE_AFTER = 3
DOWN_FOR_S = 15 * 60
_down_until: dict[str, float] = {}
_failing: dict[str, int] = {}


def is_down(provider: str) -> bool:
    """Whether a provider is being passed over now (health and voice ask)."""
    return _down_until.get(provider, 0.0) > time.monotonic()


def _unusable(provider: str, error: Exception) -> bool:
    if isinstance(error, openai.APITimeoutError):
        return False  # slow, not down: the run's own time limit handles it
    status = getattr(error, "status_code", None)
    if isinstance(error, openai.APIStatusError) and status in FALLBACK_STATUS:
        return True
    if isinstance(error, openai.APIConnectionError) or (
        isinstance(error, openai.APIStatusError) and status is not None and status >= 500
    ):
        _failing[provider] = _failing.get(provider, 0) + 1
        return _failing[provider] >= OUTAGE_AFTER
    return False


class _Completions:
    def __init__(self, routes: list[tuple[str, AsyncOpenAI, str]]):
        self._routes = routes

    async def create(self, **kwargs: Any) -> Any:
        now = time.monotonic()
        routes = [r for r in self._routes if _down_until.get(r[0], 0.0) <= now]
        routes = routes or self._routes[-1:]
        for i, (provider, client, model) in enumerate(routes):
            try:
                answer = await client.chat.completions.create(**{**kwargs, "model": model})
                _failing.pop(provider, None)
                return answer
            except openai.APIError as e:
                if not _unusable(provider, e) or i == len(routes) - 1:
                    raise
                _failing.pop(provider, None)
                _down_until[provider] = time.monotonic() + DOWN_FOR_S
                status = getattr(e, "status_code", None)
                logger.warning(
                    "%s failed (%s%s); using %s for %d min",
                    provider,
                    type(e).__name__,
                    f" {status}" if status else "",
                    routes[i + 1][0],
                    DOWN_FOR_S // 60,
                )
        raise AssertionError("unreachable")


class _Chat:
    def __init__(self, completions: _Completions):
        self.completions = completions


class FallbackClient:
    """The client the agent calls, `chat.completions.create` only: the primary provider,
    and on a key, payment or outage error the next one, with its own model id. Both serve
    the same open-weight model (Muse Spark Open on Baseten, then on Fireworks)."""

    def __init__(self, routes: list[tuple[str, AsyncOpenAI, str]]):
        self.chat = _Chat(_Completions(routes))


def provider_key(settings: Settings) -> tuple[str, str]:
    """The configured provider's API key and the name of its variable."""
    return {
        "meta": (settings.model_api_key, "MODEL_API_KEY"),
        "fireworks": (settings.fireworks_api_key, "FIREWORKS_API_KEY"),
    }.get(settings.llm_provider, (settings.baseten_api_key, "BASETEN_API_KEY"))


def make_llm(settings: Settings, *, timeout: float = 90.0) -> LLM:
    """The configured provider's client. The agent's steps answer in seconds, so 90 s per
    request (times the client's two retries) is plenty; page extraction passes more."""
    key, key_name = provider_key(settings)
    if settings.llm_provider == "meta":
        base_url, model, reasoning = settings.meta_base_url, settings.meta_model, ""
    elif settings.llm_provider == "fireworks":
        base_url, model = settings.fireworks_base_url, settings.fireworks_model
        reasoning = settings.fireworks_reasoning_effort
    else:
        base_url, model = settings.baseten_base_url, settings.baseten_model
        reasoning = settings.baseten_reasoning_effort
    if not key:
        raise RuntimeError(f"{key_name} is not set; add it to .env (see .env.example)")
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=2)
    # Muse Spark Open on Fireworks takes over from Baseten when its key or credits stop
    # (judging week runs past the hackathon's credits). No audio there: voice stays Baseten's.
    if (
        settings.llm_provider == "baseten"
        and settings.llm_fallback == "fireworks"
        and settings.fireworks_api_key
    ):
        backup = AsyncOpenAI(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
            timeout=timeout,
            max_retries=2,
        )
        routes = [("baseten", client, model), ("fireworks", backup, settings.fireworks_model)]
        return LLM(
            client=FallbackClient(routes),  # type: ignore[arg-type]
            model=model,
            provider=settings.llm_provider,
            default_reasoning=reasoning,
            fallback="fireworks",
        )
    return LLM(
        client=client, model=model, provider=settings.llm_provider, default_reasoning=reasoning
    )


def json_schema(model: type[BaseModel]) -> dict:
    """A model's JSON schema with $refs inlined and clean_schema applied, for tool
    parameters and structured output."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, list):
            return [inline(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return inline(defs[node["$ref"].rsplit("/", 1)[-1]])
        return {k: inline(v) for k, v in node.items()}

    return clean_schema(inline(schema))


def clean_schema(node: Any) -> Any:
    """Drop Pydantic titles and turn `X | None` into plain optional X: providers vary in
    how well they handle anyOf with null."""
    if isinstance(node, list):
        return [clean_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    node = {k: v for k, v in node.items() if k != "title"}
    any_of = node.get("anyOf")
    if any_of and len(any_of) == 2 and {"type": "null"} in any_of:
        other = next(o for o in any_of if o != {"type": "null"})
        node = {**{k: v for k, v in node.items() if k != "anyOf"}, **other}
        if node.get("default", ...) is None:
            del node["default"]
    return {k: clean_schema(v) for k, v in node.items()}
