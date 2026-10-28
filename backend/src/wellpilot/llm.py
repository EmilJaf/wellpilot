"""One OpenAI-compatible client for every provider, plus JSON-schema helpers.

Baseten and Fireworks (Muse Spark Open) and the Meta Model API (Muse Spark) all speak Chat
Completions; only the base URL, key and model differ. Chat Completions is used for all so
the agent has one code path. Note: the Meta Model API accepts only tool_choice="auto", so
never send another value.
"""

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from wellpilot.config import Settings


@dataclass(frozen=True)
class LLM:
    client: AsyncOpenAI
    model: str
    provider: str
    default_reasoning: str = ""  # used when a caller asks for no particular effort


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
