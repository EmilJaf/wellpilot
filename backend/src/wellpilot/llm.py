"""One OpenAI-compatible client for both providers, plus JSON-schema helpers.

Baseten (Muse Spark Open) and the Meta Model API (Muse Spark) both speak Chat Completions;
only the base URL, key and model differ. Chat Completions is used for both so the agent
has one code path. Note: the Meta Model API accepts only tool_choice="auto", so never send
another value.
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


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "meta":
        key, key_name = settings.model_api_key, "MODEL_API_KEY"
        base_url, model, reasoning = settings.meta_base_url, settings.meta_model, ""
    else:
        key, key_name = settings.baseten_api_key, "BASETEN_API_KEY"
        base_url, model = settings.baseten_base_url, settings.baseten_model
        reasoning = settings.baseten_reasoning_effort
    if not key:
        raise RuntimeError(f"{key_name} is not set; add it to .env (see .env.example)")
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=120.0, max_retries=2)
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
