"""One OpenAI-compatible client for both providers.

Baseten (Muse Spark Open) and the Meta Model API (Muse Spark) both speak Chat Completions;
only the base URL, key and model differ. Chat Completions is used for both so the agent
has one code path. Note: the Meta Model API accepts only tool_choice="auto", so never send
another value.
"""

from dataclasses import dataclass

from openai import AsyncOpenAI

from wellpilot.config import Settings


@dataclass(frozen=True)
class LLM:
    client: AsyncOpenAI
    model: str
    provider: str


def make_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "meta":
        key, key_name = settings.model_api_key, "MODEL_API_KEY"
        base_url, model = settings.meta_base_url, settings.meta_model
    else:
        key, key_name = settings.baseten_api_key, "BASETEN_API_KEY"
        base_url, model = settings.baseten_base_url, settings.baseten_model
    if not key:
        raise RuntimeError(f"{key_name} is not set; add it to .env (see .env.example)")
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=120.0, max_retries=2)
    return LLM(client=client, model=model, provider=settings.llm_provider)
