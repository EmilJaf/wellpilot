"""Ask by voice: a short recording of the question, transcribed by the same model that
answers it (Muse Spark Open on Baseten takes audio as a content block, like a page image).

Baseten's audio guide sends the clip as an `audio_url` block with a data: URL; its API
reference documents an `input_audio` block instead. The guide's shape is tried first and
the reference's on a 400, so either server-side reading works.
"""

import asyncio

import openai
from pydantic import BaseModel, Field

from wellpilot.llm import LLM, json_schema

TIMEOUT_S = 60.0
SYSTEM = (
    "You transcribe a spoken question for an assistant that analyses oil and gas wells. "
    "Return the question only, in English, as one or two sentences, with well names written "
    "like F-12 or F-15 D and dates as YYYY-MM-DD where they are spoken. Do not answer it."
)


class Transcript(BaseModel):
    text: str = Field(description="The question as spoken, in English.")
    language: str = Field(description="The language heard, as an ISO 639-1 code.")


def _content(audio_b64: str, reference_shape: bool) -> list[dict]:
    block = (
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        if reference_shape
        else {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
    )
    return [{"type": "text", "text": "Transcribe this question."}, block]


async def transcribe(llm: LLM, audio_wav_b64: str) -> tuple[Transcript, int]:
    """The transcript and the tokens it cost. Raises what the client raises (a provider
    error, a timeout) or pydantic's ValidationError when the reply is not a transcript."""

    async def ask(reference_shape: bool):
        return await asyncio.wait_for(
            llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _content(audio_wav_b64, reference_shape)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "transcript", "schema": json_schema(Transcript)},
                },
                reasoning_effort="low",
            ),
            TIMEOUT_S,
        )

    try:
        response = await ask(reference_shape=False)
    except openai.BadRequestError:
        response = await ask(reference_shape=True)
    transcript = Transcript.model_validate_json(response.choices[0].message.content or "")
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return transcript, tokens
