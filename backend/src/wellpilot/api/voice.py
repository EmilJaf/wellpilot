"""POST /api/transcribe: a question spoken into the microphone, as text for the ask bar.

The browser records the clip and turns it into 16 kHz mono WAV; this sends it to the model
and returns the words. It costs tokens, so it has its own per-client limit, counts against
the daily budget, and is refused without a key or on a provider that takes no audio.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from wellpilot.agent.transcribe import transcribe
from wellpilot.api.runs import Refused, RunManager, client_id
from wellpilot.llm import is_down

router = APIRouter(prefix="/api", tags=["voice"])


class VoiceRequest(BaseModel):
    audio_wav_b64: str = Field(min_length=64, max_length=4_000_000)


@router.post("/transcribe")
async def transcribe_question(body: VoiceRequest, request: Request) -> dict:
    manager: RunManager = request.app.state.runs
    if len(body.audio_wav_b64) > manager.voice_max_audio_b64:
        raise HTTPException(413, "the recording is too long; keep it under 15 seconds")
    privileged = manager.is_operator(request.headers.get("x-operator-key"))
    try:
        llm = manager.llm_factory()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    # Fireworks takes no audio: while Baseten is passed over, voice waits for it.
    if llm.provider != "baseten" or is_down("baseten"):
        raise HTTPException(503, "voice input needs Muse Spark Open on Baseten")
    try:
        manager.allow_voice(client_id(request, manager.trusted_proxy), privileged)
    except Refused as e:
        raise HTTPException(429, str(e)) from None
    try:
        transcript, tokens = await transcribe(llm, body.audio_wav_b64)
    except TimeoutError:
        raise HTTPException(
            504, "transcription timed out; try again or type the question"
        ) from None
    except ValidationError:
        raise HTTPException(502, "the model did not return a transcript") from None
    except Exception as e:  # a provider error: say the type, never the body
        raise HTTPException(502, f"transcription failed ({type(e).__name__})") from None
    manager.charge(tokens)
    return {"question": transcript.text.strip(), "language": transcript.language, "tokens": tokens}
