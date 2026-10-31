from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/wellpilot/config.py -> repo root. In Docker, DATA_DIR is set explicitly.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        extra="ignore",
    )

    llm_provider: Literal["baseten", "meta", "fireworks"] = "baseten"

    baseten_api_key: str = ""
    baseten_base_url: str = "https://inference.baseten.co/v1"
    # The hackathon build of the model; Baseten's docs show the generic id without "-1026".
    baseten_model: str = "meta-models/muse-spark-open-1026"

    model_api_key: str = ""
    meta_base_url: str = "https://api.meta.ai/v1"
    meta_model: str = "muse-spark-1.3"

    # A second host of the same open-weight model, on credits that do not expire: the live
    # demo's fallback if the hackathon's Baseten key stops after the deadline. No audio
    # there (the mic hides). Reasoning effort is left to the model until it has been tried.
    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_model: str = "accounts/fireworks/models/muse-spark-open"
    fireworks_reasoning_effort: str = ""
    # Who answers when Baseten refuses the key, wants payment or is down: "fireworks" (the
    # same Muse Spark Open, when FIREWORKS_API_KEY is set) or "" for none.
    llm_fallback: str = "fireworks"

    # Reasoning tokens are billed as output. Page extraction is transcription, so "low"
    # saves most of its output tokens; the agent keeps the model's default unless set.
    # Muse Spark accepts minimal|low|medium|high|xhigh|max (not "none"). Empty = default.
    extract_reasoning_effort: str = "low"
    agent_reasoning_effort: str = ""
    # Muse Spark Open always reasons and Baseten defaults it to xhigh: slow and costly for
    # an interactive agent, so the agent asks for medium there unless told otherwise.
    baseten_reasoning_effort: str = "medium"

    data_dir: Path = REPO_ROOT / "data"
    # The built frontend (pnpm build), served by the API in production. In development Vite
    # serves it instead; a missing directory just means the API serves no pages.
    frontend_dir: Path = REPO_ROOT / "frontend" / "dist"
    # Public demo limits: every run costs tokens.
    max_active_runs: int = 3
    runs_per_client_per_hour: int = 6
    daily_token_budget: int = 3_000_000  # input + output, all runs, per UTC day
    run_timeout_s: float = 600.0  # a run that takes longer stops with its last answer
    # Ask by voice: a short clip transcribed by the model. Its own hourly limit per client (a
    # clip costs a fraction of a run) and a cap on the clip, base64 (about 16 s of 16 kHz WAV).
    voice_per_client_per_hour: int = 20
    voice_max_audio_b64: int = 1_500_000
    attacks_per_client_per_hour: int = 10  # the model's attempts on the checker
    # No tokens, but CPU on a one-core machine: a reader's edits checked (0.2-0.7 s each),
    # and stores built for dates not cached yet (about 1 s each).
    checks_per_client_per_hour: int = 120
    stores_per_client_per_hour: int = 30
    python_sandbox: bool = True  # run_python in Modal Sandboxes, when Modal is set up
    # Agent SQL in a worker process under a hard memory cap (data/sql_worker.py): DuckDB's
    # own limit does not bound lists built by scalar functions.
    isolate_sql: bool = True
    # A second opinion on each accepted answer (the same model as a reviewer; about $0.01
    # and, when it raises points, one to three more agent steps).
    review_answers: bool = True
    # Sent as the X-Operator-Key header (the UI reads localStorage.operatorKey), it skips
    # the per-client and daily limits and may use one extra slot: the demo cannot be locked
    # out by strangers. Empty disables it.
    operator_key: str = ""
    # Which proxy header names the visitor: "fly" trusts Fly-Client-IP, "xff" the last
    # X-Forwarded-For hop, "none" the socket peer. "auto" is "fly" on Fly, else "none": a
    # header a visitor can set must not decide their rate limit.
    trusted_proxy: Literal["auto", "fly", "xff", "none"] = "auto"
    # A public deployment (true on Fly) lists only featured runs, never every visitor's.
    public: bool = False
    # Comma-separated run ids listed by GET /api/runs. Empty lists every saved run, which
    # is for development only: in public it would show every visitor's question.
    featured_runs: str = ""
    # The commit the image was built from (Dockerfile build arg); the deploy job checks the
    # live site reports the commit it deployed.
    git_commit: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
