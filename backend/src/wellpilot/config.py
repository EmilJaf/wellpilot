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

    llm_provider: Literal["baseten", "meta"] = "baseten"

    baseten_api_key: str = ""
    baseten_base_url: str = "https://inference.baseten.co/v1"
    # The hackathon build of the model; Baseten's docs show the generic id without "-1026".
    baseten_model: str = "meta-models/muse-spark-open-1026"

    model_api_key: str = ""
    meta_base_url: str = "https://api.meta.ai/v1"
    meta_model: str = "muse-spark-1.3"

    # Reasoning tokens are billed as output. Page extraction is transcription, so "low"
    # saves most of its output tokens; the agent keeps the model's default unless set.
    # Muse Spark accepts minimal|low|medium|high|xhigh|max (not "none"). Empty = default.
    extract_reasoning_effort: str = "low"
    agent_reasoning_effort: str = ""

    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""
    # Public demo limits: every run costs tokens.
    max_active_runs: int = 3
    runs_per_client_per_hour: int = 6
    daily_token_budget: int = 3_000_000  # input + output, all runs, per UTC day
    # Comma-separated run ids listed by GET /api/runs. Empty lists every saved run, which
    # is for development only: in public it would show every visitor's question.
    featured_runs: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite+aiosqlite:///{self.data_dir / 'wellpilot.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
