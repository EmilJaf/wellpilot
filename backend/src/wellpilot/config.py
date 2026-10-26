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

    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite+aiosqlite:///{self.data_dir / 'wellpilot.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
