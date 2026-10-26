from collections.abc import Callable

from fastapi import FastAPI

from wellpilot import __version__
from wellpilot.api import data, runs
from wellpilot.config import Settings, get_settings
from wellpilot.llm import LLM, make_llm


def create_app(
    settings: Settings | None = None, llm_factory: Callable[[], LLM] | None = None
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="WellPilot", version=__version__)
    app.state.runs = runs.RunManager(
        settings.processed_dir,
        settings.runs_dir,
        llm_factory or (lambda: make_llm(settings)),
        max_active=settings.max_active_runs,
        default_reasoning=settings.agent_reasoning_effort,
    )
    app.include_router(runs.router)
    app.include_router(data.router)

    @app.get("/api/health")
    async def health() -> dict:
        key = (
            settings.model_api_key if settings.llm_provider == "meta" else settings.baseten_api_key
        )
        return {
            "status": "ok",
            "version": __version__,
            "llm_provider": settings.llm_provider,
            "llm_configured": bool(key),
            "data_ready": (settings.processed_dir / "production_daily.parquet").exists(),
        }

    return app


app = create_app()
