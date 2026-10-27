import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wellpilot import __version__
from wellpilot.api import data, runs
from wellpilot.config import Settings, get_settings
from wellpilot.llm import LLM, make_llm


def create_app(
    settings: Settings | None = None, llm_factory: Callable[[], LLM] | None = None
) -> FastAPI:
    settings = settings or get_settings()
    on_fly = bool(os.environ.get("FLY_APP_NAME"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await app.state.runs.aclose()  # runs in flight are saved as interrupted

    app = FastAPI(title="WellPilot", version=__version__, lifespan=lifespan)
    proxy = settings.trusted_proxy
    app.state.runs = runs.RunManager(
        settings.processed_dir,
        settings.runs_dir,
        llm_factory or (lambda: make_llm(settings)),
        max_active=settings.max_active_runs,
        default_reasoning=settings.agent_reasoning_effort,
        runs_per_client_per_hour=settings.runs_per_client_per_hour,
        daily_token_budget=settings.daily_token_budget,
        run_timeout_s=settings.run_timeout_s,
        featured=tuple(r.strip() for r in settings.featured_runs.split(",") if r.strip()) or None,
        public=settings.public or on_fly,
        operator_key=settings.operator_key,
        trusted_proxy=("fly" if on_fly else "none") if proxy == "auto" else proxy,
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

    _serve_frontend(app, settings.frontend_dir)
    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """The built single-page app, from the same origin as the API (no CORS): hashed assets
    as static files, index.html for every other path that is not the API."""
    if not (dist / "index.html").is_file():
        return
    root = dist.resolve()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        file = (root / path).resolve()
        if path and file.is_file() and root in file.parents:  # favicon.svg, never ../
            return FileResponse(file)
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})


app = create_app()
