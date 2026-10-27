import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from wellpilot import __version__
from wellpilot.agent.sandbox import modal_pool
from wellpilot.api import data, redteam, runs, voice
from wellpilot.config import Settings, get_settings
from wellpilot.llm import LLM, make_llm


def create_app(
    settings: Settings | None = None, llm_factory: Callable[[], LLM] | None = None
) -> FastAPI:
    settings = settings or get_settings()
    on_fly = bool(os.environ.get("FLY_APP_NAME"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One sandbox started ahead, so a run does not wait for a cold start (about 20 s):
        # kept so on the public site; on a developer's machine, once, ending by its time limit.
        keeper = None
        if sandboxes is not None and public:
            keeper = asyncio.create_task(sandboxes.keep_warm())
        elif sandboxes is not None:
            sandboxes.warm()
        yield
        if keeper is not None:
            keeper.cancel()
        await app.state.runs.aclose()  # runs in flight are saved as interrupted

    sandboxes = modal_pool(settings.processed_dir) if settings.python_sandbox else None
    public = settings.public or on_fly
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
        voice_per_client_per_hour=settings.voice_per_client_per_hour,
        voice_max_audio_b64=settings.voice_max_audio_b64,
        attacks_per_client_per_hour=settings.attacks_per_client_per_hour,
        review_answers=settings.review_answers,
        run_timeout_s=settings.run_timeout_s,
        featured=tuple(r.strip() for r in settings.featured_runs.split(",") if r.strip()) or None,
        public=public,
        operator_key=settings.operator_key,
        trusted_proxy=("fly" if on_fly else "none") if proxy == "auto" else proxy,
        sandboxes=sandboxes,
    )
    app.include_router(runs.router)
    app.include_router(data.router)
    app.include_router(voice.router)
    app.include_router(redteam.router)

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
            # Audio input is a Muse Spark Open on Baseten feature; the mic hides otherwise.
            "voice": bool(key) and settings.llm_provider == "baseten",
            # The agent can run its own Python (Modal Sandboxes are set up).
            "python": sandboxes is not None,
            "data_ready": (settings.processed_dir / "production_daily.parquet").exists(),
        }

    _serve_frontend(app, settings.frontend_dir, https=settings.public or on_fly)
    return app


def _serve_frontend(app: FastAPI, dist: Path, https: bool = False) -> None:
    """The built single-page app, from the same origin as the API (no CORS): hashed assets
    as static files, index.html for every other path that is not the API. Link previews
    need absolute URLs, so index.html gets the site's own address where it says __ORIGIN__
    (https in public: the proxy in front speaks it, the app sees http)."""
    if not (dist / "index.html").is_file():
        return
    root = dist.resolve()
    index = (root / "index.html").read_text()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def frontend(path: str, request: Request) -> FileResponse | HTMLResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        file = (root / path).resolve()
        if path and file.is_file() and root in file.parents:  # favicon.svg, never ../
            return FileResponse(file)
        scheme = "https" if https else request.url.scheme
        origin = f"{scheme}://{request.url.netloc}"
        return HTMLResponse(
            index.replace("__ORIGIN__", origin), headers={"Cache-Control": "no-cache"}
        )


app = create_app()
