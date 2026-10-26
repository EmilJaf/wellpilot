from fastapi import FastAPI

from wellpilot import __version__
from wellpilot.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="WellPilot", version=__version__)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "llm_provider": settings.llm_provider}

    return app


app = create_app()
