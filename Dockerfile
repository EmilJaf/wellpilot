# WellPilot: one image, the API serving the built frontend and the processed Volve data.
# Build from the repo root (the context needs data/processed: scripts/fetch_data.sh):
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t wellpilot .
# Keys are never baked in: pass them at run time (fly secrets set ...).

FROM node:24-slim AS web
WORKDIR /web
RUN npm install --global pnpm@10
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.13-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app/backend
# Dependencies first, so code changes reuse this layer.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY backend/ ./
RUN uv sync --locked --no-dev

# Volve data (Equinor Open Data Licence, attributed in the app's footer and the README).
COPY data/processed /app/data/processed
# The featured runs, frozen in git and replayed in CI; only FEATURED_RUNS are listed in
# public. Not the local data/runs: today's eval runs in it would count against the live
# site's daily token budget, which is read back from the run files at startup.
RUN mkdir -p /app/data/runs && cp /app/backend/tests/fixtures/runs/*.json /app/data/runs/
COPY --from=web /web/dist /app/frontend/dist

RUN useradd --create-home --uid 10001 wellpilot && chown -R wellpilot /app/data
USER wellpilot
# Last, so a new commit rebuilds only this layer: /api/health reports it.
ARG GIT_COMMIT=""
ENV DATA_DIR=/app/data \
    FRONTEND_DIR=/app/frontend/dist \
    PUBLIC=true \
    GIT_COMMIT=$GIT_COMMIT
EXPOSE 8080
# One worker: runs, the token budget and rate limits live in this process. A short
# graceful shutdown leaves the lifespan hook time to save runs in flight before Fly's kill.
CMD ["uvicorn", "wellpilot.main:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-graceful-shutdown", "3"]
