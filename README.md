# WellPilot

An AI agent that helps oil and gas engineers analyze well performance, from raw field data to explainable forecasts.

You ask a question in plain language, for example "Why did well F-12 decline faster after 2014?". The agent plans the analysis and runs real tools over the field data: decline curve fitting, forecasting and anomaly detection. It reads the well reports, checks its own results, and returns a dashboard. Every conclusion traces back to the source data and to the code that actually ran.

The agent runs on **Muse Spark Open**, Meta's open-weight model.

> Work in progress for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026).

## Built during the hackathon

Everything in this repository was built during the hackathon, starting from the first commit on Oct 26, 2026. No code was carried over from earlier projects. Third-party open-source libraries are listed in `backend/pyproject.toml` and `frontend/package.json`.

## Data

Contains data from the Volve field dataset, made available by Equinor and the former Volve license partners (ExxonMobil Exploration & Production Norway AS, Bayerngas Norge AS) under the [Equinor Open Data Licence](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf). The data is provided "as is". The data stays under that licence, not under this repository's code license. This project is not affiliated with or endorsed by Equinor or the Volve license partners.

Dataset page: https://www.equinor.com/energy/volve-data-sharing

## Development

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), Node.js, pnpm.

```bash
cp .env.example .env                   # fill in the API keys
git config core.hooksPath .githooks    # scan for secrets before each commit (needs gitleaks)

cd backend
uv sync
uv run uvicorn wellpilot.main:app --reload   # API on :8000

cd frontend
pnpm install
pnpm dev                                     # UI on :5173, proxies /api to :8000
```

Data: download these files from the Volve dataset:
- `Production_data/Volve production data.xlsx` into `data/raw/production/`;
- `Reports/Volve PUD .pdf` into `data/raw/reports/Volve_PUD.pdf`.

Then build the processed data:

```bash
cd backend
uv run python -m wellpilot.ingest.production     # cleaned production Parquet
uv run python -m wellpilot.ingest.reports        # report pages as PNG + text
uv run python -m wellpilot.ingest.report_facts   # facts read by Muse Spark (calls the LLM)
```

Tests and lint:

```bash
cd backend && uv run pytest && uv run ruff check .
cd frontend && pnpm lint && pnpm build
```

## Deploy

One Docker image: the API serves the built frontend and the processed data, so the app runs as one process on one machine (runs, the token budget and the rate limits live in its memory). The configuration targets [Fly.io](https://fly.io).

Build and smoke-test the image locally first. No API key goes in, so no tokens are spent:

```bash
docker build -t wellpilot:smoke .
scripts/smoke_image.sh
```

The image is built from the repository root and copies `data/processed` and `data/runs`, so run the ingest first. Then, once:

```bash
brew install flyctl
fly auth login
fly launch --copy-config --no-deploy        # creates the app; it may pick another name
fly secrets set BASETEN_API_KEY=...          # the model key, never committed
fly secrets set OPERATOR_KEY=$(openssl rand -hex 24)   # lifts the public limits for you
fly secrets set FEATURED_RUNS=fcfb4188de3b   # the runs the public site lists and replays
fly deploy
```

In the browser console of the deployed site, `localStorage.operatorKey = '<the operator key>'` lets your own runs skip the per-visitor and daily limits.

Public limits (settings in `backend/src/wellpilot/config.py`, overridable with Fly secrets): 3 runs at once, 6 runs per visitor per hour, 3M tokens per day, 600 s per run, 14 steps per run. Agent-written SQL is checked before it runs, and the list of runs shows only `FEATURED_RUNS`.

## License

Code: [MIT](LICENSE). Volve data: the Equinor Open Data Licence, see [Data](#data).
