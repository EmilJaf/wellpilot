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

## License

Code: [MIT](LICENSE). Volve data: the Equinor Open Data Licence, see [Data](#data).
