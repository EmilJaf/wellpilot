"""Data the dashboard plots and shows next to the agent's evidence."""

from datetime import date

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from wellpilot.agent.tools import NoArgs, ReportPageArgs, list_wells, read_report_page, table
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.reports import REPORTS_BY_ID, page_dir

router = APIRouter(prefix="/api", tags=["data"])

DAILY_COLUMNS = [
    "date",
    "flow_kind",
    "on_stream_hrs",
    "oil_sm3",
    "water_sm3",
    "gas_sm3",
    "water_inj_sm3",
    "oil_rate_per_producing_day",
    "water_cut",
    "bhp_bar",
    "whp_bar",
    "choke_pct",
]


def _store(request: Request) -> ProductionStore:
    return request.app.state.runs.store()


@router.get("/wells")
def wells(request: Request) -> dict:
    return list_wells(_store(request), NoArgs())


@router.get("/wells/{well}/daily")
def well_daily(
    well: str, request: Request, start: date | None = None, end: date | None = None
) -> dict:
    daily = _store(request).well_daily(well, start, end)
    if daily.empty:
        raise HTTPException(404, f"no daily data for well {well!r} in that window")
    hours = daily["on_stream_hrs"]
    daily["oil_rate_per_producing_day"] = (daily["oil_sm3"] * 24 / hours).where(hours >= 12)
    return table(daily[DAILY_COLUMNS])


@router.get("/reports/{doc_id}/pages/{page}")
def report_page(doc_id: str, page: int, request: Request) -> dict:
    try:
        return read_report_page(_store(request), ReportPageArgs(doc_id=doc_id, page=page))
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/reports/{doc_id}/pages/{page}/image")
def report_page_image(doc_id: str, page: int, request: Request) -> FileResponse:
    # doc_id must be a registered report and page an int, so no path can escape the dir.
    if doc_id not in REPORTS_BY_ID:
        raise HTTPException(404, f"unknown document {doc_id!r}")
    processed = request.app.state.runs.processed_dir
    path = page_dir(processed, doc_id) / f"p{page:03d}.png"
    if not path.is_file():
        raise HTTPException(404, f"{doc_id} has no page {page}")
    return FileResponse(path, media_type="image/png")
