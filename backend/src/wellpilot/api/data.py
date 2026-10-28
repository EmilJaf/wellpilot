"""Data the dashboard plots and shows next to the agent's evidence."""

import inspect
import json
from datetime import date

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from wellpilot.agent.crosscheck import plateau
from wellpilot.agent.evidence import code_ref
from wellpilot.agent.tools import (
    TOOLS,
    NoArgs,
    OperationsArgs,
    ReportPageArgs,
    field_screen,
    field_vs_design,
    list_wells,
    plan_vs_actual,
    read_report_page,
    table,
    well_operations,
)
from wellpilot.agent.vision import VISION_TOOL
from wellpilot.analysis.inside import inside_well
from wellpilot.analysis.timeline import TIMELINE_FILE
from wellpilot.api.runs import Refused, client_id
from wellpilot.backtest import RESULTS_FILE, SUMMARY_FILE, case_detail, film
from wellpilot.data.store import ProductionStore
from wellpilot.ingest.figure import CURVE_FILE, OVERLAY_FILE
from wellpilot.ingest.log_plots import PLOTS_BY_ID, PLOTS_DIR
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


def _store(request: Request, end: date | None = None) -> ProductionStore:
    """The data, or (with end) only what an agent cut at that day could see. A date not
    cached yet costs a store build, counted against the visitor's hourly limit (429)."""
    manager = request.app.state.runs
    if end is None:
        return manager.store()
    try:
        return manager.store(
            end,
            client_id(request, manager.trusted_proxy),
            manager.is_operator(request.headers.get("x-operator-key")),
        )
    except Refused as e:
        raise HTTPException(429, str(e)) from None


@router.get("/wells")
def wells(request: Request, end: date | None = None) -> dict:
    return list_wells(_store(request, end), NoArgs())


# The views of the field take `end` like the well charts: during a time-machine answer they
# show only the data up to its date, so the page gives away nothing the agent could not see.
@router.get("/field/plan")
def field_plan(request: Request, end: date | None = None) -> dict:
    """The field's production by year against the 2005 plan: the same result the agent's
    plan_vs_actual tool returns."""
    try:
        return plan_vs_actual(_store(request, end), NoArgs())
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/field/plan-figure")
def field_plan_figure(request: Request, end: date | None = None) -> dict:
    """The plan's forecast curve as drawn in Figure 4.2 (PUD p.23), read off the picture
    (by SAM 3.1, checked by colour; python -m wellpilot.ingest.figure), with the field's
    oil rate per calendar day by month to set it against."""
    path = request.app.state.runs.processed_dir / CURVE_FILE
    if not path.exists():
        raise HTTPException(
            404, "the figure has not been read; run python -m wellpilot.ingest.figure"
        )
    monthly = (
        _store(request, end)
        .query(
            "SELECT strftime(date, '%Y-%m') AS month, sum(oil_sm3) / count(DISTINCT date) AS rate "
            "FROM production_daily GROUP BY 1 HAVING sum(oil_sm3) > 0 ORDER BY 1"
        )
        .rows
    )
    figure = json.loads(path.read_text())
    curve = figure["curve"]
    store = _store(request)
    said = []
    if "report_facts" in store.tables:
        # What the plan's text says about the same plateau, quoted where it is exact.
        said = store.query(
            "SELECT page, quote, statement_en FROM report_facts WHERE doc_id = 'volve_pud' "
            "AND quote_check = 'exact' AND (quote ILIKE 'Platåraten%' OR quote ILIKE "
            "'regularitet på 94%') ORDER BY page, quote"
        ).rows.to_dict(orient="records")
    return {
        **figure,
        "plateau": plateau(curve["month"], curve["rate"]),
        "plan_text": said,
        "actual": {
            "month": monthly["month"].tolist(),
            "rate": [round(float(v)) for v in monthly["rate"]],
            "how": "the field's oil per calendar day, by month",
        },
    }


@router.get("/field/plan-figure/image")
def field_plan_figure_image(request: Request) -> FileResponse:
    """Figure 4.2 with the curve that was read drawn over it in ink."""
    path = request.app.state.runs.processed_dir / OVERLAY_FILE
    if not path.exists():
        raise HTTPException(404, "the figure has not been read")
    return FileResponse(path, media_type="image/png")


@router.get("/field/screen")
def field_screen_view(request: Request, as_of: date | None = None) -> dict:
    """Every well ranked by what needs attention, as of a day (default: the end of the data)."""
    return field_screen(_store(request, as_of), NoArgs())


@router.get("/field/design")
def field_design(request: Request, end: date | None = None) -> dict:
    """The field by month against the plan's design limits: the field_vs_design tool's result."""
    try:
        return field_vs_design(_store(request, end), NoArgs())
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/code/{ref}")
def tool_code(ref: str) -> dict:
    """The source of the tool function an evidence item's code ref names, and whether it is
    still the code that ran (the ref's hash matches the source)."""
    name = ref.split("@", 1)[0]
    for tool in [*TOOLS, VISION_TOOL]:
        fn = tool.run
        if f"{fn.__module__}.{fn.__qualname__}" == name:
            lines, first = inspect.getsourcelines(fn)
            return {
                "ref": code_ref(fn),
                "same_as_run": code_ref(fn) == ref,
                "file": "backend/src/" + fn.__module__.replace(".", "/") + ".py",
                "line": first,
                "source": "".join(lines),
            }
    raise HTTPException(404, f"no tool function {name!r}")


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


@router.get("/field/timeline")
def field_timeline(request: Request) -> dict:
    """The field screen replayed month by month, with each producer's oil rate and the work
    done on its wells (built by python -m wellpilot.analysis.timeline)."""
    path = request.app.state.runs.processed_dir / TIMELINE_FILE
    if not path.exists():
        raise HTTPException(404, "no timeline yet; run python -m wellpilot.analysis.timeline")
    return json.loads(path.read_text())


@router.get("/wells/{well}/operations")
def well_operation_jobs(well: str, request: Request, end: date | None = None) -> dict:
    """The well's daily-report jobs (drilling, completion, work on the well), as the
    well_operations tool gives them; `end` hides later ones, as for the daily data."""
    store = _store(request, end)
    if "well_operations" not in store.tables:
        return {"well": well, "jobs": []}
    try:
        return well_operations(store, OperationsArgs(well=well))
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/wells/{well}/inside")
def well_inside(well: str, request: Request, end: date | None = None) -> dict:
    """The well along its depth: its production logs, perforated zones and the work done on
    it with what it produced before and after (analysis.inside); `end` hides later ones."""
    store = _store(request, end)
    if "production_logs" not in store.tables:
        raise HTTPException(404, "no production logs are loaded")
    try:
        return inside_well(store, well)
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/log-plots/{plot_id}/image")
def log_plot_image(plot_id: str, request: Request) -> FileResponse:
    """A production-log plot's rate track with the edges read off it drawn in red."""
    if plot_id not in PLOTS_BY_ID:
        raise HTTPException(404, f"no plot {plot_id!r}")
    path = request.app.state.runs.processed_dir / PLOTS_DIR / f"{plot_id}.png"
    if not path.exists():
        raise HTTPException(
            404, "the plot has not been read; run python -m wellpilot.ingest.log_plots"
        )
    return FileResponse(path, media_type="image/png")


@router.get("/reports/{doc_id}/pages/{page}")
def report_page(doc_id: str, page: int, request: Request) -> dict:
    try:
        return read_report_page(_store(request), ReportPageArgs(doc_id=doc_id, page=page))
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/reports/{doc_id}/pages/{page}/quotes")
def report_page_quotes(doc_id: str, page: int, request: Request) -> list[dict]:
    """Where each exact quote of the page's facts stands on the page image, for marking."""
    store = _store(request)
    if "report_facts" not in store.tables or doc_id not in REPORTS_BY_ID:
        return []
    columns = store.query("SELECT * FROM report_facts LIMIT 0").rows.columns
    if "quote_boxes" not in columns:
        return []
    rows = store.query(
        "SELECT quote, quote_boxes FROM report_facts "
        "WHERE doc_id = ? AND page = ? AND quote_check = 'exact' ORDER BY fact_no",
        [doc_id, page],
    ).rows
    return [{"quote": r.quote, "boxes": json.loads(r.quote_boxes)} for r in rows.itertuples()]


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


@router.get("/backtest")
def backtest(request: Request) -> dict:
    processed = request.app.state.runs.processed_dir
    if not (processed / SUMMARY_FILE).exists():
        raise HTTPException(404, "no backtest yet; run python -m wellpilot.backtest")
    results = pd.read_parquet(processed / RESULTS_FILE)
    return {
        "summary": json.loads((processed / SUMMARY_FILE).read_text()),
        "results": table(results),
    }


@router.get("/backtest/film/{well}")
def backtest_film(well: str, request: Request) -> dict:
    """Every forecast made for a well, in cutoff order, for the accuracy tab's film."""
    processed = request.app.state.runs.processed_dir
    if not (processed / RESULTS_FILE).exists():
        raise HTTPException(404, "no backtest yet; run python -m wellpilot.backtest")
    try:
        return film(processed, _store(request), well)
    except ValueError as e:
        raise HTTPException(404, str(e)) from None


@router.get("/backtest/cases/{well}/{cutoff}")
def backtest_case(well: str, cutoff: date, request: Request) -> dict:
    processed = request.app.state.runs.processed_dir
    if not (processed / RESULTS_FILE).exists():
        raise HTTPException(404, "no backtest yet; run python -m wellpilot.backtest")
    try:
        return case_detail(processed, _store(request), well, cutoff.isoformat())
    except ValueError as e:
        raise HTTPException(404, str(e)) from None
