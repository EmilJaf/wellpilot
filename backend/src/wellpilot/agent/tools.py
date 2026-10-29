"""Tools the agent can call, with JSON schemas generated from Pydantic argument models.

Tools never raise into the loop: bad arguments, unknown wells and failed fits come back
as {"error": ...} so the model can read the message and correct itself. Tables are
returned column-wise and numbers rounded, which keeps tool results (and so the prompt
the model re-reads every step) small.
"""

import ast
import json
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wellpilot.analysis.anomalies import (
    gauge_failures,
    mark_field_wide,
    regime_changes,
    shut_ins,
    threshold_crossing,
)
from wellpilot.analysis.decline import (
    DeclineFit,
    calibrated_band,
    fit_decline,
    forecast,
    hindcasts,
    p50_volume,
)
from wellpilot.analysis.screen import screen
from wellpilot.analysis.waterflood import waterflood
from wellpilot.data.store import ProductionStore, QueryError
from wellpilot.ingest.production_logs import (
    PLOTS_FILE,
    SURVEYS,
    inflow,
    inflow_below,
    water_cut_above,
)
from wellpilot.ingest.report_facts import Topic
from wellpilot.ingest.reports import REPORTS_BY_ID
from wellpilot.llm import json_schema

SQL_ROW_LIMIT = 200
MAX_TEXT_CHARS = 2000  # one string value in a tool result
# A whole tool result, as JSON. Results go back to the model on every later step, so this
# bounds what one call can add to a run's token bill.
MAX_RESULT_CHARS = 40_000
MIN_R2_LOG = 0.7
MAX_RESIDUAL_TREND = 0.05


def _round(x: float) -> float | int:
    """Whole numbers from 1000 up, 2 decimals from 1 up, 4 significant digits below, and
    float noise such as 9.8e-17 (a fitted decline of zero) as plain 0."""
    magnitude = abs(x)
    if magnitude >= 1000:
        return round(x)
    if magnitude >= 1:
        return round(x, 2)
    if magnitude < 1e-6:
        return 0.0
    return float(f"{x:.4g}")


def _cut(text: str, max_text: int | None) -> str:
    """At most max_text characters, cut at a space so no number is split in half."""
    if max_text is None or len(text) <= max_text:
        return text
    cut = text[:max_text]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_text // 2 else cut) + "…"


def to_jsonable(value: Any, max_text: int | None = MAX_TEXT_CHARS, rounded: bool = True) -> Any:
    """Dates to ISO strings, numpy scalars to Python, NaN to None, floats rounded (unless
    `rounded` is false: a record of what the model sent keeps its numbers exact), strings
    cut at max_text characters (None keeps them whole)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, max_text, rounded) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v, max_text, rounded) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            return None
        return _round(float(value)) if rounded else float(value)
    if isinstance(value, Decimal):
        return to_jsonable(float(value), max_text, rounded)
    if isinstance(value, timedelta | pd.Timedelta | np.timedelta64):
        return to_jsonable(pd.Timedelta(value).total_seconds() / 86400, max_text, rounded)  # days
    if isinstance(value, str):
        return _cut(value, max_text)
    if isinstance(value, int | float):
        return value
    # Anything else DuckDB or pandas may hand back (UUID, bytes, time...) becomes text.
    return to_jsonable(str(value), max_text)


def table(frame: pd.DataFrame) -> dict:
    """Column-wise table: column names once, then one list per row."""
    return {
        "columns": [str(c) for c in frame.columns],
        "rows": to_jsonable(frame.to_numpy(dtype=object).tolist()),
    }


class Args(BaseModel):
    """Unknown argument names are an error the model can fix ("from" for "start"), not a
    silently ignored filter."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class WellArgs(Args):
    well: str = Field(description="Well name, e.g. 'F-12'.")


class WindowArgs(WellArgs):
    start: date | None = Field(None, description="First day, YYYY-MM-DD. Optional.")
    end: date | None = Field(None, description="Last day, YYYY-MM-DD. Optional.")


class DeclineArgs(WindowArgs):
    model: Literal["auto", "exponential", "hyperbolic", "harmonic"] = Field(
        "auto", description="Arps model; 'auto' fits all three and picks the lowest AIC."
    )


class ForecastArgs(DeclineArgs):
    horizon_days: int = Field(365, ge=30, le=3650, description="Days to forecast ahead.")


class SqlArgs(Args):
    sql: str = Field(description="One DuckDB SELECT statement.")


class CalcArgs(Args):
    # A bare number evaluates to itself.
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    expression: str | None = Field(
        None,
        description="Arithmetic on numbers taken from evidence, e.g. '0.4461 / 0.0566'. "
        "Operators: + - * / ** and parentheses.",
    )
    expressions: list[str] | None = Field(
        None,
        description="Several expressions in one call, e.g. ['4207 - 1888', '1888 / 4207']: "
        "use this rather than one call per step.",
        max_length=20,
    )

    @model_validator(mode="after")
    def _one(self) -> "CalcArgs":
        if (self.expression is None) == (self.expressions is None):
            raise ValueError("give either expression or expressions")
        return self


def _check_well(store: ProductionStore, well: str) -> None:
    known = store.query("SELECT DISTINCT well FROM production_daily ORDER BY well").rows
    names = known["well"].tolist()
    if well not in names:
        raise ValueError(f"unknown well {well!r}; known wells: {', '.join(names)}")


def list_wells(store: ProductionStore, _: NoArgs) -> dict:
    rows = store.query(
        """
        SELECT well,
               string_agg(DISTINCT well_type, '/' ORDER BY well_type) AS well_types,
               min(date) FILTER (WHERE oil_sm3 > 0) AS first_oil,
               max(date) AS last_date,
               sum(oil_sm3) AS cum_oil_sm3,
               sum(water_sm3) AS cum_water_sm3,
               sum(water_inj_sm3) AS cum_water_injected_sm3,
               count(*) FILTER (WHERE flow_kind = 'production' AND on_stream_hrs > 0)
                   AS producing_days
        FROM production_daily GROUP BY well ORDER BY well
        """
    ).rows
    return table(rows)


def production_monthly(store: ProductionStore, args: WindowArgs) -> dict:
    _check_well(store, args.well)
    conditions, params = ["well = ?"], [args.well]
    if args.start:
        conditions.append("date >= ?")
        params.append(pd.Timestamp(args.start))
    if args.end:
        conditions.append("date <= ?")
        params.append(pd.Timestamp(args.end))
    rows = store.query(
        f"""
        SELECT strftime(date_trunc('month', date), '%Y-%m') AS month,
               count(*) AS days,
               count(*) FILTER (WHERE flow_kind = 'production' AND on_stream_hrs > 0)
                   AS producing_days,
               sum(oil_sm3) AS oil_sm3,
               sum(water_sm3) AS water_sm3,
               sum(gas_sm3) AS gas_sm3,
               sum(water_inj_sm3) AS water_injected_sm3,
               sum(water_sm3) / nullif(sum(oil_sm3) + sum(water_sm3), 0) AS water_cut,
               sum(on_stream_hrs) FILTER (WHERE flow_kind = 'production')
                   / (24 * count(*) FILTER (WHERE flow_kind = 'production')) AS uptime,
               24 * sum(oil_sm3)
                   / nullif(sum(on_stream_hrs) FILTER (WHERE flow_kind = 'production'), 0)
                   AS oil_rate_per_producing_day,
               avg(bhp_bar) FILTER (WHERE on_stream_hrs >= 12) AS bhp_bar,
               avg(whp_bar) FILTER (WHERE on_stream_hrs >= 12) AS whp_bar,
               avg(choke_pct) FILTER (WHERE on_stream_hrs >= 12) AS choke_pct
        FROM production_daily WHERE {" AND ".join(conditions)}
        GROUP BY 1 ORDER BY 1
        """,
        params,
    ).rows
    return table(rows)


def detect_events(store: ProductionStore, args: WellArgs) -> list[dict]:
    _check_well(store, args.well)
    daily = store.well_daily(args.well)
    events = gauge_failures(daily, store.well_issues(args.well))
    for signal in ("oil_rate", "water_cut"):
        events += regime_changes(daily, signal)
    for threshold in (0.05, 0.5, 0.8):
        crossing = threshold_crossing(daily, "water_cut", threshold)
        if crossing:
            events.append(crossing)

    # Shut-ins of every producer, so overlaps can be marked field-wide.
    producers = store.query(
        "SELECT DISTINCT well FROM production_daily WHERE oil_sm3 > 0 ORDER BY well"
    ).rows["well"]
    all_shut_ins = [e for w in producers for e in shut_ins(store.well_daily(w), min_days=7)]
    marked = mark_field_wide(events + all_shut_ins)
    mine = sorted((e for e in marked if e.well == args.well), key=lambda e: e.start)
    return [
        to_jsonable(
            {
                "kind": e.kind,
                "signal": e.signal,
                "start": e.start,
                "end": e.end,
                "summary": e.summary,
                "details": e.details,
            }
        )
        for e in mine
    ]


def fit_warnings(fit: DeclineFit) -> list[str]:
    warnings = []
    if fit.r2_log < MIN_R2_LOG:
        warnings.append(
            f"poor fit: r2_log {fit.r2_log:.3f} < {MIN_R2_LOG}. The window may span a plateau "
            "or a regime change; refit after the last regime change, or treat any decline "
            "rate from this fit as an inference."
        )
    if fit.di_per_year <= 1e-6:
        warnings.append("no decline in this window (di at 0): it is a plateau, not a decline.")
    if abs(fit.residual_trend) > MAX_RESIDUAL_TREND:
        direction = "faster" if fit.residual_trend < 0 else "slower"
        warnings.append(
            f"residual_trend {fit.residual_trend:+.3f}: by the end of the window the well "
            f"declines {direction} than the fitted curve."
        )
    return warnings


def _fit_summary(fit: DeclineFit) -> dict:
    return {
        "model": fit.model,
        "qi_sm3_per_producing_day": fit.qi,
        "di_per_year": fit.di_per_year,
        "b": fit.b,
        "window_start": fit.t0,
        "window_end": fit.end,
        "n_points": fit.n_points,
        "r2_log": fit.r2_log,
        "rmse_log": fit.rmse_log,
        "aic": fit.aic,
        "residual_trend": fit.residual_trend,
        "uptime": fit.uptime,
    }


def _production(store: ProductionStore, well: str, start=None, end=None) -> pd.DataFrame:
    """Production days only: an injector's days would distort uptime (F-5 did both)."""
    daily = store.well_daily(well, start, end)
    return daily[daily["flow_kind"] == "production"]


def fit_decline_tool(store: ProductionStore, args: DeclineArgs) -> dict:
    _check_well(store, args.well)
    result = fit_decline(_production(store, args.well, args.start, args.end), model=args.model)
    others = [c for c in result.candidates if c is not result.best]
    return to_jsonable(
        {
            "best": _fit_summary(result.best),
            "warnings": fit_warnings(result.best),
            "other_models": [
                {"model": c.model, "aic": c.aic, "r2_log": c.r2_log, "b": c.b} for c in others
            ],
        }
    )


def forecast_tool(store: ProductionStore, args: ForecastArgs) -> dict:
    _check_well(store, args.well)
    days = _production(store, args.well, args.start, args.end)
    fit = fit_decline(days, model=args.model).best
    # The forecast covers the future of the data the agent can see, whatever window the
    # fit used: from the day after the as-of date, or after the well's last day of data.
    last = pd.Timestamp(store.as_of) if store.as_of else store.well_daily(args.well)["date"].max()
    first = last + pd.Timedelta(days=1)
    fc = forecast(fit, args.horizon_days, start=first)
    # The bootstrap band shows the fit's scatter, not how wrong such forecasts turn out: an
    # 80% band from the errors of forecasts made the same way from earlier dates.
    window = (
        (pd.Timestamp(args.end or last) - pd.Timestamp(args.start)).days if args.start else None
    )
    past = hindcasts(_production(store, args.well), last, window, args.horizon_days, args.model)
    errors = [e for _, e in past]
    calibrated = calibrated_band(fc.cumulative["p50"], errors)
    # Forecasts made before this window began saw another regime (a plateau, before water
    # broke through): their errors may say little about this one.
    earlier = sum(1 for made, _ in past if args.start and made < pd.Timestamp(args.start))
    # Which Arps model to use is a choice too: the P50 each would give on this window.
    by_model = {
        c.model: p50_volume(c, first, args.horizon_days)
        for c in fit_decline(days, model="auto").candidates
    }
    monthly = (
        pd.DataFrame({"p90": fc.p90, "p50": fc.p50, "p10": fc.p10}, index=fc.dates)
        .resample("MS")
        .mean()
    )
    monthly.index = monthly.index.strftime("%Y-%m")
    return to_jsonable(
        {
            "fit": _fit_summary(fit),
            "warnings": fit_warnings(fit) + fc.warnings,
            "forecast_from": fc.dates[0],
            "forecast_to": fc.dates[-1],
            "horizon_days": args.horizon_days,
            "cumulative_oil_sm3": fc.cumulative,
            "calibrated_band_sm3": calibrated,
            "calibrated_band_note": (
                f"P90-P10 from the errors of {calibrated['past_forecasts']} forecasts made the "
                "same way from earlier dates on data up to then; use it for confidence, the "
                "bootstrap band above only reflects the fit's scatter. With "
                f"{calibrated['past_forecasts']} past forecasts such a range holds about "
                f"{calibrated['coverage']:.0%} of outcomes."
                + (
                    f" {earlier} of them were made before this fit window began, in another "
                    "regime, so the band may be too narrow."
                    if earlier
                    else ""
                )
                if calibrated
                else f"only {len(errors)} earlier forecasts to calibrate on (5 needed): no "
                "calibrated band; the bootstrap band understates the uncertainty."
            ),
            "p50_by_model_sm3": by_model,
            "p50_by_model_note": "The P50 each Arps model gives on this window: when they "
            "differ widely, the choice of model drives the forecast as much as the data does.",
            "monthly_rate_sm3_per_producing_day": table(monthly.rename_axis("month").reset_index()),
            "note": "P90 = low case, P10 = high case. Volumes = rate x historical uptime.",
        }
    )


def run_sql(store: ProductionStore, args: SqlArgs) -> dict:
    result = store.query_untrusted(args.sql)
    rows = result.rows
    truncated = result.truncated or len(rows) > SQL_ROW_LIMIT
    return {
        **table(rows.head(SQL_ROW_LIMIT)),
        # A capped count is not the count: say so instead of reporting the cap.
        "row_count": f"more than {len(rows)}" if result.truncated else len(rows),
        "truncated": truncated,
    }


_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        # Floats only: Python's big integers would let (10**100)**100 run for minutes.
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        value = _OPERATORS[type(node.op)](left, right)
        if isinstance(value, complex) or not math.isfinite(value):
            raise ValueError("result is not a finite real number")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"only numbers and + - * / ** ( ) are allowed, got {type(node).__name__}")


def calculate(_: ProductionStore, args: CalcArgs) -> dict:
    if args.expressions is not None:
        results = []
        for expression in args.expressions:
            try:
                value = _calculate(expression)
            except ValueError as e:
                raise ValueError(f"{expression!r}: {e}") from e
            results.append({"expression": expression, "value": value})
        return {"results": results}
    return {"expression": args.expression, "value": _calculate(args.expression or "")}


def _calculate(text: str) -> float:
    if len(text) > 200:
        raise ValueError("expression longer than 200 characters")
    # Thousands grouped with a comma, space or underscore ("1,133,053", "4 050"); any other
    # comma is a decimal comma, which would silently change the value.
    expression = re.sub(r"(?<=\d)[,\s\u00a0\u202f_](?=\d{3}\b)", "", text.strip())
    if "," in expression:
        raise ValueError("use '.' as the decimal separator (write 0,80 as 0.80)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse expression: {e.msg}") from e
    try:
        value = _evaluate(tree)
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
    except OverflowError as e:
        raise ValueError("number too large") from e
    return to_jsonable(float(value))


class ReportSearchArgs(Args):
    query: str | None = Field(
        None, description="Keywords in English or Norwegian, e.g. 'water breakthrough'."
    )
    topic: Topic | None = Field(None, description="Only facts with this topic.")
    well: str | None = Field(None, description="Only facts about this well, e.g. 'F-12'.")
    limit: int = Field(8, ge=1, le=30)


class ReportPageArgs(Args):
    doc_id: str = Field(description="Document id from search_reports, e.g. 'volve_pud'.")
    page: int = Field(ge=1)


def _require_reports(store: ProductionStore) -> None:
    if "report_pages" not in store.tables:
        raise ValueError("no reports ingested; run python -m wellpilot.ingest.reports")


def _documents(doc_ids) -> dict:
    return {
        d: {"title": REPORTS_BY_ID[d].title, "issued": REPORTS_BY_ID[d].doc_date}
        for d in sorted(set(doc_ids))
    }


def search_reports(store: ProductionStore, args: ReportSearchArgs) -> dict:
    _require_reports(store)
    # Match on 5-letter prefixes: "breakthrough" finds "breaks through", "injection" finds
    # "injected", "vanngjennombrudd" finds itself in any inflection.
    words = [t for t in re.findall(r"\w+", (args.query or "").casefold()) if len(t) >= 3]
    terms = list(dict.fromkeys(t[:5] for t in words))[:8]

    if "report_facts" not in store.tables:
        # Facts not extracted yet: fall back to keyword hits in the page text.
        if not terms:
            raise ValueError("report facts are not extracted yet; pass a query to search text")
        hits = " + ".join("CASE WHEN lower(text) LIKE ? THEN 1 ELSE 0 END" for _ in terms)
        rows = store.query(
            f"SELECT doc_id, page, {hits} AS score, text FROM report_pages "
            f"WHERE score > 0 ORDER BY score DESC, page LIMIT {args.limit}",
            [f"%{t}%" for t in terms],
        ).rows
        rows["snippet"] = [_snippet(text, terms) for text in rows.pop("text")]
        return {"documents": _documents(rows["doc_id"]), "pages": table(rows)}

    score = " + ".join(
        "CASE WHEN lower(statement_en || ' ' || quote) LIKE ? THEN 1 ELSE 0 END" for _ in terms
    )
    params: list = [f"%{t}%" for t in terms]
    conditions = ["score > 0"] if terms else []
    if args.topic:
        conditions.append("topic = ?")
        params.append(args.topic)
    if args.well:
        conditions.append("list_contains(string_split(wells, ', '), ?)")
        params.append(args.well)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = store.query(
        f"""
        SELECT doc_id, page, topic, source, statement_en,
               CASE WHEN quote_check = 'exact' THEN quote END AS quote,
               quote_check, wells
        FROM (SELECT *, {score or "0"} AS score FROM report_facts) {where}
        ORDER BY score DESC, doc_id, page, fact_no LIMIT {args.limit}
        """,
        params,
    ).rows
    return {"documents": _documents(rows["doc_id"]), "facts": table(rows)}


def _snippet(text: str, terms: list[str], width: int = 300) -> str:
    lower = text.casefold()
    at = min((i for t in terms if (i := lower.find(t)) >= 0), default=0)
    start = max(0, at - width // 3)
    return text[start : start + width]


def read_report_page(store: ProductionStore, args: ReportPageArgs) -> dict:
    _require_reports(store)
    if args.doc_id not in REPORTS_BY_ID:
        raise ValueError(f"unknown document {args.doc_id!r}; known: {', '.join(REPORTS_BY_ID)}")
    page = store.query(
        "SELECT page, pages, text, image FROM report_pages WHERE doc_id = ? AND page = ?",
        [args.doc_id, args.page],
    ).rows
    if page.empty:
        raise ValueError(f"{args.doc_id} has no page {args.page}")
    result = {
        "document": _documents([args.doc_id])[args.doc_id],
        "page": args.page,
        "pages": int(page.loc[0, "pages"]),
        "text": page.loc[0, "text"],  # whole: PUD pages are at most ~7k characters
        "image": page.loc[0, "image"],
    }
    if "report_facts" in store.tables:
        facts = store.query(
            "SELECT topic, source, statement_en, "
            "CASE WHEN quote_check = 'exact' THEN quote END AS quote, "
            "quote_check, wells FROM report_facts "
            "WHERE doc_id = ? AND page = ? ORDER BY fact_no",
            [args.doc_id, args.page],
        ).rows
        summary = store.query(
            "SELECT page_summary_en FROM report_summaries WHERE doc_id = ? AND page = ?",
            [args.doc_id, args.page],
        ).rows
        result["summary_en"] = summary.iloc[0, 0] if len(summary) else None
        result["facts"] = table(facts)
    return to_jsonable(result, max_text=None)


# Table 4.5 of the 2005 plan (PUD p.25): one row per year, "2010 2,14 1,49 2,13 2,82" for
# mean, P90, P50 and P10 oil in MSm3 ("-" for none). The rows are exact quotes Muse Spark
# read from the page image, checked against the PDF's text.
PLAN_DOC, PLAN_PAGE = "volve_pud", 25
PLAN_ROW = re.compile(r"^(?:\w+ )?(\d{4})((?:\s+(?:-|\d+,\d+)){4})$")


def _cells(quote: str) -> str:
    """A table row as plain words: models write cells apart with spaces or with "|"."""
    return " ".join(quote.replace("|", " ").split())


def _exact_quotes(store: ProductionStore, doc_id: str, page: int) -> pd.Series:
    """A page's quotes that stand word for word on the page."""
    return (
        store.query(
            "SELECT quote FROM report_facts "
            "WHERE doc_id = ? AND page = ? AND quote_check = 'exact'",
            [doc_id, page],
        )
        .rows["quote"]
        .dropna()
    )


def plan_vs_actual(store: ProductionStore, _: NoArgs) -> dict:
    _require_reports(store)
    plan: dict[int, list[float]] = {}
    quotes = []
    for quote in _exact_quotes(store, PLAN_DOC, PLAN_PAGE):
        match = PLAN_ROW.match(_cells(quote))
        if match:
            values = [0.0 if v == "-" else float(v.replace(",", ".")) for v in match[2].split()]
            plan[int(match[1])] = values
            quotes.append(quote.strip())
    if not plan:
        raise ValueError("the plan's production table (PUD p.25) is not in the report facts")
    actual = store.query(
        "SELECT year(date) AS year, sum(oil_sm3) / 1e6 AS oil_msm3 FROM production_daily "
        "GROUP BY 1 HAVING sum(oil_sm3) > 0 ORDER BY 1"
    ).rows
    by_year = dict(zip(actual["year"].astype(int), actual["oil_msm3"], strict=True))
    years = sorted(set(plan) | set(by_year))
    rows = [
        {
            "year": y,
            "actual_oil_msm3": by_year.get(y),
            "plan_p90_msm3": plan[y][1] if y in plan else None,
            "plan_p50_msm3": plan[y][2] if y in plan else None,
            "plan_p10_msm3": plan[y][3] if y in plan else None,
        }
        for y in years
    ]
    # The plan assumed first oil in March 2007 (PUD p.23); the field started in February
    # 2008. Calendar years compare the plan's year 2 with the field's year 1, so the field
    # also gets compared year by year from its own start.
    plan_first = min(y for y, v in plan.items() if v[2] > 0)
    actual_first = min(by_year) if by_year else plan_first
    aligned = [
        {
            "production_year": k + 1,
            "plan_year": plan_first + k,
            "plan_p50_msm3": plan[plan_first + k][2] if plan_first + k in plan else None,
            "actual_year": actual_first + k,
            "actual_oil_msm3": by_year.get(actual_first + k),
        }
        for k in range(max(max(plan) - plan_first, max(by_year, default=0) - actual_first) + 1)
    ]
    plan_end = max(y for y, v in plan.items() if v[2] > 0)
    after = {y: v for y, v in by_year.items() if y > plan_end}
    late_wells = store.query(
        "SELECT well, min(date) AS first_oil FROM production_daily WHERE oil_sm3 > 0 "
        "GROUP BY well HAVING year(min(date)) > ? ORDER BY 2",
        [plan_end],
    ).rows
    return to_jsonable(
        {
            "unit": "million Sm3 (MSm3) of oil per calendar year, whole field",
            "plan_source": f"{REPORTS_BY_ID[PLAN_DOC].title}, page {PLAN_PAGE}, Table 4.5",
            "plan_rows_quoted": list(dict.fromkeys(quotes)),
            "years": table(pd.DataFrame(rows)),
            "by_production_year": table(pd.DataFrame(aligned)),
            "totals_msm3": {
                "actual": sum(by_year.values()),
                "plan_p90": sum(v[1] for v in plan.values()),
                "plan_p50": sum(v[2] for v in plan.values()),
                "plan_p10": sum(v[3] for v in plan.values()),
            },
            "plan_p50_last_year": plan_end,
            "actual_last_year": max(by_year) if by_year else None,
            # A last year cut by the data (as-of runs) is partial, not a collapse.
            "actual_data_ends": store.query("SELECT max(date) AS d FROM production_daily").rows.loc[
                0, "d"
            ],
            "actual_after_plan_p50_ends_msm3": sum(after.values()),
            "wells_first_oil_after_plan_p50_ends": table(late_wells),
            "note": "years: calendar years. The plan assumed first oil in March 2007 and the "
            "field started in February 2008, so by_production_year compares each year of "
            "production with the plan's year of the same number (plan 2007 with field 2008).",
        }
    )


DESIGN_PAGE = 23
# The plan's field constraints table (PUD p.23, Table 4.3): a label, the value as printed.
DESIGN_LIMITS = (
    (
        "liquid_sm3_per_day",
        re.compile(r"^Væskekapasitet\s+([\d.]+)\s*Sm3"),
        "field liquid capacity",
    ),
    ("oil_sm3_per_day", re.compile(r"^Oljekapasitet\s+([\d.]+)\s*Sm3"), "field oil capacity"),
    ("water_cut", re.compile(r"^Maksimum vannkutt\s+(\d+,\d+)"), "maximum water cut"),
    (
        "water_inj_sm3_per_day",
        re.compile(r"^Vanninjeksjons kapasitet\s+([\d.]+)\s*Sm3"),
        "field water injection capacity",
    ),
)


def _plan_number(text: str) -> float:
    """A number as the plan prints it: "0,80" is 0.8, "13000" and "13.000" are 13000."""
    return float(text.replace(",", ".")) if "," in text else float(text.replace(".", ""))


def field_vs_design(store: ProductionStore, _: NoArgs) -> dict:
    _require_reports(store)
    facts = _exact_quotes(store, PLAN_DOC, DESIGN_PAGE)
    limits = {}
    for key, pattern, label in DESIGN_LIMITS:
        for quote in facts:
            if match := pattern.match(_cells(quote)):
                limits[key] = {"limit": label, "value": _plan_number(match[1]), "quote": quote}
                break
    if not limits:
        raise ValueError("the plan's design limits (PUD p.23) are not in the report facts")
    if "liquid_sm3_per_day" in limits and "water_cut" in limits:
        # The maximum water cut is a share at full liquid rate: the water the platform was
        # built to handle is the two together. A 0.80 share at a lower liquid rate is not
        # over capacity.
        limits["water_sm3_per_day"] = {
            "limit": "field water handling (derived: liquid capacity x maximum water cut)",
            "value": limits["liquid_sm3_per_day"]["value"] * limits["water_cut"]["value"],
            "quote": None,
        }
    months = store.query(
        """
        WITH daily AS (
            SELECT date, sum(coalesce(oil_sm3, 0)) AS oil, sum(coalesce(water_sm3, 0)) AS water,
                   sum(coalesce(water_inj_sm3, 0)) AS inj
            FROM production_daily GROUP BY date
        )
        SELECT strftime(date_trunc('month', date), '%Y-%m') AS month, count(*) AS days,
               sum(oil) / count(*) AS oil_sm3_per_day,
               sum(oil + water) / count(*) AS liquid_sm3_per_day,
               sum(water) / nullif(sum(oil + water), 0) AS water_cut,
               sum(water) / count(*) AS water_sm3_per_day,
               sum(inj) / count(*) AS water_inj_sm3_per_day,
               max(oil + water) AS max_day_liquid_sm3,
               max(water) AS max_day_water_sm3
        FROM daily GROUP BY 1 HAVING sum(oil) > 0 ORDER BY 1
        """
    ).rows
    crossings = {}
    for key, limit in limits.items():
        column = key if key in months else None
        if column is None:
            continue
        above = months[months[column] >= limit["value"]]
        crossings[key] = {
            "first_month_at_or_above": above["month"].iloc[0] if len(above) else None,
            "months_at_or_above": len(above),
            "max": months[column].max(),
            "max_month": months.loc[months[column].idxmax(), "month"],
        }
    if "water_sm3_per_day" in crossings:
        worst = months.loc[months["max_day_water_sm3"].idxmax()]
        crossings["water_sm3_per_day"]["max_day"] = worst["max_day_water_sm3"]
        crossings["water_sm3_per_day"]["max_day_month"] = worst["month"]
    economic = [q for q in facts if "minimum økonomisk rate" in q]
    return to_jsonable(
        {
            "scope": "whole field, all producers and injectors summed per day, then per month",
            "plan_source": f"{REPORTS_BY_ID[PLAN_DOC].title}, page {DESIGN_PAGE}, Table 4.3",
            "design_limits": limits,
            "field_against_limits": crossings,
            "months": table(months),
            "plan_end_quote": economic[0] if economic else None,
            "note": "These are field and platform limits (PUD p.35 lists the same platform "
            "capacities); compare them with field totals, never one well. The maximum water cut "
            "is a share at full liquid rate: the water the platform can handle is liquid "
            "capacity x maximum water cut (water_sm3_per_day). The field reaching that share at "
            "a lower liquid rate means it ran at the design water cut, not over its water "
            "capacity. Rates are per calendar day of each month.",
        }
    )


def field_screen(store: ProductionStore, _: NoArgs) -> dict:
    return to_jsonable(screen(store))


class OperationsArgs(WindowArgs):
    day: date | None = Field(
        None, description="One report day, YYYY-MM-DD: every line of that day's operations."
    )


GAP_DAYS = 3  # reports this close together belong to one job


def well_operations(store: ProductionStore, args: OperationsArgs) -> dict:
    """The well's daily drilling and well-work reports, grouped into jobs."""
    _check_well(store, args.well)
    if "well_operations" not in store.tables:
        raise ValueError("no daily operation reports are loaded")
    if args.day is not None:
        lines = store.query(
            'SELECT start, "end", depth_mmd, activity, sub_activity, remark '
            "FROM well_operation_steps WHERE well = ? AND date = ? ORDER BY start",
            [args.well, pd.Timestamp(args.day)],
        ).rows
        day = store.query(
            "SELECT summary, planned, perforations FROM well_operations "
            "WHERE well = ? AND date = ?",
            [args.well, pd.Timestamp(args.day)],
        ).rows
        if day.empty:
            raise ValueError(f"no report for {args.well} on {args.day}")
        return {
            "well": args.well,
            "day": args.day.isoformat(),
            **{k: v for k, v in day.iloc[0].items() if v},
            "operations": table(lines),
        }
    where, params = ["well = ?"], [args.well]
    if args.start:
        where.append("date >= ?")
        params.append(pd.Timestamp(args.start))
    if args.end:
        where.append("date <= ?")
        params.append(pd.Timestamp(args.end))
    days = store.query(
        "SELECT date, contractor, summary, activities, perforations FROM well_operations "
        f"WHERE {' AND '.join(where)} ORDER BY date",
        params,
    ).rows
    jobs: list[dict] = []
    for _, row in days.iterrows():
        last = jobs[-1] if jobs else None
        if last and (row["date"] - pd.Timestamp(last["end"])).days <= GAP_DAYS:
            job = last
        else:
            job = {"start": row["date"], "days": [], "activities": set(), "contractors": set()}
            jobs.append(job)
        job["end"] = row["date"].date().isoformat()
        job["days"].append(row)
        job["activities"] |= {a.strip() for a in str(row["activities"]).split(",") if a.strip()}
        if row["contractor"]:
            job["contractors"].add(row["contractor"])
    out = []
    for job in jobs:
        # What a job was: plug and abandonment ends a well; then drilling, completion, work
        # on a producing well. Report codes as written ("plug abandon").
        kinds = {
            "plug abandon": "abandonment",
            "drilling": "drilling",
            "completion": "completion",
            "workover": "workover",
        }
        kind = next((v for k, v in kinds.items() if k in job["activities"]), "other")
        days_ = job["days"]
        # A drilling job runs for months; its first and last days say what it was.
        shown = days_ if kind != "drilling" or len(days_) <= 6 else [*days_[:3], *days_[-3:]]
        out.append(
            {
                "start": job["start"].date().isoformat(),
                "end": job["end"],
                "kind": kind,
                "report_days": len(days_),
                "contractor": ", ".join(sorted(job["contractors"])),
                "perforations": [d["perforations"] for d in days_ if d["perforations"]],
                "summaries": [
                    {"date": d["date"].date().isoformat(), "summary": d["summary"]}
                    for d in shown
                    if d["summary"]
                ],
            }
        )
    return {
        "well": args.well,
        "jobs": out,
        "note": "Daily reports exist for drilling and for work on the well (wireline, plugs, "
        "perforations, production logs), not for days of normal production. A job's date is "
        "the day the work was done; kind is drilling, completion, workover (work on a "
        "producing or injecting well) or abandonment. Ask for one day to see every operation "
        "line.",
    }


class LogArgs(WellArgs):
    below_m: list[float] = Field(
        default_factory=list,
        max_length=6,
        description="Depths in m MD (e.g. a plug's depth from well_operations): the share of "
        "each pass's oil and water entering below each.",
    )


PROFILE_STEP_M = 10.0


def _how_read(store: ProductionStore, survey: str, read_by: str) -> dict:
    """Where a survey's numbers come from and what checked them: its LAS files (and its plot,
    read the same way as a picture-only survey, against them), or its plots, read by SAM."""
    path = store.processed_dir / PLOTS_FILE
    plots = (
        [p for p in json.loads(path.read_text()) if p["survey"] == survey] if path.exists() else []
    )
    if read_by == "LAS":
        las = next(s for s in SURVEYS if s.survey == survey)
        out: dict = {"read_from": f"LAS files of the interpreted log ({las.source})"}
        if plots:
            out["check"] = {
                "its_plot_read_by": next(iter(plots[0]["phases"].values()))["read_by"],
                "plot_against_files_median_share_gap": {
                    p["pass"]: {
                        k.split("_")[0]: v["median_share_gap"]
                        for k, v in p["checks"].items()
                        if k.endswith("_against_las")
                    }
                    for p in plots
                },
            }
        return out
    first = plots[0]
    oil = first["phases"]["oil"]
    return {
        "read_from": f"plots in {first['source']['document']} (pages "
        f"{', '.join(str(p['source']['page']) for p in plots)}), pictures with no data "
        f"behind them, read by {oil['read_by']}"
        + (f" from the prompt {oil['prompt']!r}" if "prompt" in oil else ""),
        "rate_unit": first["rate_unit"],
        "check": {
            **(
                {
                    "agreement_with_colour_trace_median_share_gap": {
                        p["pass"]: p["phases"]["oil"]["agreement_with_colour"]["median_share_gap"]
                        for p in plots
                    }
                }
                if "agreement_with_colour" in oil
                else {}
            ),
            "upper_two_zones_share": {
                p["pass"]: p["checks"]["upper_two_zones_share"] for p in plots
            },
            "report_says": first["checks"]["report_says"],
            "perforated_zones_m": first["checks"]["zones_m"],
        },
    }


def _depth_work(store: ProductionStore, well: str, surveys: list[dict]) -> dict:
    """What has been done along the well's depth up to the store's date: its perforated
    intervals (a survey report's zones and the perforations its daily reports name), the
    blank pipe between them, each plug already set with the well before and after it, and
    how old the latest log is against the water cut now."""
    from wellpilot.analysis.inside import PLUG, _back_on
    from wellpilot.analysis.inside import _production as rates

    zones = [
        [float(a), float(b)]
        for s in surveys
        for a, b in ((s.get("check") or {}).get("perforated_zones_m") or [])
    ]
    plugs = []
    if "well_operations" in store.tables:
        for job in well_operations(store, OperationsArgs(well=well))["jobs"]:
            if job["kind"] in ("drilling", "abandonment"):
                continue
            zones += [
                [float(a), float(b)]
                for interval in job["perforations"]
                for a, b in re.findall(r"(\d{4}(?:\.\d+)?)-(\d{4}(?:\.\d+)?)", interval)
            ]
            # Each plug on the day its report says it was set (a job may run for days).
            set_on = [
                (d["date"], depth) for d in job["summaries"] for depth in PLUG.findall(d["summary"])
            ]
            for day, depth in set_on:
                start, end = pd.Timestamp(job["start"]), pd.Timestamp(job["end"])
                back = _back_on(store, well, end)
                plugs.append(
                    {
                        "date": day,
                        "depth_m": float(depth.replace(",", ".")),
                        "window_days": 60,  # before the job, and from the well's return
                        "before": rates(store, well, start - pd.Timedelta(days=60), start),
                        "after": rates(store, well, back, back + pd.Timedelta(days=60))
                        if back is not None
                        else None,
                    }
                )
    zones = sorted({(min(z), max(z)) for z in zones})
    # A completion report's overall span ("3130-3281 mMD") over the zones a log report
    # names one by one is the gun's span, not one open interval: the zones stand.
    zones = [
        z for z in zones if sum(1 for y in zones if y != z and z[0] <= y[0] and y[1] <= z[1]) < 2
    ]
    blank = [[a[1], b[0]] for a, b in zip(zones, zones[1:], strict=False) if b[0] > a[1]]
    since = None
    if surveys:
        last = surveys[-1]
        now = store.query(
            "SELECT max(date) AS d FROM production_daily WHERE well = ? AND on_stream_hrs > 0",
            [well],
        ).rows.loc[0, "d"]
        if pd.notna(now):
            now = pd.Timestamp(now)
            cut = store.query(
                "SELECT median(water_cut) AS w FROM production_daily WHERE well = ? "
                "AND date BETWEEN ? AND ? AND on_stream_hrs >= 12",
                [well, now - pd.Timedelta(days=30), now],
            ).rows.loc[0, "w"]
            since = {
                "latest_survey": last["date"],
                "days": int((now - pd.Timestamp(last["date"])).days),
                "water_cut_then": last["production_30_days_before"]["water_cut"],
                "water_cut_now": None if pd.isna(cut) else float(cut),
                "as_of": now.date().isoformat(),
            }
    return {
        "perforated_m": [list(z) for z in zones],
        "blank_pipe_m": blank,
        "plugs_set": plugs,
        "since_survey": since,
    }


def _after_plugs(survey: dict, plugs: list[dict], passes: dict) -> None:
    """A plug set after a survey has already shut off what entered below it. On that log a
    depth at or below the plug (or anywhere in the blank pipe around it) cuts nothing new, and
    a depth above it cuts only the flow between the two: of_flow_left gives that as a share of
    what the log measured above the plug. F-12's 2010 log was run the day before its 3210 m
    plug, so its shares below 3210 m were gone by 2014."""
    later = [p for p in plugs if p["date"] > survey["date"]]
    if not later:
        return
    plug = min(p["depth_m"] for p in later)  # the shallowest shuts off the most
    survey["plugs_set_since"] = [{"date": p["date"], "depth_m": p["depth_m"]} for p in later]
    for p in survey["passes"]:
        gone = inflow_below(passes[p["pass"]], plug)
        if not gone:
            continue
        for row in p["below"]:
            row["already_isolated"] = row["depth_m"] >= plug
            row["of_flow_left"] = _left(row["depth_m"], row, plug, gone)
        for gap in p["at_blank_pipe"]:
            # Blank pipe has no inflow, so a gap that holds the plug, or lies below it, adds
            # nothing to what the plug already shuts off.
            gap["already_isolated"] = gap["blank_pipe_m"][1] >= plug
            gap["of_flow_left_below_top"] = _left(
                gap["blank_pipe_m"][0], gap["below_top"], plug, gone
            )


def _left(depth: float, shares: dict, plug: float, gone: dict) -> dict:
    """What a plug at `depth` would cut of the flow left above an old plug at `plug`, which
    already shut off `gone` of each phase."""
    if depth >= plug:
        return {k: 0.0 for k in gone}
    return {
        k: round(max(0.0, (shares[k] - gone[k]) / (1 - gone[k])), 3) if gone[k] < 1 else 0.0
        for k in gone
        if k in shares
    }


def production_log(store: ProductionStore, args: LogArgs) -> dict:
    """Where along the well the oil and the water came in, from its production logs."""
    _check_well(store, args.well)
    if "production_logs" not in store.tables:
        raise ValueError("no production logs are loaded")
    logs = store.query(
        "SELECT survey, date, pass, read_by, depth_m, oil, water FROM production_logs "
        "WHERE well = ? AND pass != 'shut-in' ORDER BY date, "
        "CASE pass WHEN 'low rate' THEN 0 WHEN 'medium rate' THEN 1 ELSE 2 END, depth_m",
        [args.well],
    ).rows
    if logs.empty:
        known = store.query("SELECT DISTINCT well FROM production_logs ORDER BY well").rows
        held = ", ".join(known["well"]) or "none"
        raise ValueError(f"no production log for {args.well} in the data; logs exist for: {held}")
    plots = store.processed_dir / PLOTS_FILE
    separator = (
        {
            (p["survey"], p["pass"]): p["separator_oil_sm3_per_day"]
            for p in json.loads(plots.read_text())
        }
        if plots.exists()
        else {}
    )
    surveys, survey_logs_by = [], {}
    for name, survey_logs in logs.groupby("survey", sort=False):
        survey_logs_by[name] = dict(tuple(survey_logs.groupby("pass", sort=False)))
        day = survey_logs["date"].iloc[0]
        las = next((s for s in SURVEYS if s.survey == name), None)
        passes, profiles = [], {}  # flowing passes, logged at the well's surface rates
        for pass_name, log in survey_logs.groupby("pass", sort=False):
            shares = inflow(log, PROFILE_STEP_M)
            profiles[pass_name] = {row[0]: row[1:] for row in shares["from_below"]["rows"]}
            columns = shares.pop("from_below")["columns"][1:]
            below = []
            for depth in args.below_m:
                share = inflow_below(log, depth)
                if share is not None:
                    below.append({"depth_m": depth, **share})
            surface = (
                {"surface_liquid_sm3_per_day": las.surface_sm3_per_day.get(pass_name)}
                if las
                else {"surface_oil_sm3_per_day": separator.get((name, pass_name))}
            )
            passes.append({"pass": pass_name, **surface, **shares, "below": below})
        depths = sorted(set.intersection(*(set(p) for p in profiles.values())))
        names = list(profiles)
        profile = pd.DataFrame(
            [[d, *[v for n in names for v in profiles[n][d]]] for d in depths],
            columns=["depth_m", *[f"{n.split()[0]}_{c}" for n in names for c in columns]],
        )
        month = store.query(
            "SELECT median(water_cut) AS water_cut, "
            "median(oil_sm3 * 24 / on_stream_hrs) AS oil_rate FROM production_daily "
            "WHERE well = ? AND date BETWEEN ? AND ? AND on_stream_hrs >= 12",
            [args.well, day - pd.Timedelta(days=30), day - pd.Timedelta(days=1)],
        ).rows.iloc[0]
        surveys.append(
            {
                "survey": name,
                "date": day.date().isoformat(),
                **_how_read(store, name, survey_logs["read_by"].iloc[0]),
                "passes": passes,
                "profile_from_below": table(profile),
                "production_30_days_before": {
                    "water_cut": month["water_cut"],
                    "oil_sm3_per_producing_day": month["oil_rate"],
                },
            }
        )
    work = _depth_work(store, args.well, surveys)
    for survey in surveys:
        passes = survey_logs_by[survey["survey"]]
        before = survey["production_30_days_before"]
        for p in survey["passes"]:
            # A pass run well below the well's own rate may weight its zones differently.
            liquid, oil = p.get("surface_liquid_sm3_per_day"), p.get("surface_oil_sm3_per_day")
            well_oil, well_cut = before["oil_sm3_per_producing_day"], before["water_cut"]
            if liquid and well_oil and well_cut is not None and well_cut < 1:
                p["rate_vs_well"] = round(liquid / (well_oil / (1 - well_cut)), 2)
            elif oil and well_oil:
                p["rate_vs_well"] = round(oil / well_oil, 2)
            for row in p["below"]:
                inside = next(
                    (z for z in work["perforated_m"] if z[0] < row["depth_m"] < z[1]), None
                )
                if inside:
                    row["inside_perforation_m"] = inside
            # Where a bridge plug can go: the shares below each end of every blank-pipe gap,
            # whether or not the model asked for those depths.
            log = passes[p["pass"]]
            p["at_blank_pipe"] = [
                {"blank_pipe_m": gap, "below_top": top, "below_bottom": bottom}
                for gap in work["blank_pipe_m"]
                if (top := inflow_below(log, gap[0])) and (bottom := inflow_below(log, gap[1]))
            ]
            # What a plug would leave flowing: the water cut of the flow entering above it.
            if (cut := p.get("water_cut_at_top")) is not None:
                for row in p["below"]:
                    if "water_share" in row:
                        row["water_cut_left_above"] = water_cut_above(cut, row)
                for gap in p["at_blank_pipe"]:
                    if "water_share" in gap["below_top"]:
                        gap["water_cut_left_above"] = water_cut_above(cut, gap["below_top"])
        _after_plugs(survey, work["plugs_set"], passes)
    return to_jsonable(
        {
            "well": args.well,
            "surveys": surveys,
            **work,
            "note": "A production log measures the flow passing each depth while the well "
            "flows; going up, it grows past each zone that produces. A share at a depth is "
            "the part of the pass's oil or water entering below that depth (1 at the bottom of "
            "the well's inflow, 0 above the top zone). comes_in_from_m to to_m holds 90% of "
            "each phase's inflow. Only shares are given: the files' rates have no units, and "
            "the plots' are downhole volumes. A plug set at a depth shuts off what enters "
            "below it: the shares below say what it could remove, not what it did; "
            "plugs_set gives what each plug already set did to the well (water cut and oil "
            "before and after). A bridge plug is set in blank pipe, never inside a perforated "
            "interval (inside_perforation_m names the interval a depth falls in; blank_pipe_m "
            "lists the gaps, and each pass's at_blank_pipe gives the shares below the top and "
            "the bottom of every gap: the depths a plug can actually be set at). A plug set "
            "after a survey (its plugs_set_since) already shut off what entered below it: on "
            "that log a depth or gap at or below it is already_isolated (a plug there cuts "
            "nothing new), and of_flow_left gives what a shallower plug would cut as a share "
            "of the flow left above the old plug. water_cut_left_above is the water cut of the "
            "flow entering above a depth on that pass, what a plug there would leave flowing "
            "(for a gap, a plug anywhere in it): compare it with the well's water cut now to "
            "say what a plug would change. rate_vs_well is the pass's surface rate over the "
            "well's own in the month before: a pass run far below it may weight the zones "
            "differently. since_survey "
            "says how old the latest log is and how far the "
            "water cut has moved since: an old log may no longer show where the water enters, "
            "and a new one would settle it. Each survey's passes were run at several surface "
            "rates. A survey read off pictures carries the checks of that reading.",
        }
    )


def waterflood_tool(store: ProductionStore, args: WindowArgs) -> dict:
    _check_well(store, args.well)
    result = waterflood(store, args.well, args.start, args.end)
    for key in ("monthly", "by_year", "field_voidage"):
        result[key] = table(result[key])
    return to_jsonable(result)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ProductionStore, Any], Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json_schema(self.args),
            },
        }


DAILY_COLUMNS = (
    "date, well, wellbore, npd_wellbore_code, flow_kind ('production'|'injection'), "
    "well_type ('OP'|'WI'), on_stream_hrs, oil_sm3, gas_sm3, water_sm3, water_inj_sm3, "
    "bhp_bar, bht_c, dp_tubing_bar, whp_bar, wht_c, annulus_press_bar, choke_pct, "
    "dp_choke_bar, water_cut, gor_sm3_sm3"
)

TOOLS = [
    Tool(
        "list_wells",
        "List all wells with type, first oil, last date, cumulative volumes, producing days.",
        NoArgs,
        list_wells,
    ),
    Tool(
        "production_monthly",
        "Monthly production for one well: days in the data and producing days, volumes, water "
        "cut, uptime, oil rate per producing day, and BHP/WHP/choke averaged over days on "
        "stream at least 12 h. A month with fewer days than the calendar month is partial "
        "(the data is cut there, or the well starts or stops), so its volumes are not a "
        "collapse. Missing gauge readings are null, not zero. Pass start and end to keep the "
        "result small.",
        WindowArgs,
        production_monthly,
    ),
    Tool(
        "detect_events",
        "Detect events for one well: gauge failures, shut-ins (marked field-wide when other "
        "wells were shut in too), regime changes in oil rate and water cut, and water cut "
        "crossing 5%, 50%, 80%. Each event has a summary and the numbers behind it. An oil "
        "regime change gives the mean of the periods before and after it (relative_change) "
        "and the change across it, 4 weeks each side (change_across): call it a step only "
        "when the change across is large; a large change between periods with a small one "
        "across is a gradual shift (water taking over, depletion), not a step. It also gives "
        "the choke 4 weeks each side: choke_moved true means the operator cut or opened the "
        "choke at the same time, so part of the change is the operator's, not the "
        "reservoir's. A regime change's date is where the data seen so far splits best: with "
        "data cut at an earlier date it can move by weeks, so name it as 'around' that date.",
        WellArgs,
        detect_events,
    ),
    Tool(
        "fit_decline",
        "Fit Arps decline curves to one well's oil rate per producing day within a window. "
        "Returns parameters, r2_log, residual_trend and warnings when the fit is not "
        "trustworthy. Choose the window after the last regime change.",
        DeclineArgs,
        fit_decline_tool,
    ),
    Tool(
        "forecast_production",
        "Fit a decline curve on a window and forecast oil with P90/P50/P10 bands.",
        ForecastArgs,
        forecast_tool,
    ),
    Tool(
        "run_sql",
        "Run one read-only DuckDB SELECT. Tables: production_daily("
        + DAILY_COLUMNS
        + "); production_issues(date, well, field, issue, raw_value, action) logs every "
        "raw value the ingest set to missing or flagged. Returns at most "
        f"{SQL_ROW_LIMIT} rows; aggregate instead of selecting raw daily rows.",
        SqlArgs,
        run_sql,
    ),
    Tool(
        "calculate",
        "Evaluate arithmetic on numbers from evidence (ratios, differences, changes). Use it "
        "for every derived number you state, so the number has evidence; put all you need "
        "in one call with 'expressions'. Not needed to write a share as a percent: declare "
        "'69%' with the evidence's 0.69.",
        CalcArgs,
        calculate,
    ),
    Tool(
        "search_reports",
        "Search facts extracted from the field's reports, such as the February 2005 "
        "development plan (PUD, in Norwegian): planned plateau rates, expected water "
        "breakthrough, facility limits such as maximum water cut. Each fact has an English "
        "statement, the verbatim original quote, and quote_check: 'exact' (the quote is on "
        "the page as written), 'tokens' (its words and numbers are on the page but not in "
        "that order, e.g. a table row) or 'unverified' (not found on the page). Only exact "
        "quotes are given; for the others, use the English statement. Use it to compare what "
        "happened with what was planned.",
        ReportSearchArgs,
        search_reports,
    ),
    Tool(
        "plan_vs_actual",
        "The whole field's oil production per year against the 2005 development plan's "
        "forecast (P90/P50/P10, Table 4.5 on PUD page 25, quoted rows), with totals, the "
        "plan's last producing year and what was produced after it. Use it to compare the "
        "field's life with the plan.",
        NoArgs,
        plan_vs_actual,
    ),
    Tool(
        "waterflood",
        "Waterflood diagnostics for one producer: WOR (water/oil) against cumulative oil by "
        "month with a fitted log10(WOR) trend extended to water cuts 0.80, 0.90 and 0.95 (the "
        "oil left before each), liquid rate, choke and water cut by year, and the field's "
        "injection against production by year. On a waterflood, oil often falls because water "
        "takes the well's capacity, not because pressure is lost: use this alongside the "
        "decline fit. Optional start/end (YYYY-MM-DD) bound the trend window. Without a "
        "reliable trend, wor_fit is null and no_trend says why; the tables are still data.",
        WindowArgs,
        waterflood_tool,
    ),
    Tool(
        "field_screen",
        "Every well at once, ranked by what needs attention as of the end of the data: its "
        "recent oil rate, water cut, uptime and 90-day change, and issues found by the "
        "detectors (a dead downhole gauge, water cut above the platform's 0.80 design limit, "
        "a steep recent decline, long shut-ins, a step down in rate, a stopped well). Use it "
        "for questions about the field or which wells to look at.",
        NoArgs,
        field_screen,
    ),
    Tool(
        "field_vs_design",
        "The whole field's monthly oil and liquid rates, water cut and water injection against "
        "the design limits of the 2005 development plan (Table 4.3 on PUD page 23, quoted): "
        "liquid and oil capacity, maximum water cut, water injection capacity, with the first "
        "month each was reached. These are field and platform limits: compare them with the "
        "field, not with one well.",
        NoArgs,
        field_vs_design,
    ),
    Tool(
        "well_operations",
        "What was done to a well, from its daily drilling and well-work reports: drilling, "
        "completion and later jobs such as a plug set to shut off water, a new zone "
        "perforated or a production log, grouped into jobs with each day's summary and "
        "perforated intervals. A sudden change in a well's rate or water cut often follows "
        "such a job: check here before reading it as reservoir behaviour. Optional start/end "
        "(YYYY-MM-DD) bound the dates; 'day' returns every operation line of one report.",
        OperationsArgs,
        well_operations,
    ),
    Tool(
        "production_log",
        "Where along a well the oil and the water came in, from the production logs (PLT) "
        "run in it, each survey dated: for each flowing pass, the depths each phase enters "
        "between, the share of each entering below every 10 m, and below depths you give (a "
        "plug's depth from well_operations shows what that plug could shut off). Comparing "
        "surveys shows how the inflow changed, e.g. zones that gave oil and later water. Use "
        "it to explain a water shut-off or a new perforation, or why a well makes water. A "
        "survey kept only as pictures was read off its plots; it says so and carries the "
        "checks of that reading. Logs exist only for some wells; the error names them.",
        LogArgs,
        production_log,
    ),
    Tool(
        "read_report_page",
        "Read one report page: its text (original language), English summary and facts.",
        ReportPageArgs,
        read_report_page,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS]


def execute(store: ProductionStore, name: str, arguments: str) -> dict:
    """Run a tool call from the model. Always returns {"result": ...} or {"error": ...}."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}; available: {', '.join(TOOLS_BY_NAME)}"}
    try:
        args = tool.args.model_validate_json(arguments or "{}")
    except ValidationError as e:
        return {"error": f"invalid arguments for {name}: {e}"}
    try:
        result = tool.run(store, args)
    except (ValueError, QueryError) as e:
        return {"error": str(e)}
    except Exception as e:  # e.g. a fit that does not converge: report it, keep the run
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}
    size = len(json.dumps(result, ensure_ascii=False, default=str))
    if size > MAX_RESULT_CHARS:
        return {
            "error": f"{name} returned {size} characters, over the {MAX_RESULT_CHARS} limit. "
            "Narrow it: fewer rows or columns, a shorter window, or aggregate in SQL."
        }
    return {"result": result}
