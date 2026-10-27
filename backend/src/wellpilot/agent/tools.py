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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wellpilot.analysis.anomalies import (
    gauge_failures,
    mark_field_wide,
    regime_changes,
    shut_ins,
    threshold_crossing,
)
from wellpilot.analysis.decline import B_BOUNDS, DeclineFit, fit_decline, forecast
from wellpilot.analysis.screen import screen
from wellpilot.analysis.waterflood import waterflood
from wellpilot.data.store import ProductionStore, QueryError
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


def to_jsonable(value: Any, max_text: int | None = MAX_TEXT_CHARS) -> Any:
    """Dates to ISO strings, numpy scalars to Python, NaN to None, floats rounded, strings
    cut at max_text characters (None keeps them whole)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, max_text) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v, max_text) for v in value]
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
        return _round(float(value)) if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return to_jsonable(float(value))
    if isinstance(value, timedelta | pd.Timedelta | np.timedelta64):
        return to_jsonable(pd.Timedelta(value).total_seconds() / 86400)  # days
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

    expression: str = Field(
        description="Arithmetic on numbers taken from evidence, e.g. '0.7532 / 0.0575'. "
        "Operators: + - * / ** and parentheses."
    )


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
    if fit.model == "hyperbolic" and fit.b >= B_BOUNDS[1] - 1e-6:
        warnings.append(
            f"b is at its upper bound {B_BOUNDS[1]:g}: the window does not constrain the "
            "decline's curvature, so the forecast and its band are not reliable."
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
    fit = fit_decline(_production(store, args.well, args.start, args.end), model=args.model).best
    # The forecast covers the future of the data the agent can see, whatever window the
    # fit used: from the day after the as-of date, or after the well's last day of data.
    last = pd.Timestamp(store.as_of) if store.as_of else store.well_daily(args.well)["date"].max()
    fc = forecast(fit, args.horizon_days, start=last + pd.Timedelta(days=1))
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
    if len(args.expression) > 200:
        raise ValueError("expression longer than 200 characters")
    # Thousands grouped with a comma, space or underscore ("1,133,053", "4 050"); any other
    # comma is a decimal comma, which would silently change the value.
    expression = re.sub(r"(?<=\d)[,\s\u00a0\u202f_](?=\d{3}\b)", "", args.expression.strip())
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
    return {"expression": args.expression, "value": to_jsonable(float(value))}


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
            "note": "The plan assumed first oil in March 2007; years are calendar years.",
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
               sum(inj) / count(*) AS water_inj_sm3_per_day,
               max(oil + water) AS max_day_liquid_sm3
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
            "capacities). A single well's water cut above 0.80 does not breach them; the field's "
            "water cut does. Rates are per calendar day of each month.",
        }
    )


def field_screen(store: ProductionStore, _: NoArgs) -> dict:
    return to_jsonable(screen(store))


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
        "crossing 5%, 50%, 80%. Each event has a summary and the numbers behind it.",
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
        "Evaluate arithmetic on numbers from evidence (ratios, differences, percentages). "
        "Use it for every derived number you state, so the number has evidence.",
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
        "decline fit. Optional start/end (YYYY-MM-DD) bound the trend window.",
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
