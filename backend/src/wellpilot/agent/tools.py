"""Tools the agent can call, with JSON schemas generated from Pydantic argument models.

Tools never raise into the loop: bad arguments, unknown wells and failed fits come back
as {"error": ...} so the model can read the message and correct itself.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from wellpilot.analysis.anomalies import (
    gauge_failures,
    mark_field_wide,
    regime_changes,
    shut_ins,
    threshold_crossing,
)
from wellpilot.analysis.decline import fit_decline, forecast
from wellpilot.data.store import ProductionStore, QueryError

SQL_ROW_LIMIT = 200


def to_jsonable(value: Any) -> Any:
    """Dates to ISO strings, numpy scalars to Python, NaN to None, floats rounded."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return round(float(value), 4) if math.isfinite(value) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return to_jsonable(frame.to_dict("records"))


class NoArgs(BaseModel):
    pass


class WellArgs(BaseModel):
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


class SqlArgs(BaseModel):
    sql: str = Field(description="One DuckDB SELECT statement.")


def _check_well(store: ProductionStore, well: str) -> None:
    known = store.query("SELECT DISTINCT well FROM production_daily ORDER BY well").rows
    names = known["well"].tolist()
    if well not in names:
        raise ValueError(f"unknown well {well!r}; known wells: {', '.join(names)}")


def list_wells(store: ProductionStore, _: NoArgs) -> list[dict]:
    rows = store.query(
        """
        SELECT well,
               string_agg(DISTINCT well_type, '/') AS well_types,
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
    return _records(rows)


def production_monthly(store: ProductionStore, args: WindowArgs) -> list[dict]:
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
        SELECT date_trunc('month', date) AS month,
               sum(oil_sm3) AS oil_sm3,
               sum(water_sm3) AS water_sm3,
               sum(gas_sm3) AS gas_sm3,
               sum(water_inj_sm3) AS water_injected_sm3,
               sum(water_sm3) / nullif(sum(oil_sm3) + sum(water_sm3), 0) AS water_cut,
               sum(on_stream_hrs) / (24 * count(*)) AS uptime,
               24 * sum(oil_sm3) / nullif(sum(on_stream_hrs), 0) AS oil_rate_per_producing_day,
               avg(bhp_bar) AS bhp_bar,
               avg(whp_bar) AS whp_bar,
               avg(choke_pct) AS choke_pct
        FROM production_daily WHERE {" AND ".join(conditions)}
        GROUP BY 1 ORDER BY 1
        """,
        params,
    ).rows
    return _records(rows)


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


def _fit_summary(fit) -> dict:
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


def fit_decline_tool(store: ProductionStore, args: DeclineArgs) -> dict:
    _check_well(store, args.well)
    result = fit_decline(store.well_daily(args.well, args.start, args.end), model=args.model)
    return to_jsonable(
        {
            "best": _fit_summary(result.best),
            "candidates": [_fit_summary(c) for c in result.candidates],
        }
    )


def forecast_tool(store: ProductionStore, args: ForecastArgs) -> dict:
    _check_well(store, args.well)
    fit = fit_decline(store.well_daily(args.well, args.start, args.end), model=args.model).best
    fc = forecast(fit, args.horizon_days)
    monthly = (
        pd.DataFrame({"p90": fc.p90, "p50": fc.p50, "p10": fc.p10}, index=fc.dates)
        .resample("MS")
        .mean()
        .rename_axis("month")
        .reset_index()
    )
    return to_jsonable(
        {
            "fit": _fit_summary(fit),
            "cumulative_oil_sm3": fc.cumulative,
            "monthly_rate_sm3_per_producing_day": monthly.to_dict("records"),
            "note": "P90 = low case, P10 = high case. Volumes = rate x historical uptime.",
        }
    )


def run_sql(store: ProductionStore, args: SqlArgs) -> dict:
    result = store.query(args.sql)
    rows = result.rows
    return {
        "rows": _records(rows.head(SQL_ROW_LIMIT)),
        "row_count": len(rows),
        "truncated": result.truncated or len(rows) > SQL_ROW_LIMIT,
    }


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
                "parameters": _clean_schema(self.args.model_json_schema()),
            },
        }


def _clean_schema(node: Any) -> Any:
    """Drop Pydantic titles and turn `X | None` into plain optional X: providers vary in
    how well they handle anyOf with null."""
    if isinstance(node, list):
        return [_clean_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    node = {k: v for k, v in node.items() if k != "title"}
    any_of = node.get("anyOf")
    if any_of and len(any_of) == 2 and {"type": "null"} in any_of:
        other = next(o for o in any_of if o != {"type": "null"})
        node = {**{k: v for k, v in node.items() if k != "anyOf"}, **other}
        if node.get("default", ...) is None:
            del node["default"]
    return {k: _clean_schema(v) for k, v in node.items()}


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
        "Monthly production for one well: volumes, water cut, uptime, oil rate per producing "
        "day, average BHP/WHP/choke. Missing gauge readings are null, not zero.",
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
        "Returns parameters, r2_log and residual_trend (< 0: falling faster than the curve "
        "at the end of the window). Choose the window after the last regime change.",
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
        + "); production_issues(date, well, column, issue, raw_value, action) logs every "
        "raw value the ingest set to missing or flagged. Returns at most "
        f"{SQL_ROW_LIMIT} rows; aggregate instead of selecting raw daily rows.",
        SqlArgs,
        run_sql,
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
        return {"result": tool.run(store, args)}
    except (ValueError, QueryError) as e:
        return {"error": str(e)}
