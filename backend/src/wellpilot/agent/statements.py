"""Answers written as statements, not prose (design D in docs/checker-redesign.md).

The model never types a number, a date or a unit. It picks values out of the tool
results by reference ("E1/wells[well=F-12]/water_cut") and a statement kind; code prints
the sentence from the values and their own labels. A wrong pick is then a true sentence
about another value, never a false one. Free words join the statements and may hold no
digits, dates, units or quote marks; a well name only of a well the statements are about.

Behind a flag (run_agent(statements=True), eval --statements) for a feasibility run: the
rendered answer becomes an ordinary Submission, so the answer panel, the second opinion
and the reveal work as they are.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from wellpilot.agent.checker import (
    CAUSAL,
    HEDGE,
    MAX_CLAIM_NUMBERS,
    WELL,
    DeclaredNumber,
    Finding,
    Submission,
    _well_key,
    claim_numbers,
)
from wellpilot.agent.evidence import Evidence, EvidenceLog

Kind = Literal["value", "change", "compare", "event", "forecast", "quote"]


class Part(BaseModel):
    say: str | None = Field(
        None,
        description="Free words joining statements and saying what they mean ('P-2 needs "
        "attention most:', ', while'). No digits, dates, month names, units or quote marks: "
        "those come from refs. A well name only of a well a statement here is about.",
    )
    kind: Kind | None = Field(
        None,
        description="A statement printed from tool results: value (one value with its "
        "labels), change (ref -> to, the same quantity), compare (ref against another value "
        "in the same unit), event (a record: gauge failure, shut-in, regime change, screen "
        "issue, well-work job), forecast (a forecast_production result), quote (a report "
        "fact's exact quote with its English statement).",
    )
    ref: str | None = Field(
        None,
        description="Where the value is: an evidence id, then field names and row "
        "selectors, e.g. 'E1/wells[well=F-12]/water_cut', 'E7/monthly[month=2010-12]/"
        "water_cut', 'E5[kind=regime_change,signal=oil_rate,start=2010-05-02]', 'E13', "
        "'E4/facts[0]'. [n] picks a row by its place (from 0).",
    )
    to: str | None = Field(None, description="change: the value it changed to.")
    against: str | None = Field(None, description="compare: the value it is compared with.")
    how: Literal["ratio", "difference"] | None = Field(
        None, description="compare: ratio (times) or difference (above or below)."
    )
    form: Literal["clause", "phrase"] | None = Field(
        None,
        description="value: 'clause' (F-12's water cut was 69% ...) or 'phrase' (F-12's water "
        "cut of 69% ...), to fit the free words around it. Default clause.",
    )


class StatementFinding(BaseModel):
    parts: list[Part] = Field(min_length=1, description="Statements and free words, in order.")
    confidence: Literal["fact", "inference"] = Field(
        description="'fact' if the statements show it directly; 'inference' for an "
        "interpretation, a cause, a forecast, or any value from a result with warnings."
    )


class StatementSubmission(BaseModel):
    summary: list[Part] = Field(
        min_length=1, description="The answer to the question, as statements and free words."
    )
    findings: list[StatementFinding] = Field(min_length=1)


class BadRef(ValueError):
    pass


# ---- references -------------------------------------------------------------------------

STEP = re.compile(r"/?([A-Za-z_][\w]*)?((?:\[[^\]]*\])*)")
SELECTOR = re.compile(r"\[([^\]]*)\]")
DATE_KEYS = ("date", "month", "year", "start", "since", "as_of", "end")


@dataclass
class Cell:
    """A value found by a reference, with what its path says about it."""

    ref: str
    evidence: Evidence
    value: Any
    keys: list[str] = field(default_factory=list)  # field names walked, in order
    rows: list[dict] = field(default_factory=list)  # each record or row the path selected
    parent: Any = None  # the record the last field was read from

    @property
    def key(self) -> str:
        return self.keys[-1] if self.keys else ""

    @property
    def record(self) -> dict | None:
        """The nearest selected record that names a kind (an event, an issue, a job)."""
        return next((r for r in reversed(self.rows) if "kind" in r), None)

    @property
    def well(self) -> str | None:
        for row in reversed(self.rows):
            if isinstance(row.get("well"), str):
                return row["well"]
        top = self.evidence.result
        if isinstance(top, dict) and isinstance(top.get("well"), str):
            return top["well"]
        well = self.evidence.arguments.get("well")
        return well if isinstance(well, str) else None


def _rows(node: Any) -> list | None:
    if isinstance(node, dict) and isinstance(node.get("columns"), list) and "rows" in node:
        return [dict(zip(node["columns"], row, strict=False)) for row in node["rows"]]
    return node if isinstance(node, list) else None


def _select(node: Any, selector: str, where: str) -> dict:
    rows = _rows(node)
    if rows is None:
        raise BadRef(f"{where} is not a list of rows, so [{selector}] cannot pick one")
    if selector.strip().isdigit():
        n = int(selector)
        if n >= len(rows):
            raise BadRef(f"{where} has {len(rows)} rows: [{n}] is past the end")
        return rows[n]
    want = {}
    for pair in selector.split(","):
        if "=" not in pair:
            raise BadRef(f"[{selector}] in {where}: write field=value pairs or a row number")
        k, v = pair.split("=", 1)
        want[k.strip()] = v.strip()
    hits = [
        r for r in rows if isinstance(r, dict) and all(_same(r.get(k), v) for k, v in want.items())
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        k, v = next(iter(want.items()))
        seen = sorted({str(r.get(k)) for r in rows if isinstance(r, dict) and r.get(k) is not None})
        sample = (
            ", ".join(seen[:3] + (["..."] if len(seen) > 6 else []) + seen[-3:]) if seen else ""
        )
        known = ", ".join(sorted({c for r in rows if isinstance(r, dict) for c in r})[:12])
        hint = f"its {k} values: {sample}" if seen else f"its fields: {known}"
        raise BadRef(f"no row of {where} has {selector} ({hint})")
    others = sorted({c for r in hits for c in r if c not in want and isinstance(r[c], str)})
    raise BadRef(
        f"{len(hits)} rows of {where} match [{selector}]: add a field to tell them apart "
        f"({', '.join(others[:6])})"
    )


def _same(value: Any, text: str) -> bool:
    if value is None:
        return text in ("null", "None", "")
    if isinstance(value, bool):
        return str(value).lower() == text.lower()
    if isinstance(value, int | float):
        try:
            return math.isclose(float(value), float(text))
        except ValueError:
            return False
    return str(value) == text


def resolve(ref: str, log: EvidenceLog) -> Cell:
    """Follow a reference into the evidence; raise BadRef with what is there instead."""
    ref = (ref or "").strip()
    m = re.match(r"(E\d+)", ref)
    if not m:
        raise BadRef(f"'{ref}': a reference starts with an evidence id such as E3")
    item = log.get(m[1])
    if item is None:
        raise BadRef(f"'{ref}': there is no evidence {m[1]}")
    cell = Cell(ref, item, item.result)
    rest, where = ref[m.end() :], m[1]
    while rest:
        step = STEP.match(rest)
        if not step or not step.group(0):
            raise BadRef(f"'{ref}': cannot read '{rest}'")
        rest = rest[step.end() :]
        name, selectors = step[1], SELECTOR.findall(step[2] or "")
        node = cell.value
        if name:
            if not isinstance(node, dict):
                raise BadRef(f"{where} is not a record, so it has no field '{name}'")
            if name not in node and isinstance(node.get("columns"), list) and "rows" in node:
                raise BadRef(_table_hint(node, where, name))
            if name not in node:
                fields = ", ".join(k for k in node if not k.endswith("note"))
                raise BadRef(f"{where} has no field '{name}'; its fields: {fields}")
            cell.parent = node
            cell.value = node[name]
            cell.keys.append(name)
            where = f"{where}/{name}"
        for selector in selectors:
            row = _select(cell.value, selector, where)
            cell.rows.append(row)
            cell.value = row
            where = f"{where}[{selector}]"
    return cell


def _table_hint(node: dict, where: str, name: str) -> str:
    """A table is read a row first: say so with an example made of its own first row."""
    columns = [str(c) for c in node["columns"]]
    first = (node["rows"] or [[None]])[0]
    key, column = (
        columns[0],
        name
        if name in columns
        else next((c for c in columns[1:] if c not in ("days", "producing_days")), columns[-1]),
    )
    example = f"{where}[{key}={first[0]}]/{column}"
    told = f"'{name}' is one of its columns" if name in columns else f"it has no field '{name}'"
    return (
        f"{where} is a table ({told}): pick its row first, then the column, e.g. "
        f"'{example}'. Its columns: {', '.join(columns)}"
    )


def number(cell: Cell) -> float:
    v = cell.value
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        what = "a record" if isinstance(v, dict) else "a list" if isinstance(v, list) else repr(v)
        raise BadRef(f"{cell.ref} is {what}, not a number")
    return float(v)


# ---- what a value is ----------------------------------------------------------------------

SHARE = re.compile(r"(^|_)(water_cut|uptime|share|fraction|coverage|wc)(_|$)")
SIGNALS = {
    "oil_rate": ("oil rate", "rate"),
    "water_cut": ("water cut", "share"),
    "bhp_bar": ("downhole pressure", "bar"),
    "whp_bar": ("wellhead pressure", "bar"),
    "choke_pct": ("choke opening", "pct"),
    "on_stream_hrs": ("hours on stream", "hours"),
}
SIDES = {
    "before_mean": "average {q} before {d}",
    "after_mean": "average {q} after {d}",
    "across_before": "{q} over the 4 weeks before {d}",
    "across_after": "{q} over the 4 weeks after {d}",
    "choke_before_pct": "choke opening over the 4 weeks before {d}",
    "choke_after_pct": "choke opening over the 4 weeks after {d}",
    "rate_before": "oil rate in the 30 days before {d}",
    "rate_after": "oil rate in the 30 days after {d}",
}
PAIRS = {
    "before_mean": "after_mean",
    "across_before": "across_after",
    "choke_before_pct": "choke_after_pct",
    "rate_before": "rate_after",
}
NAMES = {
    "oil_sm3": ("oil", "Sm3"),
    "water_sm3": ("water", "Sm3"),
    "cum_oil_sm3": ("cumulative oil", "Sm3"),
    "cum_water_sm3": ("cumulative water", "Sm3"),
    "cum_water_injected_sm3": ("cumulative water injected", "Sm3"),
    "field_oil_sm3": ("the field's cumulative oil", "Sm3"),
    "oil_rate_per_producing_day": ("oil rate per producing day", "rate"),
    "oil_sm3_per_day": ("oil rate", "rate"),
    "water_sm3_per_day": ("water rate", "rate"),
    "liquid_sm3_per_day": ("liquid rate", "rate"),
    "liquid_rate_sm3_per_day": ("liquid rate", "rate"),
    "choke_pct": ("choke opening", "pct"),
    "wor": ("water-oil ratio", "ratio"),
    "zero_days": ("days the gauge read zero", "days"),
    "flowing_days": ("of them flowing", "days"),
    "oil_sm3_during": ("oil produced while it read zero", "Sm3"),
    "last_good_value": ("last good reading", "bar"),
    "months_after_first_oil": ("months after first oil", "months"),
    "producing_days": ("producing days", "days"),
    "di_per_year": ("decline rate of the fit", "per_year"),
    "qi_sm3_per_producing_day": ("starting rate of the fit", "rate"),
    "actual_oil_msm3": ("actual oil", "MSm3"),
    "plan_p50_msm3": ("plan's central estimate of oil", "MSm3"),
    "plan_p90_msm3": ("plan's low case of oil", "MSm3"),
    "plan_p10_msm3": ("plan's high case of oil", "MSm3"),
}


@dataclass
class Meaning:
    words: str  # what the value is, without the well: "median water cut over the 90 days to ..."
    unit: str  # share, pct, rate, Sm3, MSm3, days, months, hours, bar, per_year, ratio, plain
    when: str = ""  # " in 2010-12", " on 2010-12-31", "" when the words hold the date
    declares: list[tuple[str, float]] = field(default_factory=list)  # numbers `when` holds
    field: bool = False  # a field-wide value


def _unit_of(key: str) -> str:
    if SHARE.search(key):
        return "share"
    for suffix, unit in (
        ("_sm3_per_day", "rate"),
        ("_per_producing_day", "rate"),
        ("_msm3", "MSm3"),
        ("_sm3", "Sm3"),
        ("_pct", "pct"),
        ("_bar", "bar"),
        ("_days", "days"),
        ("_m", "m"),
        ("_per_year", "per_year"),
    ):
        if key.endswith(suffix):
            return unit
    return "plain"


def _plain(key: str) -> str:
    key = re.sub(r"_(sm3_per_day|per_producing_day|msm3|sm3|pct|bar|days|per_year)$", "", key)
    return key.replace("_", " ")


def _when(cell: Cell) -> str:
    for row in reversed(cell.rows):
        for k in ("date", "month", "year"):
            if row.get(k) is not None:
                return f" on {row[k]}" if k == "date" else f" in {row[k]}"
    return ""


def _around(cell: Cell) -> tuple[str, list[tuple[str, float]]]:
    """What a record's before and after windows lie around: a plug's job (production_log's
    plugs_set: before the job, and from the well's return) or the record's date."""
    row = cell.rows[-1]
    date = row.get("date") or row.get("start") or ""
    depth = row.get("depth_m")
    if isinstance(depth, int | float) and not isinstance(depth, bool):
        shown = _round(float(depth))
        return f"the job that set the plug at {shown} m on {date}", [(shown, float(depth))]
    return str(date), []


def meaning(cell: Cell) -> Meaning:
    """The value's own labels, from the tool, its path and the record around it."""
    key, tool, top = cell.key, cell.evidence.tool, cell.evidence.result
    record = cell.record
    parent = cell.keys[-2] if len(cell.keys) > 1 else ""
    as_of = top.get("as_of") if isinstance(top, dict) else None
    if tool == "field_screen" and as_of:
        window = f"over the {top.get('window_days', 90)} days to {as_of}"
        screen = {
            "oil_rate_sm3_per_day": ("median oil rate " + window, "rate"),
            "water_cut": ("median water cut " + window, "share"),
            "uptime": ("uptime " + window, "share"),
            "oil_rate_change_90d": ("change in median oil rate over the last 90 days", "change"),
            "share_of_field_oil": (f"share of the field's oil to {as_of}", "share"),
            "cum_oil_sm3": (f"cumulative oil to {as_of}", "Sm3"),
        }
        if key in screen:
            return Meaning(*screen[key])
    if record and record.get("kind") == "regime_change" and parent == "details":
        q, unit = SIGNALS.get(record.get("signal", ""), (record.get("signal", ""), "plain"))
        d = record.get("start", "")
        if key in SIDES:
            if key.startswith("choke"):
                unit = "pct"
            return Meaning(SIDES[key].format(q=q, d=d), unit)
        if key == "relative_change":
            return Meaning(f"change in average {q} from the period before {d} to after", "change")
        if key == "change_across":
            return Meaning(
                f"change in {q} across {d} (4 weeks each side)",
                "points" if unit == "share" else "change",
            )
    if record and key in SIDES and parent == "details":
        return Meaning(
            SIDES[key].format(q="", d=record.get("start", "")).replace("  ", " "), "rate"
        )
    if tool == "forecast_production" and isinstance(top, dict):
        span = f"from {top.get('forecast_from')} to {top.get('forecast_to')}"
        if parent == "cumulative_oil_sm3":
            case = {
                "p50": "central estimate",
                "p90": "low end of the fit's band",
                "p10": "high end of the fit's band",
            }
            return Meaning(f"forecast oil {span}, {case.get(key, key)}", "Sm3")
        if parent == "calibrated_band_sm3" and key in ("p90", "p10"):
            end = "low" if key == "p90" else "high"
            return Meaning(
                f"forecast oil {span}, {end} end of the range from past forecasts' errors",
                "Sm3",
            )
        if parent == "p50_by_model_sm3":
            return Meaning(f"forecast oil {span}, central estimate of the {key} model", "Sm3")
    if parent in ("fit", "best"):
        fit = top.get("fit") or top.get("best") or {}
        span = f" of the fit over {fit.get('window_start')} to {fit.get('window_end')}"
        words, unit = NAMES.get(key, (_plain(key), _unit_of(key)))
        return Meaning(words.replace(" of the fit", "") + span, unit)
    if parent in ("before", "after") and cell.rows and isinstance(cell.parent, dict):
        words, unit = NAMES.get(key, (_plain(key), _unit_of(key)))
        around, declares = _around(cell)
        days = cell.rows[-1].get("window_days")
        if not days:
            span = f" from {cell.parent.get('from')} to {cell.parent.get('to')}"
        elif parent == "before":
            span = f" in the {days} days before {around}"
        else:
            span = f" in the first {days} days back on stream after {around}"
        return Meaning(words, unit, span, declares=declares)
    limits = top.get("design_limits", {}) if isinstance(top, dict) else {}
    if tool == "field_vs_design" and len(cell.keys) == 3 and parent in limits and key == "value":
        limit = limits[parent]
        return Meaning(f"plan's {limit.get('limit', parent)}", _unit_of(parent), field=True)
    words, unit = NAMES.get(key, (_plain(key), _unit_of(key)))
    when = _when(cell)
    if record and not when and record.get("start"):
        when = f" ({record.get('kind', '').replace('_', ' ')} from {record['start']})"
    field_wide = tool in ("plan_vs_actual", "field_vs_design") or key.startswith("field")
    return Meaning(words, unit, when, field=field_wide)


# ---- printing numbers ---------------------------------------------------------------------

UNIT_TEXT = {
    "rate": " Sm3/d",
    "Sm3": " Sm3",
    "MSm3": " million Sm3",
    "days": " days",
    "months": " months",
    "hours": " hours",
    "bar": " bar",
    "m": " m",
    "per_year": " per year",
}


def _round(v: float) -> str:
    n = abs(v)
    if n >= 1_000_000:
        return f"{v / 1e6:.3g} million"
    if n >= 10_000:
        return f"{float(f'{v:.3g}'):,.0f}"
    if n >= 100:
        return f"{v:,.0f}"
    if n == 0:
        return "0"
    return f"{v:.3g}"


def _percent(p: float) -> str:
    n = abs(p)
    text = f"{p:.0f}" if n >= 10 else f"{p:.1f}".rstrip("0").rstrip(".") if n >= 1 else f"{p:.2g}"
    return text + "%"


def fmt(v: float, unit: str) -> str:
    """Three significant digits at most, thousands whole, with the unit."""
    if unit == "share":
        return _percent(v * 100)
    if unit == "pct":
        return _percent(v)
    if unit in ("change", "points"):
        return _percent(abs(v) * 100) if unit == "change" else f"{abs(v) * 100:.0f} points"
    text = _round(v)
    if unit == "Sm3" and text.endswith("million"):
        return text + " Sm3"
    return text + UNIT_TEXT.get(unit, "")


# ---- statements ---------------------------------------------------------------------------


@dataclass
class Printed:
    text: str
    numbers: list[DeclaredNumber] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    projection: bool = False  # what has not happened yet: never a fact
    wells: set[str] = field(default_factory=set)  # the wells it is about
    # The statement with fewer numbers, for a claim that would hold too many; the values
    # it leaves out go to the finding's detail.
    short: str | None = None
    short_numbers: list[DeclaredNumber] = field(default_factory=list)
    detail: str = ""


@dataclass
class Context:
    """What the sentence so far has said: the well "its" stands for, windows spelled out."""

    well: str | None = None
    said: set[str] = field(default_factory=set)

    def owns(self, well: str | None) -> bool:
        return bool(well and self.well and _well_key(self.well) == _well_key(well))


def _owner(cell: Cell, m: Meaning, ctx: Context) -> str:
    if m.field or m.words.startswith("the field") or m.words.startswith("plan"):
        return "the " if not m.words.startswith("the ") else ""
    well = cell.well
    if not well:
        return "the "
    if ctx.owns(well):
        return "its "
    ctx.well = well
    return f"{well}'s "


def _about(cell: Cell, m: Meaning) -> set[str]:
    return set() if m.field or not cell.well else {cell.well}


def _number(shown: str, value: float, cell: Cell) -> DeclaredNumber:
    return DeclaredNumber(text=_lead(shown), value=value, evidence_id=cell.evidence.id)


def _extra(m: Meaning, cell: Cell) -> list[DeclaredNumber]:
    return [DeclaredNumber(text=t, value=v, evidence_id=cell.evidence.id) for t, v in m.declares]


def _value(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    cell = resolve(part.ref or "", log)
    v = number(cell)
    m = meaning(cell)
    owner = _owner(cell, m, ctx)
    shown = fmt(v, m.unit)
    out = Printed(
        "", [_number(shown, v, cell), *_extra(m, cell)], [cell.evidence.id], wells=_about(cell, m)
    )
    if m.unit in ("change", "points"):
        verb = "down" if v < 0 else "up"
        out.text = f"{owner}{m.words} was {verb} {shown}{m.when}"
    elif part.form == "phrase":
        out.text = f"{owner}{m.words} of {shown}{m.when}"
    else:
        out.text = f"{owner}{m.words} was {shown}{m.when}"
    return out


def _windows(a: Cell, b: Cell) -> bool:
    """Two ends that are one record's before and after windows (production_log's plugs)."""
    return (
        a.key == b.key
        and a.keys[-2:-1] == ["before"]
        and b.keys[-2:-1] == ["after"]
        and bool(a.rows and b.rows)
        and a.rows[-1] == b.rows[-1]
        and bool(a.rows[-1].get("window_days"))
    )


def _change(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    a, b = resolve(part.ref or "", log), resolve(part.to or "", log)
    va, vb = number(a), number(b)
    ma, mb = meaning(a), meaning(b)
    paired = PAIRS.get(a.key) == b.key and a.record is not None and a.record is b.record
    windows = _windows(a, b)
    if not (paired or windows) and (a.key != b.key or ma.unit != mb.unit):
        raise BadRef(
            f"change: {a.ref} ({ma.words}) and {b.ref} ({mb.words}) are not the same quantity; "
            "a change runs between two values of one field (two rows, or a record's before "
            "and after)"
        )
    if a.well != b.well:
        raise BadRef(f"change: {a.ref} is {a.well}'s and {b.ref} is {b.well}'s")
    out = Printed(
        "",
        evidence=list(dict.fromkeys([a.evidence.id, b.evidence.id])),
        wells=_about(a, ma) | _about(b, mb),
    )
    owner = _owner(a, ma, ctx)
    sa, sb = fmt(va, ma.unit), fmt(vb, mb.unit)
    ends = [_number(sa, va, a), _number(sb, vb, b)]
    verb = "stayed at" if vb == va else "rose" if vb > va else "fell"
    level = ma.unit in ("share", "pct", "change", "points")
    pct = _percent(abs(vb / va - 1) * 100) if va and not level else ""
    who = a.well or "the field"
    if paired:
        q = SIGNALS.get(a.record.get("signal", ""), ("",))[0]
        q = "choke opening" if a.key.startswith("choke") else q or "oil rate"
        d = a.record.get("start", "")
        if a.key == "before_mean" and pct:
            less = "less" if vb < va else "more"
            out.short = f"{owner}{q} averaged {pct} {less} after {d} than before"
            out.text = f"{out.short} ({sa} before, {sb} after)"
            out.detail = f"{who} {q}: {sa} before and {sb} after {d}, the periods' averages."
        else:
            span = (
                "over the 4 weeks either side of"
                if a.key != "before_mean"
                else "between the periods before and after"
            )
            out.text = f"{owner}{q} {verb} from {sa} to {sb} {span} {d}"
        out.numbers = ends
    elif windows:
        around, declares = _around(a)
        days = a.rows[-1]["window_days"]
        seen = f"{a.evidence.id}:{around}"
        if seen in ctx.said:
            span, extra = "over the same days", []
        else:
            ctx.said.add(seen)
            span = (
                f"between the {days} days before {around} and the first {days} days back on stream"
            )
            extra = [_number(t, v, a) for t, v in declares]
        if pct:
            out.short = f"{owner}{ma.words} {verb} {pct} {span}"
            out.text = f"{owner}{ma.words} {verb} {pct} from {sa} to {sb} {span}"
            out.detail = f"{who} {ma.words}: {sa} before, {sb} after."
        else:
            out.text = f"{owner}{ma.words} {verb} from {sa} to {sb} {span}"
        out.numbers, out.short_numbers = ends + extra, extra
    else:
        by = f" {pct}" if pct else ""
        out.text = f"{owner}{ma.words} {verb}{by} from {sa}{ma.when} to {sb}{mb.when}"
        out.numbers = ends + _extra(ma, a) + _extra(mb, b)
    return out


def _lead(text: str) -> str:
    """The number as printed, without its unit word ('1,550 Sm3/d' -> '1,550')."""
    m = re.match(r"[\d,.]+(?:%| million)?", text)
    return m[0] if m else text


def _compare(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    a, b = resolve(part.ref or "", log), resolve(part.against or "", log)
    va, vb = number(a), number(b)
    ma, mb = meaning(a), meaning(b)
    if ma.unit != mb.unit:
        raise BadRef(f"compare: {a.ref} is in {ma.unit} and {b.ref} in {mb.unit}")
    out = Printed(
        "",
        evidence=list(dict.fromkeys([a.evidence.id, b.evidence.id])),
        wells=_about(a, ma) | _about(b, mb),
    )
    left = f"{_owner(a, ma, ctx)}{ma.words} of {fmt(va, ma.unit)}{ma.when}"
    right = f"{_owner(b, mb, ctx)}{mb.words} of {fmt(vb, mb.unit)}{mb.when}"
    if (part.how or "ratio") == "ratio":
        if not vb:
            raise BadRef(f"compare: {b.ref} is zero, so there is no ratio")
        r = va / vb
        text = (
            f"{left} is {r:.2g} times {right}"
            if r >= 1.5
            else f"{left} is {_percent(r * 100)} of {right}"
        )
    else:
        diff = va - vb
        side = "above" if diff > 0 else "below" if diff < 0 else "equal to"
        text = (
            f"{left} is {fmt(abs(diff), ma.unit)} {side} {right}"
            if diff
            else f"{left} equals {right}"
        )
    out.text = text
    out.numbers += [
        _number(fmt(va, ma.unit), va, a),
        _number(fmt(vb, mb.unit), vb, b),
        *_extra(ma, a),
        *_extra(mb, b),
    ]
    return out


EVENT_WORDS = {
    "gauge_failure": "{w} downhole gauge read zero from {start} to {end}",
    "dead_gauge": "{w} downhole gauge has read zero since {since}",
    "shut_in": "{w} was shut in from {start} to {end}",
    "regime_change": "{w} {signal} shifted around {start}",
    "threshold_crossing": (
        "{w} 30-day median {signal} first went {direction} {threshold} on {start}"
    ),
    "shift_down": "{w} oil rate shifted down from {since}",
    "step_down": "{w} oil rate stepped down at {since}",
}


def _event(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    cell = resolve(part.ref or "", log)
    record = cell.value
    if not isinstance(record, dict) or not ({"kind", "start", "since"} & record.keys()):
        raise BadRef(f"{cell.ref} is not a record of an event or job (it has no kind or dates)")
    well = cell.well
    owner = "the field's" if not well else ("its" if ctx.owns(well) else f"{well}'s")
    ctx.well = well or ctx.well
    kind = record.get("kind", "")
    details = record.get("details") or {}
    signal = SIGNALS.get(
        record.get("signal", ""), (str(record.get("signal", "")).replace("_", " "),)
    )[0]
    values = {
        "w": owner,
        "signal": signal,
        "start": record.get("start"),
        "end": record.get("end"),
        "since": record.get("since"),
        "direction": details.get("direction", "past"),
        "threshold": _percent(details["threshold"] * 100)
        if signal == "water cut" and "threshold" in details
        else details.get("threshold"),
    }
    if kind in EVENT_WORDS:
        text = EVENT_WORDS[kind].format(**values)
        if kind == "shut_in" and details.get("field_wide"):
            text += ", with other wells (field-wide)"
        if kind in ("gauge_failure", "shut_in") and not record.get("end"):
            text = text.replace(" to None", "")
    elif record.get("text") and record.get("since"):  # a screen issue the tool words itself
        who = f"{well}'s" if well else "the field's"
        text = f"{who} {kind.replace('_', ' ')} since {record['since']}: {record['text']}"
    elif "report_days" in record or record.get("perforations") is not None:  # a well-work job
        text = f"{owner} {kind} job ran from {record.get('start')} to {record.get('end')}"
        perfs = [p for p in record.get("perforations") or [] if isinstance(p, dict)]
        if perfs:
            spans = ", ".join(
                f"{p.get('top_m', p.get('from_m'))}-{p.get('bottom_m', p.get('to_m'))} m"
                for p in perfs
            )
            text += f" and perforated {spans}"
    else:
        start = record.get("start") or record.get("since")
        text = f"{owner} {kind.replace('_', ' ')} from {start}"
    return Printed(
        text.replace("  ", " "), evidence=[cell.evidence.id], wells={well} if well else set()
    )


def _forecast(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    cell = resolve(part.ref or "", log)
    item = cell.evidence
    top = item.result
    if item.tool != "forecast_production" or cell.keys or not isinstance(top, dict):
        raise BadRef(
            f"forecast: {cell.ref} must be a forecast_production result itself, e.g. '{item.id}'"
        )
    well = item.arguments.get("well") or top.get("well")
    owner = "it" if ctx.owns(well) else str(well)
    ctx.well = well
    p50 = top["cumulative_oil_sm3"]["p50"]
    out = Printed("", evidence=[item.id], projection=True, wells={well} if well else set())
    out.numbers.append(DeclaredNumber(text=_lead(fmt(p50, "Sm3")), value=p50, evidence_id=item.id))
    text = (
        f"{owner} is forecast to make about {fmt(p50, 'Sm3')} of oil from "
        f"{top['forecast_from']} to {top['forecast_to']}"
    )
    band = top.get("calibrated_band_sm3")
    if band:
        lo, hi = band["p90"], band["p10"]
        out.numbers += [
            DeclaredNumber(text=_lead(fmt(lo, "Sm3")), value=lo, evidence_id=item.id),
            DeclaredNumber(text=_lead(fmt(hi, "Sm3")), value=hi, evidence_id=item.id),
        ]
        text += (
            f", range {_lead(fmt(lo, 'Sm3'))} to {fmt(hi, 'Sm3')} from the errors of "
            f"{band.get('past_forecasts')} earlier forecasts made the same way"
        )
    else:
        text += ", with no range from past forecasts (too few to calibrate one)"
    out.text = text
    return out


DOC_NAMES = {"volve_pud": "the development plan", "f11b_completion": "F-11 B's completion report"}


def _quote(part: Part, log: EvidenceLog, ctx: Context) -> Printed:
    cell = resolve(part.ref or "", log)
    fact = cell.value
    if not isinstance(fact, dict) or not fact.get("quote"):
        raise BadRef(f"quote: {cell.ref} is not a report fact with a quote")
    if fact.get("quote_check", "exact") != "exact":
        raise BadRef(f"quote: {cell.ref}'s quote is not found word for word on its page")
    doc = DOC_NAMES.get(fact.get("doc_id"), fact.get("doc_id") or "the report")
    page = fact.get("page")
    english = fact.get("statement_en")
    text = f"{doc} says “{fact['quote'].strip()}” (page {page}"
    text += f": {english.rstrip('.')})" if english else ")"
    return Printed(text, evidence=[cell.evidence.id])


RENDER = {
    "value": _value,
    "change": _change,
    "compare": _compare,
    "event": _event,
    "forecast": _forecast,
    "quote": _quote,
}

# Free words may hold none of what a reference prints.
MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
FORBIDDEN = [
    (re.compile(r"\d"), "a digit"),
    (re.compile(rf"\b(?:{MONTHS})\b(?!\s+(?:be|have|not|also|well)\b)", re.I), "a month name"),
    (re.compile(r"\b(?:sm3|m3|bbl|barrels?|bar|psi|percent|per ?cent|stb)\b|%", re.I), "a unit"),
    (re.compile(r"[\"“”«»„]"), "a quote mark"),
    (
        re.compile(
            r"\b(?:half|third|quarter|twice|double|triple|one|two|three|four|five|six|seven|"
            r"eight|nine|ten|eleven|twelve|twenty|thirty|hundred|thousand|million|dozen)\b",
            re.I,
        ),
        "an amount in words",
    ),
]


def _lint(text: str) -> str | None:
    for pattern, what in FORBIDDEN:
        if m := pattern.search(text):
            return f"'{m[0]}' is {what}"
    return None


@dataclass
class Rendered:
    submission: Submission | None
    problems: list[str]
    texts: list[str]  # the summary and each finding, as printed (for the model and the log)


# Words after a statement that continue it with no comma ("and F-14's ...", "than ..."),
# and words a statement can follow with no colon ("while", "with", "than"); not "before"
# or "after", which end a phrase as often as they lead into one ("than before").
CONTINUES = re.compile(r"(?:and|or|nor|than|to|of|in|on|at|for|from|by|as)\b", re.I)
LEADS_IN = re.compile(
    r"\b(?:and|or|nor|but|yet|so|while|whereas|though|although|because|since|as|with|"
    r"without|of|at|to|than|when|which|that|by|from|in|on|for|where|whose|"
    r"plus|including|like|unlike|versus|against|is|was|were|are|also|then|both|even|only)$",
    re.I,
)


def _join(chunks: list[tuple[bool, str]]) -> str:
    """Free words (True) and printed statements (False) as sentences. Free words keep
    their own punctuation; without it, a statement after words that do not lead into it
    ("F-12 needs attention most") takes a colon, or a semicolon once its sentence has
    one, and words after a statement take a comma unless they continue it ("and")."""
    text, last_said = "", False
    for said, chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not text or re.match(r"[,.;:)]", chunk):
            sep = ""
        elif re.search(r"[.!?,;:(]$", text):
            sep = " "
        elif said:
            sep = " " if last_said or CONTINUES.match(chunk) else ", "
        elif not last_said:
            sep = "; "
        elif LEADS_IN.search(text):
            sep = " "
        else:
            sep = "; " if ":" in re.split(r"[.!?]\s+", text)[-1] else ": "
        text += sep + chunk
        last_said = said
    text = re.sub(r"\s+", " ", text).strip().rstrip(",;:").rstrip()
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)
    return text if re.search(r"[.!?]$", text) else text + "."


def _print(where: str, parts: list[Part], log: EvidenceLog, problems: list[str]) -> Printed:
    pieces: list[tuple[bool, Printed]] = []
    ctx = Context()
    named: dict[str, str] = {}  # wells the free words name, by key
    for n, part in enumerate(parts, start=1):
        if part.kind is None:
            words = part.say or ""
            if not words.strip():
                problems.append(f"{where}, part {n}: give either 'say' (free words) or a 'kind'")
                continue
            if bad := _lint(WELL.sub(" ", words)):
                problems.append(
                    f"{where}, part {n}: free words may not hold numbers, dates, units, "
                    f"amounts in words or quotes: {bad}. Take it from a tool result with a "
                    "statement instead."
                )
            wells = [w.strip() for w in WELL.findall(words)]
            for w in wells:
                named.setdefault(_well_key(w), w)
            # Words that start with their one well make it the subject: the next statement
            # about it says "its". Words naming another well, or two, leave the next
            # statement to name its own.
            if wells:
                one = len({_well_key(w) for w in wells}) == 1
                ctx.well = wells[0] if one and words.lstrip().startswith(wells[0]) else None
            pieces.append((True, Printed(words)))
            continue
        try:
            printed = RENDER[part.kind](part, log, ctx)
        except BadRef as e:
            problems.append(f"{where}, part {n} ({part.kind}): {e}")
            continue
        except (KeyError, TypeError) as e:
            problems.append(f"{where}, part {n} ({part.kind}): {part.ref} lacks {e}")
            continue
        if part.say:
            problems.append(
                f"{where}, part {n}: a part is either free words or a statement, not both"
            )
        pieces.append((False, printed))
    statements = [p for said, p in pieces if not said]
    about = {_well_key(w) for p in statements for w in p.wells}
    for key, name in named.items():
        if key not in about:
            problems.append(
                f"{where}: the free words name {name}, but none of its statements is about "
                f"{name}: add one, or leave the name out."
            )
    # A claim a reader takes in at once: past MAX_CLAIM_NUMBERS numbers, the statements
    # that have a shorter form print it, from the last one back, and their values go to
    # the detail.
    short: set[int] = set()

    def joined() -> str:
        chunks = [(said, p.short if id(p) in short else p.text) for said, p in pieces]
        return _same_window(_join(chunks))

    text = joined()
    for p in reversed(statements):
        if len(claim_numbers(text)) <= MAX_CLAIM_NUMBERS:
            break
        if p.short is not None:
            short.add(id(p))
            text = joined()
    out = Printed(text)
    details = []
    for p in statements:
        cut = id(p) in short
        out.numbers += p.short_numbers if cut else p.numbers
        if cut:
            details.append(p.detail)
        out.evidence += p.evidence
        out.projection |= p.projection
        out.wells |= p.wells
    out.detail = " ".join(details)
    out.evidence = list(dict.fromkeys(out.evidence))
    return out


WINDOW = re.compile(r"over the (\d+) days to (\d{4}-\d{2}-\d{2})")


def _same_window(text: str) -> str:
    """'over the 90 days to 2010-12-31' once per sentence; after that 'over the same 90
    days', which still says which values they are."""
    seen: set[str] = set()

    def once(m: re.Match) -> str:
        if m[0] in seen:
            return f"over the same {m[1]} days"
        seen.add(m[0])
        return m[0]

    return ". ".join(WINDOW.sub(once, sentence) for sentence in text.split(". "))


def render(submission: StatementSubmission, log: EvidenceLog) -> Rendered:
    """Print the answer from its references and check what code can: every reference
    exists and is the kind of value its statement needs, free words hold nothing a
    reference should, a fact rests on no warned result or projection and states no cause,
    and a cause in the summary is hedged."""
    problems: list[str] = []
    findings: list[Finding] = []
    texts: list[str] = []
    summary = _print("Summary", submission.summary, log, problems)
    words = " ".join(p.say or "" for p in submission.summary)
    for sentence in re.split(r"(?<=[.;!?])\s+", words):
        if CAUSAL.search(sentence) and not HEDGE.search(sentence):
            problems.append(
                f"Summary: '{CAUSAL.search(sentence)[0]}' states a cause without a hedge; a "
                "cause is an inference ('likely because', 'consistent with')."
            )
    texts.append(summary.text)
    for n, item in enumerate(submission.findings, start=1):
        where = f"Finding {n}"
        printed = _print(where, item.parts, log, problems)
        texts.append(printed.text)
        if item.confidence == "fact":
            warned = [e for e in printed.evidence if (ev := log.get(e)) and ev.warnings]
            words = " ".join(p.say or "" for p in item.parts)
            if warned:
                problems.append(
                    f"{where} is marked fact but rests on {', '.join(warned)}, which carries "
                    "warnings: mark it inference."
                )
            elif printed.projection:
                problems.append(f"{where} is marked fact but holds a forecast: mark it inference.")
            elif m := CAUSAL.search(words):
                problems.append(
                    f"{where} is marked fact but states a cause ('{m[0]}'): mark it inference."
                )
        if not printed.evidence:
            problems.append(
                f"{where} has no statement: give at least one value, change, event, forecast "
                "or quote."
            )
        findings.append(
            Finding(
                claim=printed.text,
                detail=printed.detail or None,
                evidence_ids=printed.evidence or ["E1"],
                confidence=item.confidence,
                numbers=_unique(printed.numbers),
            )
        )
    result = Submission(
        summary=summary.text,
        findings=findings,
        summary_numbers=_unique(summary.numbers),
    )
    return Rendered(result, problems, texts)


def _unique(numbers: list[DeclaredNumber]) -> list[DeclaredNumber]:
    seen, out = set(), []
    for n in numbers:
        if (n.text, n.evidence_id) not in seen:
            seen.add((n.text, n.evidence_id))
            out.append(n)
    return out
