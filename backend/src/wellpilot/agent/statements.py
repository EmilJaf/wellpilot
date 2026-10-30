"""Answers written as statements, not prose (design D in docs/checker-redesign.md).

The model never types a number, a date, a well name or a unit. It picks values out of
the tool results by reference ("E1/wells[well=F-12]/water_cut") and a statement kind;
code prints the sentence from the values and their own labels. A wrong pick is then a
true sentence about another value, never a false one. Free words join the statements and
may hold no digits, dates, units, well names or quote marks.

Behind a flag (run_agent(statements=True), eval --statements) for a feasibility run: the
rendered answer becomes an ordinary Submission, so the answer panel, the second opinion
and the reveal work as they are.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from wellpilot.agent.checker import CAUSAL, HEDGE, DeclaredNumber, Finding, Submission
from wellpilot.agent.evidence import Evidence, EvidenceLog

Kind = Literal["value", "change", "compare", "event", "forecast", "quote"]


class Part(BaseModel):
    say: str | None = Field(
        None,
        description="Free words joining statements ('needs attention most:', 'while'). No "
        "digits, dates, month names, units, well names or quote marks: those come from refs.",
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
            if name not in node:
                fields = ", ".join(k for k in node if not k.endswith("note"))
                raise BadRef(f"{where} has no field '{name}'; its fields: {fields}")
            cell.value = node[name]
            cell.keys.append(name)
            where = f"{where}/{name}"
        for selector in selectors:
            row = _select(cell.value, selector, where)
            cell.rows.append(row)
            cell.value = row
            where = f"{where}[{selector}]"
    return cell


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
    limits = top.get("design_limits", {}) if isinstance(top, dict) else {}
    if tool == "field_vs_design" and len(cell.keys) == 3 and parent in limits and key == "value":
        limit = limits[parent]
        return Meaning(f"plan's {limit.get('limit', parent)}", _unit_of(parent), field=True)
    words, unit = NAMES.get(key, (_plain(key), _unit_of(key)))
    when = _when(cell)
    if record and not when and record.get("start"):
        when = f" ({record.get('kind', '').replace('_', ' ')} from {record['start']})"
    field_wide = tool in ("plan_vs_actual", "field_vs_design") or key.startswith("field")
    return Meaning(words, unit, when, field_wide)


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


def _owner(cell: Cell, m: Meaning, last_well: list[str | None]) -> str:
    well = None if m.field else cell.well
    if m.field or m.words.startswith("the field") or m.words.startswith("plan"):
        return "the " if not m.words.startswith("the ") else ""
    if not well:
        return "the "
    if last_well[0] == well:
        return "its "
    last_well[0] = well
    return f"{well}'s "


def _declare(out: Printed, shown: str, cell: Cell) -> None:
    out.numbers.append(
        DeclaredNumber(text=_lead(shown), value=number(cell), evidence_id=cell.evidence.id)
    )


def _value(part: Part, log: EvidenceLog, last: list) -> Printed:
    cell = resolve(part.ref or "", log)
    v = number(cell)
    m = meaning(cell)
    out = Printed("", evidence=[cell.evidence.id])
    owner = _owner(cell, m, last)
    shown = fmt(v, m.unit)
    _declare(out, shown, cell)
    if m.unit in ("change", "points"):
        verb = "down" if v < 0 else "up"
        out.text = f"{owner}{m.words} was {verb} {shown}{m.when}"
    elif part.form == "phrase":
        out.text = f"{owner}{m.words} of {shown}{m.when}"
    else:
        out.text = f"{owner}{m.words} was {shown}{m.when}"
    return out


def _change(part: Part, log: EvidenceLog, last: list) -> Printed:
    a, b = resolve(part.ref or "", log), resolve(part.to or "", log)
    va, vb = number(a), number(b)
    ma, mb = meaning(a), meaning(b)
    paired = PAIRS.get(a.key) == b.key and a.record is not None and a.record is b.record
    if not paired and (a.key != b.key or ma.unit != mb.unit):
        raise BadRef(
            f"change: {a.ref} ({ma.words}) and {b.ref} ({mb.words}) are not the same quantity; "
            "a change runs between two values of one field (two rows, or a record's before "
            "and after)"
        )
    if a.well != b.well:
        raise BadRef(f"change: {a.ref} is {a.well}'s and {b.ref} is {b.well}'s")
    out = Printed("", evidence=list(dict.fromkeys([a.evidence.id, b.evidence.id])))
    owner = _owner(a, ma, last)
    sa, sb = fmt(va, ma.unit), fmt(vb, mb.unit)
    verb = "stayed at" if vb == va else "rose" if vb > va else "fell"
    if paired:
        q, unit = SIGNALS.get(a.record.get("signal", ""), ("", ma.unit))
        q = "choke opening" if a.key.startswith("choke") else q or "oil rate"
        d = a.record.get("start", "")
        if a.key == "before_mean" and ma.unit not in ("share", "pct") and va:
            pct = _percent(abs(vb / va - 1) * 100)
            less = "less" if vb < va else "more"
            text = (
                f"{owner}{q} averaged {pct} {less} after {d} than before ({sa} before, {sb} after)"
            )
        else:
            span = (
                "over the 4 weeks either side of"
                if a.key != "before_mean"
                else "between the periods before and after"
            )
            text = f"{owner}{q} {verb} from {sa} to {sb} {span} {d}"
    else:
        pct = ""
        if ma.unit not in ("share", "pct", "change", "points") and va:
            pct = f" {_percent(abs(vb / va - 1) * 100)}"
        text = f"{owner}{ma.words} {verb}{pct} from {sa}{ma.when} to {sb}{mb.when}"
    out.text = text
    out.numbers += [
        DeclaredNumber(text=_lead(sa), value=va, evidence_id=a.evidence.id),
        DeclaredNumber(text=_lead(sb), value=vb, evidence_id=b.evidence.id),
    ]
    return out


def _lead(text: str) -> str:
    """The number as printed, without its unit word ('1,550 Sm3/d' -> '1,550')."""
    m = re.match(r"[\d,.]+(?:%| million)?", text)
    return m[0] if m else text


def _compare(part: Part, log: EvidenceLog, last: list) -> Printed:
    a, b = resolve(part.ref or "", log), resolve(part.against or "", log)
    va, vb = number(a), number(b)
    ma, mb = meaning(a), meaning(b)
    if ma.unit != mb.unit:
        raise BadRef(f"compare: {a.ref} is in {ma.unit} and {b.ref} in {mb.unit}")
    out = Printed("", evidence=list(dict.fromkeys([a.evidence.id, b.evidence.id])))
    left = f"{_owner(a, ma, last)}{ma.words} of {fmt(va, ma.unit)}{ma.when}"
    right = f"{_owner(b, mb, last)}{mb.words} of {fmt(vb, mb.unit)}{mb.when}"
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
        DeclaredNumber(text=_lead(fmt(va, ma.unit)), value=va, evidence_id=a.evidence.id),
        DeclaredNumber(text=_lead(fmt(vb, mb.unit)), value=vb, evidence_id=b.evidence.id),
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


def _event(part: Part, log: EvidenceLog, last: list) -> Printed:
    cell = resolve(part.ref or "", log)
    record = cell.value
    if not isinstance(record, dict) or not ({"kind", "start", "since"} & record.keys()):
        raise BadRef(f"{cell.ref} is not a record of an event or job (it has no kind or dates)")
    well = cell.well
    owner = "the field's" if not well else ("its" if last[0] == well else f"{well}'s")
    last[0] = well or last[0]
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
    return Printed(text.replace("  ", " "), evidence=[cell.evidence.id])


def _forecast(part: Part, log: EvidenceLog, last: list) -> Printed:
    cell = resolve(part.ref or "", log)
    item = cell.evidence
    top = item.result
    if item.tool != "forecast_production" or cell.keys or not isinstance(top, dict):
        raise BadRef(
            f"forecast: {cell.ref} must be a forecast_production result itself, e.g. '{item.id}'"
        )
    well = item.arguments.get("well") or top.get("well")
    owner = "it" if last[0] == well else str(well)
    last[0] = well
    p50 = top["cumulative_oil_sm3"]["p50"]
    out = Printed("", evidence=[item.id], projection=True)
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


def _quote(part: Part, log: EvidenceLog, last: list) -> Printed:
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


def _join(chunks: list[str]) -> str:
    text = ""
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if text and not re.match(r"[,.;:)]", chunk):
            text += " "
        text += chunk
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:1].upper() + text[1:]
    return text if re.search(r"[.!?)]$", text) else text + "."


def _print(where: str, parts: list[Part], log: EvidenceLog, problems: list[str]) -> Printed:
    out = Printed("")
    chunks: list[str] = []
    last: list[str | None] = [None]
    for n, part in enumerate(parts, start=1):
        if part.kind is None:
            if not (part.say or "").strip():
                problems.append(f"{where}, part {n}: give either 'say' (free words) or a 'kind'")
                continue
            if bad := _lint(part.say or ""):
                problems.append(
                    f"{where}, part {n}: free words may not hold numbers, dates, units, well "
                    f"names, amounts in words or quotes: {bad}. Take it from a tool result "
                    "with a statement instead."
                )
            chunks.append(part.say or "")
            continue
        try:
            printed = RENDER[part.kind](part, log, last)
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
        chunks.append(printed.text)
        out.numbers += printed.numbers
        out.evidence += printed.evidence
        out.projection |= printed.projection
    out.text = _same_window(_join(chunks))
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
