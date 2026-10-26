"""Deterministic claim checker for the agent's final findings.

Before an answer is accepted, every number and ISO date in each finding must be found in
the evidence that finding cites, and a finding marked "fact" may not rest on evidence
that carries warnings (e.g. a poor decline fit). Problems go back to the model as plain
sentences so it can fix them and resubmit.
"""

import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from wellpilot.agent.evidence import Evidence, EvidenceLog

WELL = re.compile(r"\bF-\d+(?:\s?[A-Z]\b)?")
DATE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?)\b")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*(%|x\b|×|million\b|k\b))?(?!\w)"
)
SCALES = {"%": (0.01, 1.0), "million": (1e6,), "k": (1e3,)}
SMALL_INTEGER = 10  # "3 wells", "2 years": too generic to demand evidence for
# Causes are interpretations: the data shows timing and size, not why.
CAUSAL = re.compile(
    r"\b(because|caus(?:e|ed|es|ing)|dr(?:ove|ives?|iven)|due to|led to|leads? to|"
    r"result(?:s|ed)? (?:from|in)|responsible for)\b",
    re.IGNORECASE,
)


class Finding(BaseModel):
    claim: str = Field(description="One finding in one or two sentences, with its numbers.")
    evidence_ids: list[str] = Field(
        description="Ids (E1, E2, ...) of the evidence every number and date in the claim "
        "comes from."
    )
    confidence: Literal["fact", "inference"] = Field(
        description="'fact' if the cited evidence shows it directly; 'inference' for "
        "interpretations such as likely causes."
    )


class Submission(BaseModel):
    summary: str = Field(description="Two or three sentences answering the question.")
    findings: list[Finding] = Field(min_length=1)


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _flatten(v)
    else:
        yield value


def _numbers_in_text(text: str) -> list[tuple[float, int, str | None]]:
    """(value, decimals, suffix) for each number, ignoring well names and ISO dates."""
    text = THOUSANDS.sub("", DATE.sub(" ", WELL.sub(" ", text)))
    found = []
    for match in NUMBER.finditer(text):
        digits, suffix = match.group(1), match.group(2)
        mantissa = re.split(r"[eE]", digits)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        found.append((float(digits), decimals, suffix))
    return found


class _Pool:
    """Every number and date that appears in a set of evidence results."""

    def __init__(self, evidence: Iterable[Evidence]) -> None:
        self.numbers: list[float] = []
        self.dates: set[str] = set()
        for item in evidence:
            for leaf in _flatten([item.result, item.arguments]):
                if isinstance(leaf, bool) or leaf is None:
                    continue
                if isinstance(leaf, int | float):
                    self.numbers.append(abs(float(leaf)))
                elif isinstance(leaf, str):
                    self.dates.update(DATE.findall(leaf))
                    self.numbers.extend(v for v, _, _ in _numbers_in_text(leaf))
        self.years = {d[:4] for d in self.dates}

    def has_number(self, value: float, decimals: int, suffix: str | None) -> bool:
        for scale in SCALES.get(suffix or "", (1.0,)):
            target = value * scale
            tolerance = max(0.5 * 10**-decimals * scale, 0.01 * target, 1e-9)
            if any(abs(n - target) <= tolerance for n in self.numbers):
                return True
        return False

    def has_date(self, day: str) -> bool:
        return any(d.startswith(day) or day.startswith(d) for d in self.dates)


def _unsupported(text: str, pool: _Pool) -> tuple[list[str], list[str]]:
    numbers = []
    for value, decimals, suffix in _numbers_in_text(text):
        bare_integer = decimals == 0 and not suffix and value.is_integer()
        if bare_integer and 1900 <= value <= 2100:
            # A year, or a plain number such as 2095 days or 2000 Sm3/d.
            if str(int(value)) not in pool.years and not pool.has_number(value, 0, None):
                numbers.append(str(int(value)))
        elif decimals == 0 and not suffix and value <= SMALL_INTEGER:
            continue
        elif not pool.has_number(value, decimals, suffix):
            numbers.append(f"{value:g}{suffix or ''}")
    dates = [d for d in DATE.findall(text) if not pool.has_date(d)]
    return numbers, dates


def check_submission(submission: Submission, log: EvidenceLog) -> list[str]:
    """Return problems as sentences for the model; an empty list means accepted."""
    problems = []
    for n, finding in enumerate(submission.findings, start=1):
        if not finding.evidence_ids:
            problems.append(f"Finding {n} cites no evidence; cite the evidence ids it uses.")
            continue
        unknown = [e for e in finding.evidence_ids if log.get(e) is None]
        if unknown:
            problems.append(f"Finding {n} cites unknown evidence {', '.join(unknown)}.")
            continue
        cited = [log.get(e) for e in finding.evidence_ids]
        numbers, dates = _unsupported(finding.claim, _Pool(cited))
        ids = ", ".join(finding.evidence_ids)
        if numbers:
            problems.append(
                f"Finding {n}: {', '.join(numbers)} not found in {ids}. Cite the evidence it "
                "comes from; compute derived numbers with calculate; or remove them."
            )
        if dates:
            problems.append(f"Finding {n}: date(s) {', '.join(dates)} not found in {ids}.")
        cause = CAUSAL.search(finding.claim)
        if finding.confidence == "fact" and cause:
            problems.append(
                f"Finding {n} states a cause ('{cause.group(0)}') but is marked fact. The data "
                "shows timing and size, not causes: mark it inference."
            )
        flagged = [e for e in cited if e.warnings]
        if finding.confidence == "fact" and flagged:
            reasons = "; ".join(f"{e.id}: {w}" for e in flagged for w in e.warnings)
            problems.append(
                f"Finding {n} is marked fact but rests on evidence with warnings ({reasons}). "
                "Mark it inference, or refit and cite the new evidence."
            )

    numbers, dates = _unsupported(submission.summary, _Pool(log))
    if numbers or dates:
        missing = ", ".join(numbers + dates)
        problems.append(f"Summary: {missing} not found in any evidence.")
    return problems
