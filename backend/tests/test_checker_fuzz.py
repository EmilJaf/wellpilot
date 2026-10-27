"""Property-based tests: the checker must accept any number written in a supported format
when the evidence has it, and reject numbers that are not there, whatever the format.

Each format below is one way a model might write a value; when a model invents a new
one, add it here first and let the tests say whether the checker copes.
"""

import math
from collections.abc import Callable
from datetime import date

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog


def _tool() -> None:
    """Stands in for a tool function."""


def log_of(*values: float, day: str = "2012-06-01") -> EvidenceLog:
    log = EvidenceLog()
    log.add("tool", {}, {"date": day, "values": list(values)}, _tool)
    return log


# A format writes a value as text and says what that text is worth at its precision.
Format = Callable[[float], tuple[str, float]]

LARGE: dict[str, Format] = {
    "plain": lambda v: (f"{round(v)}", round(v)),
    "comma": lambda v: (f"{round(v):,}", round(v)),
    "space": lambda v: (f"{round(v):,}".replace(",", " "), round(v)),
    "no-break space": lambda v: (f"{round(v):,}".replace(",", " "), round(v)),
    "narrow no-break space": lambda v: (f"{round(v):,}".replace(",", " "), round(v)),
    "two decimals": lambda v: (f"{v:.2f}", round(v, 2)),
    "thousands k": lambda v: (f"{v / 1000:.2f}k", round(v / 1000, 2) * 1000),
    "million": lambda v: (f"{v / 1e6:.2f} million", round(v / 1e6, 2) * 1e6),
    "scientific": lambda v: (f"{v:.3e}", float(f"{v:.3e}")),
    "glued unit": lambda v: (f"{round(v)}Sm3", round(v)),
    "norwegian": lambda v: (f"{round(v):,}".replace(",", "."), round(v)),
    "MSm3": lambda v: (f"{v / 1e6:.2f} MSm3", round(v / 1e6, 2) * 1e6),
    "kSm3": lambda v: (f"{v / 1000:.1f} kSm3", round(v / 1000, 1) * 1000),
}
# Answers from before declared numbers were read in English only.
LEGACY = {k: v for k, v in LARGE.items() if k != "norwegian"}
FRACTION: dict[str, Format] = {
    "percent": lambda f: (f"{f * 100:.1f}%", round(f * 100, 1)),
    "fraction": lambda f: (f"{f:.3f}", round(f, 3)),
}

large = st.floats(min_value=12_000, max_value=9_000_000, allow_nan=False)
fraction = st.floats(min_value=0.02, max_value=0.98, allow_nan=False)
off = st.floats(min_value=0.05, max_value=2)


def claim_with(text: str) -> str:
    return f"In 2012 the well made {text} of oil."


def realistic(style: str, value: float) -> bool:
    """Nobody writes 12,000 Sm3 as "0.01 MSm3": million formats are for millions."""
    return style not in ("million", "MSm3") or value >= 1e6


def coarsest(written: float) -> float:
    """Half a unit in the second significant digit: the loosest the checker lets a written
    number round ("4000" is good to 50)."""
    return 0.5 * 10 ** (math.floor(math.log10(written)) - 1)


def check(text: str, value: float, evidence: float, numbers: list | None = ...) -> list[str]:
    declared = [DeclaredNumber(text=text, value=value, evidence_id="E1")]
    finding = Finding(
        claim=claim_with(text),
        evidence_ids=["E1"],
        confidence="fact",
        numbers=declared if numbers is ... else numbers,
    )
    return check_submission(Submission(summary="Done.", findings=[finding]), log_of(evidence))


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)))
def test_declared_large_numbers_pass_in_every_format(value, style) -> None:
    text, written = LARGE[style](value)

    assert check(text, written, value) == []


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)))
def test_declaring_the_full_evidence_value_passes_too(value, style) -> None:
    text, _ = LARGE[style](value)

    assert check(text, value, value) == []


@settings(max_examples=100, deadline=None)
@given(value=fraction, style=st.sampled_from(sorted(FRACTION)))
def test_declared_fractions_pass_as_percent_or_fraction(value, style) -> None:
    text, written = FRACTION[style](value)

    assert check(text, written, value) == []


@settings(max_examples=200, deadline=None)
@given(
    whole=st.integers(min_value=1, max_value=99_999),
    decimals=st.integers(min_value=1, max_value=4),
    up=st.booleans(),
)
def test_a_tie_may_round_either_way(whole, decimals, up) -> None:
    """253.85 written as 253.9 (or 253.8): exactly half a unit off, plus float error."""
    value = (whole + 0.5) / 10**decimals
    text = f"{(whole + up) / 10**decimals:.{decimals}f}"

    assert check(text, value, value) == []


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)), off=off)
def test_declared_numbers_that_are_not_in_the_evidence_fail(value, style, off) -> None:
    text, written = LARGE[style](value * (1 + off))
    assume(realistic(style, value * (1 + off)))
    assume(abs(written - value) > coarsest(written))  # no rounding of the text covers value

    problems = check(text, written, value)

    assert problems and "is not in E1" in problems[0]


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)), off=off)
def test_text_that_does_not_show_its_declared_value_fails(value, style, off) -> None:
    """The evidence value declared honestly, the text written wrong: '2.00 million' for 1.6M."""
    text, written = LARGE[style](value * (1 + off))
    assume(realistic(style, value * (1 + off)))
    assume(abs(written - value) > coarsest(written))

    problems = check(text, value, value)

    assert problems and "does not show its declared value" in problems[0]


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)))
def test_an_undeclared_number_is_caught_in_every_format(value, style) -> None:
    text, written = LARGE[style](value)

    problems = check(text, written, value, numbers=[])

    assert problems and "undeclared number" in problems[0]


@settings(max_examples=150, deadline=None)
@given(
    value=large,
    style=st.sampled_from(sorted(LARGE)),
    day=st.dates(date(2008, 1, 1), date(2016, 12, 31)),
    small=st.integers(min_value=1, max_value=99),
)
def test_short_declared_numbers_leave_dates_and_wells_alone(value, style, day, small) -> None:
    """A declared "2" or "18" must not match inside 2010-07-18, F-12 or 2 100."""
    text, written = LARGE[style](value)
    claim = f"F-{small} made {text} by {day.isoformat()}, {small} months after first oil."
    finding = Finding(
        claim=claim,
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[
            DeclaredNumber(text=text, value=written, evidence_id="E1"),
            DeclaredNumber(text=str(small), value=small, evidence_id="E1"),
        ],
    )
    log = log_of(value, small, day=day.isoformat())

    assert check_submission(Submission(summary="Done.", findings=[finding]), log) == []


@settings(max_examples=150, deadline=None)
@given(
    value=large,
    style=st.sampled_from(sorted(LARGE)),
    label=st.sampled_from(["P10", "P50", "P90", "r2", "Q4", "E7"]),
)
def test_a_number_after_a_label_with_digits_is_found(value, style, label) -> None:
    """ "P50 276,078" and "r2 0.77": the label's digits are not a thousands group."""
    text, written = LARGE[style](value)
    finding = Finding(
        claim=f"The forecast gives {label} {text} in 2012.",
        evidence_ids=["E1"],
        confidence="fact",
        numbers=[DeclaredNumber(text=text, value=written, evidence_id="E1")],
    )

    assert check_submission(Submission(summary="Done.", findings=[finding]), log_of(value)) == []


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LEGACY)))
def test_legacy_text_reading_accepts_every_format(value, style) -> None:
    """Answers without declared numbers fall back to reading them from the text."""
    text, written = LEGACY[style](value)

    assert check(text, written, value, numbers=None) == []
