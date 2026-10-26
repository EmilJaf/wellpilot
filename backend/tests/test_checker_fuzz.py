"""Property-based tests: the checker must accept any number written in a supported format
when the evidence has it, and reject numbers that are not there, whatever the format.

Each format below is one way a model might write a value; when a model invents a new
one, add it here first and let the tests say whether the checker copes.
"""

import math
from collections.abc import Callable

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.evidence import EvidenceLog


def _tool() -> None:
    """Stands in for a tool function."""


def log_of(*values: float) -> EvidenceLog:
    log = EvidenceLog()
    log.add("tool", {}, {"date": "2012-06-01", "values": list(values)}, _tool)
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
}
FRACTION: dict[str, Format] = {
    "percent": lambda f: (f"{f * 100:.1f}%", round(f * 100, 1)),
    "fraction": lambda f: (f"{f:.3f}", round(f, 3)),
}

large = st.floats(min_value=12_000, max_value=9_000_000, allow_nan=False)
fraction = st.floats(min_value=0.02, max_value=0.98, allow_nan=False)
off = st.floats(min_value=0.05, max_value=2)


def claim_with(text: str) -> str:
    return f"In 2012 the well made {text} of oil."


def coarsest(written: float) -> float:
    """Half a unit in the leading digit: the loosest any rounding of written can be."""
    return 0.5 * 10 ** math.floor(math.log10(written))


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


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)), off=off)
def test_declared_numbers_that_are_not_in_the_evidence_fail(value, style, off) -> None:
    text, written = LARGE[style](value * (1 + off))
    assume(abs(written - value) > coarsest(written))  # no rounding of the text covers value

    problems = check(text, written, value)

    assert problems and "is not in E1" in problems[0]


@settings(max_examples=150, deadline=None)
@given(value=large, style=st.sampled_from(sorted(LARGE)), off=off)
def test_text_that_does_not_show_its_declared_value_fails(value, style, off) -> None:
    """The evidence value declared honestly, the text written wrong: '2.00 million' for 1.6M."""
    text, written = LARGE[style](value * (1 + off))
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
@given(value=large, style=st.sampled_from(sorted(LARGE)))
def test_legacy_text_reading_accepts_every_format(value, style) -> None:
    """Answers without declared numbers fall back to reading them from the text."""
    text, written = LARGE[style](value)

    assert check(text, written, value, numbers=None) == []
