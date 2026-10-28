"""A reader's edit of a finished answer, checked like the agent's own submission.

The answer panel lets a reader change a finding (a number, a cause, the fact label, a well
name, a quote) and press "Check again". The checker then runs on the edited claim against
the run's saved evidence. Declared numbers are the agent's; this module carries them over to
the edited text, so a changed number is checked as the number the reader wrote.
"""

import difflib
from dataclasses import asdict

from wellpilot.agent.checker import (
    QUOTED,
    WRITTEN_NUMBER,
    DeclaredNumber,
    Finding,
    Submission,
    _numbers_in_text,
    _Pool,
    _readings,
    _reads_as,
    check_submission,
    display_spans,
)
from wellpilot.agent.evidence import EvidenceLog


def _reads(text: str, european: bool) -> tuple[float, int, str | None] | None:
    found = _numbers_in_text(text, european)
    return found[0] if len(found) == 1 else None


def _convention(number: DeclaredNumber) -> bool:
    """Was the declaration read the Norwegian way ("9.000" declared 9000)? The reading whose
    value is closest to the declared one, at any of the scales the checker allows."""

    def distance(european: bool) -> float:
        reading = _reads(number.text, european)
        if reading is None:
            return float("inf")
        return min(abs(reading[0] * scale - number.value) for scale in (1, 0.01, 100))

    return distance(True) < distance(False)


def redeclare(
    text: str, numbers: list[DeclaredNumber] | None, edited: str
) -> list[DeclaredNumber] | None:
    """The declarations of `text` carried to `edited`: a number left alone keeps its
    declaration; one the edit replaced is declared anew as written, with the same evidence
    id, so the checker looks for the reader's number in the agent's source; one the edit
    removed, or turned into several, is dropped; one the text never wrote is kept. Numbers
    typed elsewhere stay undeclared, and the checker says so. None (numbers read from the
    text) stays None."""
    if numbers is None:
        return None
    unused = list(numbers)
    spans = [s for s in display_spans(text, numbers) if s["kind"] == "number"]
    opcodes = difflib.SequenceMatcher(None, text, edited, autojunk=False).get_opcodes()
    carried: list[DeclaredNumber] = []
    for span in spans:
        start, end = span["start"], span["end"]
        written = text[start:end].replace("−", "-").replace("–", "-")
        declared = next((n for n in unused if n.text.replace("−", "-") == written), None)
        if declared is None:
            continue
        unused.remove(declared)
        # Blocks the number sits in; an insertion counts only strictly inside it, so text
        # typed right before or after a number leaves it alone.
        touched = [
            (tag, i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in opcodes
            if ((i1 < end and i2 > start) if i1 < i2 else (start < i1 < end))
        ]
        lo = min(j1 + (start - i1 if tag == "equal" else 0) for tag, i1, i2, j1, j2 in touched)
        hi = max(j1 + (end - i1) if tag == "equal" else j2 for tag, i1, i2, j1, j2 in touched)
        tokens = [m for m in WRITTEN_NUMBER.finditer(edited) if m.start() < hi and m.end() > lo]
        if all(tag == "equal" for tag, *_ in touched) and (
            len(tokens) != 1 or tokens[0].group(0).strip() == written
        ):
            carried.append(declared)
            continue
        # Digits typed against it make another number ("0." before "69%"): that one is
        # checked, with the same evidence id.
        if len(tokens) != 1:
            continue
        new_text = tokens[0].group(0).strip()
        # Two numbers swapped ("from 0.97 to 0.31" -> "from 0.31 to 0.97"): each keeps the
        # agent's own declaration, with its full value.
        swapped = next((n for n in numbers if n.text == new_text and n is not declared), None)
        if swapped is not None:
            if swapped not in carried:
                carried.append(swapped)
            continue
        reading = _reads(new_text, _convention(declared))
        if reading is None:
            continue
        carried.append(
            DeclaredNumber(text=new_text, value=reading[0], evidence_id=declared.evidence_id)
        )
    # A number the reader only moved ("69% water cut" -> "a water cut of 69%", "from 186 to
    # 862" -> "to 862 from 186") is the same number: its declaration follows it.
    dropped = [
        n
        for span in spans
        if (n := next((d for d in numbers if d.text == text[span["start"] : span["end"]]), None))
        and n not in carried
        and n not in unused
    ]

    def exact(written: str) -> float | None:
        """A number as written, a percent as its fraction: "69%" and "0.69" are equal."""
        found = _numbers_in_text(written, False)
        if len(found) != 1:
            return None
        value, _, suffix = found[0]
        return round(value / 100 if suffix in ("%", "percent", "pct") else value, 9)

    declared_texts = {c.text for c in carried}
    for n in dropped:
        tokens = [m.group(0).strip() for m in WRITTEN_NUMBER.finditer(edited)]
        if n.text in tokens and n.text not in declared_texts:
            carried.append(n)
            declared_texts.add(n.text)
            continue
        # Written in another form ("69%" as "0.69"): the same value, declared as written.
        same = [t for t in tokens if t not in declared_texts and exact(t) == exact(n.text)]
        if len(same) == 1:
            reading = _reads(same[0], False)
            if reading is not None:
                carried.append(
                    DeclaredNumber(text=same[0], value=reading[0], evidence_id=n.evidence_id)
                )
                declared_texts.add(same[0])
    # A number moved inside a quote is checked with the quote, word for word: a declaration
    # would read the plan's "9.000" as nine.
    quoted = [(m.start(), m.end()) for m in QUOTED.finditer(edited)]
    carried = [
        c
        for c in carried
        if not any(
            a <= m.start() and m.end() <= b
            for m in WRITTEN_NUMBER.finditer(edited)
            if m.group(0).strip() == c.text
            for a, b in quoted
        )
    ]
    # Declarations written nowhere in the text stay as they were: the checker reads them as
    # another form of a number ("100%" covers a "1.0" written later), so an unedited claim
    # keeps its verdict.
    return carried + unused


def declare_typed(
    edited: str, numbers: list[DeclaredNumber] | None, cited: list[str], log: EvidenceLog
) -> list[DeclaredNumber] | None:
    """Numbers a reader typed that no declaration carries ("2 years" rewritten as "19
    months"), declared from the first cited result that holds them, as the agent would have
    declared them. One no cited result holds stays undeclared, and the checker says so."""
    if numbers is None:
        return None
    quoted = [(m.start(), m.end()) for m in QUOTED.finditer(edited)]
    texts = {n.text for n in numbers}
    added: list[DeclaredNumber] = []
    for m in WRITTEN_NUMBER.finditer(edited):
        token = m.group(0).strip()
        if token in texts or any(a <= m.start() and m.end() <= b for a, b in quoted):
            continue
        reading = _reads(token, False)
        # Small numbers are found anywhere ("3" in "sam-3.1"): those stay the reader's to
        # declare. A percentage must be in the source as the share it is.
        if reading is None or abs(reading[0]) <= 10:
            continue
        percent = reading[2] in ("%", "percent", "pct")
        readings = [(v / 100, u / 100, sc) for v, u, sc in _readings(token)] if percent else None
        source = next(
            (
                e
                for e in cited
                if (item := log.get(e))
                and any(_reads_as(r, _Pool([item]).numbers) for r in readings or _readings(token))
            ),
            None,
        )
        if source:
            added.append(DeclaredNumber(text=token, value=reading[0], evidence_id=source))
            texts.add(token)
    return numbers + added


def check_edits(
    question: str,
    result: dict,
    log: EvidenceLog,
    edits: list[tuple[int, str, str]],
    summary: str | None = None,
) -> dict:
    """The checker on edits of a finished answer (`result`, as a run saves it): each edit is
    (finding index from 1, claim, confidence). Declared numbers are carried to the edited
    text; only the edited parts' problems come back, so an unedited finding never changes
    the verdict. Raises IndexError for a finding the answer does not have."""
    findings = [Finding.model_validate(f) for f in result["findings"]]
    edited: set[int] = set()
    for index, claim, confidence in edits:
        if not 1 <= index <= len(findings):
            raise IndexError(f"the answer has no finding {index}")
        original = findings[index - 1]
        findings[index - 1] = original.model_copy(
            update={
                "claim": claim,
                "confidence": confidence,
                "numbers": declare_typed(
                    claim,
                    redeclare(original.claim, original.numbers, claim),
                    original.evidence_ids,
                    log,
                ),
            }
        )
        edited.add(index)
    summary_numbers = [
        DeclaredNumber.model_validate(n) for n in result.get("summary_numbers") or []
    ]
    text = result.get("summary") or ""
    if summary is not None:
        summary_numbers = redeclare(text, summary_numbers, summary) or []
        text = summary
    submission = Submission(summary=text, findings=findings, summary_numbers=summary_numbers)
    problems = check_submission(submission, log, question)
    labels = {f"Finding {n}" for n in edited} | ({"Summary"} if summary is not None else set())
    problems = [p for p in problems if any(p.startswith(label) for label in labels)]
    checked = []
    for n in sorted(edited):
        finding = findings[n - 1]
        cited = EvidenceLog.restore(asdict(e) for e in log if e.id in set(finding.evidence_ids))
        checked.append(
            {
                "index": n,
                "claim": finding.claim,
                "confidence": finding.confidence,
                "numbers": [d.model_dump() for d in finding.numbers or []],
                "spans": display_spans(finding.claim, finding.numbers, cited),
            }
        )
    return {"accepted": not problems, "problems": problems, "findings": checked}
