"""A reader's edit of a finished answer, checked like the agent's own submission.

The answer panel lets a reader change a finding (a number, a cause, the fact label, a well
name, a quote) and press "Check again". The checker then runs on the edited claim against
the run's saved evidence. Declared numbers are the agent's; this module carries them over to
the edited text, so a changed number is checked as the number the reader wrote.
"""

import difflib
from dataclasses import asdict

from wellpilot.agent.checker import (
    WRITTEN_NUMBER,
    DeclaredNumber,
    Finding,
    Submission,
    _numbers_in_text,
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
        if all(tag == "equal" for tag, *_ in touched):
            carried.append(declared)
            continue
        lo = min(j1 + (start - i1 if tag == "equal" else 0) for tag, i1, i2, j1, j2 in touched)
        hi = max(j1 + (end - i1) if tag == "equal" else j2 for tag, i1, i2, j1, j2 in touched)
        tokens = [m for m in WRITTEN_NUMBER.finditer(edited) if m.start() < hi and m.end() > lo]
        if len(tokens) != 1:
            continue
        new_text = tokens[0].group(0).strip()
        reading = _reads(new_text, _convention(declared))
        if reading is None:
            continue
        carried.append(
            DeclaredNumber(text=new_text, value=reading[0], evidence_id=declared.evidence_id)
        )
    # Declarations written nowhere in the text stay as they were: the checker reads them as
    # another form of a number ("100%" covers a "1.0" written later), so an unedited claim
    # keeps its verdict.
    return carried + unused


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
                "numbers": redeclare(original.claim, original.numbers, claim),
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
