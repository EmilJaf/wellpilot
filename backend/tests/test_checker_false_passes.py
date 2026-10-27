"""False passes: wrong or unsupported answers the checker must reject.

The other half of test_checker_false_rejections.py: every rule that lets more answers
through gets a row here showing what it must still stop.
"""

import json

from wellpilot.agent.checker import DeclaredNumber, Finding, Submission, check_submission
from wellpilot.agent.corpus import CORPUS_DIR
from wellpilot.agent.evidence import EvidenceLog


def _tool() -> None:
    """Stands in for a tool function."""


def corpus_log(name: str) -> EvidenceLog:
    case = json.loads((CORPUS_DIR / f"{name}.json").read_text())
    return EvidenceLog.restore(case["evidence"])


def add(log: EvidenceLog, tool: str, arguments: dict, result) -> str:
    return log.add(tool, arguments, result, _tool).id


def d(text: str, value: float, eid: str) -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


def check(claim: str, ids: list[str], numbers: list, log: EvidenceLog, question="") -> list[str]:
    finding = Finding(claim=claim, evidence_ids=ids, confidence="fact", numbers=numbers)
    return check_submission(Submission(summary="Done.", findings=[finding]), log, question)


F12 = "6fe2c784147c-1"  # E5 detect_events for F-12, E9 a clean fit


def test_a_duration_that_is_not_a_window_still_needs_evidence() -> None:
    log = corpus_log(F12)

    assert check("F-12 was shut in for 30 days of maintenance.", ["E5"], [], log)
    assert check("A 45-day shut-in followed.", ["E5"], [], log)


def test_a_small_integer_declared_with_another_value_is_caught() -> None:
    log = corpus_log(F12)

    problems = check("F-12 is one of 3 producers.", ["E5"], [d("3", 4, "E5")], log)

    assert problems and "does not show its declared value" in problems[0]


def test_only_whole_numbers_from_the_question_are_free() -> None:
    log = corpus_log(F12)
    question = "Did F-12 lose 72.5% of its rate after 2010?"

    assert check("F-12 lost 72.5% of its rate.", ["E5"], [], log, question)


def test_a_year_needs_evidence_of_that_year() -> None:
    log = corpus_log(F12)

    assert check("The gauge failed in 2003.", ["E5"], [], log)


def test_a_literal_echoed_from_the_select_list_is_still_not_evidence() -> None:
    log = EvidenceLog()
    eid = add(
        log,
        "run_sql",
        {"sql": "SELECT DATE '2013-05-17' AS d, 4321 AS n FROM production_daily LIMIT 1"},
        {"columns": ["d", "n"], "rows": [["2013-05-17", 4321]]},
    )

    problems = check("On 2013-05-17 the rate was 4 321 Sm3/d.", [eid], [d("4 321", 4321, eid)], log)

    assert any("2013-05-17" in p for p in problems)
    assert any("4 321" in p for p in problems)
