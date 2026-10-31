"""What each Meta model did for one answer, for the line the answer panel shows.

A reader should not have to guess where Muse Spark Open and SAM 3.1 were at work: the
model that planned, ran the tools, showed its reasoning, looked at report pages and
reviewed its own answer; SAM, whose measurements of a chart checked the model's reading of
it, and whose reading of a picture-only production log the agent's evidence rests on. Built
from a run's events and evidence, so runs saved before it get it on load.
"""

LOOK = "look_at_page"
LOG = "production_log"


# Not SAM's reading: what the report itself says, depths, and the colour-trace check.
NOT_READ = ("report_says", "perforated_zones", "agreement_with_colour", "depth", "_m")


def _numbers(node) -> list[float]:
    """The values SAM read off a survey's pictures (shares and rates)."""
    if isinstance(node, bool):
        return []
    if isinstance(node, int | float):
        return [float(node)]
    if isinstance(node, dict):
        return [
            n
            for k, v in node.items()
            if not any(x in str(k) for x in NOT_READ)
            for n in _numbers(v)
        ]
    if isinstance(node, list):
        return [n for v in node for n in _numbers(v)]
    return []


def _depths(node, under: bool = False) -> list[float]:
    """Every number a survey result gives under a depth's key."""
    if isinstance(node, bool):
        return []
    if isinstance(node, int | float):
        return [float(node)] if under else []
    if isinstance(node, dict):
        return [
            n
            for k, v in node.items()
            for n in _depths(v, under or "_m" in str(k) or "depth" in str(k))
        ]
    if isinstance(node, list):
        return [n for v in node for n in _depths(v, under)]
    return []


def _near(value: float, values: list[float]) -> bool:
    return any(abs(abs(value) - abs(x)) <= max(abs(x) * 0.005, 1e-9) for x in values)


def _answer_numbers(result: dict | None) -> list[tuple[str, float]]:
    """(evidence id, value) of every number the answer declares."""
    if not result:
        return []
    declared = list(result.get("summary_numbers") or [])
    declared += [n for f in result.get("findings") or [] for n in f.get("numbers") or []]
    return [
        (str(n.get("evidence_id")), float(n["value"]))
        for n in declared
        if isinstance(n.get("value"), int | float)
    ]


def meta_models(events: list[dict], evidence: dict[str, dict], answer: dict | None = None) -> dict:
    started = next((e for e in events if e.get("type") == "run_started"), {})
    steps = [e for e in events if e.get("type") == "step"]
    calls = [
        c for s in steps for c in s.get("tool_calls", []) if c.get("name") != "submit_findings"
    ]
    review = next((e for e in events if e.get("type") == "review"), None)
    answers = _answer_numbers(answer)
    figures, logs = [], []
    for eid, item in evidence.items():
        result = item.get("result") or {}
        if item.get("tool") == LOOK and isinstance(result, dict) and result.get("cross_check"):
            judged = [c for c in result["cross_check"] if c.get("agree") is not None]
            figures.append(
                {
                    "evidence_id": eid,
                    "agreed": sum(1 for c in judged if c["agree"]),
                    "compared": len(judged),
                    # Figure readings SAM had nothing to set against, or that were no length
                    "not_compared": sum(
                        1
                        for c in result["cross_check"]
                        if c.get("agree") is None and "not the figure" not in str(c.get("note"))
                    ),
                }
            )
        # A production log is credited to SAM only when the answer states a number from the
        # survey SAM read off its pictures, not when the agent merely called the tool.
        surveys = result.get("surveys", []) if isinstance(result, dict) else []
        read = [s for s in surveys if "sam-3.1" in str(s.get("read_from", ""))]
        values = [v for s in read for v in _numbers(s)]
        # A number the other surveys or the well's depths also hold is no credit to SAM: a depth
        # of 3147 m sits next to a rate of 3150 m3/d, and a share of 0.98 is in both logs.
        others = [v for s in surveys if s not in read for v in _numbers(s)] + _depths(result)
        used = [
            v
            for source, v in answers
            if source == eid and _near(v, values) and not _near(v, others)
        ]
        if item.get("tool") == LOG and read and used:
            logs.append(eid)
    return {
        "muse": {
            "model": started.get("model"),
            "steps": len(steps),
            "tool_calls": len(calls),
            "reasoning_shown": any(s.get("reasoning") for s in steps),
            "pages": [eid for eid, item in evidence.items() if item.get("tool") == LOOK],
            # The answer keeps the review's outcome; the review event is sent before the
            # revision it may lead to, so its own outcome can still be empty.
            "second_opinion": ((answer or {}).get("review") or {}).get("outcome")
            or (review.get("outcome") if review else None),
        },
        "sam": {"figures": figures, "logs": logs},
    }
