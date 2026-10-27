"""What each Meta model did for one answer, for the line the answer panel shows.

A reader should not have to guess where Muse Spark Open and SAM 3.1 were at work: the
model that planned, ran the tools, showed its reasoning, looked at report pages and
reviewed its own answer; SAM, whose measurements of a chart checked the model's reading of
it, and whose reading of a picture-only production log the agent's evidence rests on. Built
from a run's events and evidence, so runs saved before it get it on load.
"""

LOOK = "look_at_page"
LOG = "production_log"


def meta_models(events: list[dict], evidence: dict[str, dict]) -> dict:
    started = next((e for e in events if e.get("type") == "run_started"), {})
    steps = [e for e in events if e.get("type") == "step"]
    calls = [
        c for s in steps for c in s.get("tool_calls", []) if c.get("name") != "submit_findings"
    ]
    review = next((e for e in events if e.get("type") == "review"), None)
    figures, logs = [], []
    for eid, item in evidence.items():
        result = item.get("result") or {}
        if item.get("tool") == LOOK and result.get("cross_check"):
            judged = [c for c in result["cross_check"] if c.get("agree") is not None]
            figures.append(
                {
                    "evidence_id": eid,
                    "agreed": sum(1 for c in judged if c["agree"]),
                    "compared": len(judged),
                }
            )
        if item.get("tool") == LOG and any(
            "sam-3.1" in str(s.get("read_from", "")) for s in result.get("surveys", [])
        ):
            logs.append(eid)
    return {
        "muse": {
            "model": started.get("model"),
            "steps": len(steps),
            "tool_calls": len(calls),
            "reasoning_shown": any(s.get("reasoning") for s in steps),
            "pages": [eid for eid, item in evidence.items() if item.get("tool") == LOOK],
            "second_opinion": review.get("outcome") if review else None,
        },
        "sam": {"figures": figures, "logs": logs},
    }
