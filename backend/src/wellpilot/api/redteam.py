"""POST /api/runs/{id}/attack: the model tries to get a false claim past the checker.

The "Try to fool the checker" panel lets a reader edit a finding; this lets the model do it
instead (agent.redteam): it writes a few false versions of one verified finding, and each
goes through the checker against the run's saved evidence. It costs tokens, so it has its
own per-client limit, counts against the daily budget, and is refused without a key.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from wellpilot.agent.redteam import attack, judge
from wellpilot.api.runs import Refused, RunManager, _record, client_id

router = APIRouter(prefix="/api", tags=["redteam"])


class AttackRequest(BaseModel):
    index: int = Field(ge=1, le=50, description="the finding, 1-based")
    count: int = Field(3, ge=1, le=4)


@router.post("/runs/{run_id}/attack")
async def attack_finding(run_id: str, body: AttackRequest, request: Request) -> dict:
    manager: RunManager = request.app.state.runs
    record = _record(request, run_id)
    if not record.done:
        raise HTTPException(409, "the run is still going")
    result = record.result or {}
    findings = result.get("findings") or []
    if not findings:
        raise HTTPException(409, "the run has no answer to attack")
    if body.index > len(findings):
        raise HTTPException(422, f"the answer has no finding {body.index}")
    privileged = manager.is_operator(request.headers.get("x-operator-key"))
    try:
        llm = manager.llm_factory()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    try:
        manager.allow_attack(client_id(request, manager.trusted_proxy), privileged)
    except Refused as e:
        raise HTTPException(429, str(e)) from None
    finding = findings[body.index - 1]
    cited = [record.evidence[i] for i in finding["evidence_ids"] if i in record.evidence]
    try:
        attempts, tokens = await attack(llm, record.question, finding, cited, body.count)
    except TimeoutError:
        raise HTTPException(504, "the model took too long; try again") from None
    except ValidationError:
        raise HTTPException(502, "the model did not return attempts") from None
    except Exception as e:  # a provider error: say the type, never the body
        raise HTTPException(502, f"the model failed ({type(e).__name__})") from None
    manager.charge(tokens)
    # Judging runs the checker, CPU-bound: in a thread, so other requests are served.
    judged = await asyncio.to_thread(
        judge, record.question, result, record.evidence, body.index, attempts
    )
    return {
        "attempts": judged,
        "model": llm.model,
        "tokens": tokens,
    }
