"""What one visitor can do to the public server (audit, Oct 31): a huge body, a long edit
for the checker, a fresh IPv6 address per request."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from wellpilot.agent.edits import check_edits
from wellpilot.agent.evidence import EvidenceLog
from wellpilot.api.bodylimit import BodyLimit
from wellpilot.api.runs import client_id

RUNS = Path(__file__).parent / "fixtures" / "runs"


async def echo(scope, receive, send):
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(len(body)).encode()})


def client(limits=None) -> httpx.AsyncClient:
    app = BodyLimit(echo, default=1000, limits=limits or {"/big": 5000})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_a_body_over_its_routes_limit_is_refused_before_anything_reads_it():
    async with client() as c:
        assert (await c.post("/x", content=b"a" * 1000)).text == "1000"
        refused = await c.post("/x", content=b"a" * 1001)
        assert refused.status_code == 413 and "1,000 bytes" in refused.json()["detail"]
        assert (await c.post("/big", content=b"a" * 4000)).text == "4000"
        assert (await c.post("/big", content=b"a" * 6000)).status_code == 413


async def test_a_chunked_body_with_no_length_is_counted_too():
    async def chunks():
        for _ in range(5):
            yield b"a" * 400

    async with client() as c:
        assert (await c.post("/x", content=chunks())).status_code == 413


async def test_reads_are_not_limited():
    async with client() as c:
        assert (await c.get("/x")).status_code == 200


def request_from(ip: str):
    return SimpleNamespace(headers={"fly-client-ip": ip}, client=None)


def test_an_ipv6_visitor_counts_by_its_64():
    a = client_id(request_from("2a01:4f8:1:2:aaaa::1"), "fly")
    b = client_id(request_from("2a01:4f8:1:2:bbbb::9"), "fly")
    other = client_id(request_from("2a01:4f8:1:3::1"), "fly")
    assert a == b == "2a01:4f8:1:2::/64" and other != a
    assert client_id(request_from("203.0.113.7"), "fly") == "203.0.113.7"
    assert client_id(request_from("::ffff:203.0.113.7"), "fly") == "203.0.113.7"


def test_a_long_reader_edit_is_checked_in_well_under_a_second():
    """The checker's search for a digit group before a number scanned from the start of
    the text for every number: 9-14 s for a 2,000-character edit."""
    run = json.loads((RUNS / "99d66f7a3adb.json").read_text())
    log = EvidenceLog.restore(run["evidence"].values())
    claim = ("55% " * 200)[:800]  # the longest edit the API accepts
    started = time.monotonic()
    check_edits(run["question"], run["result"], log, [(1, claim, "fact")], None)
    assert time.monotonic() - started < 2.0
