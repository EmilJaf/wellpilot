"""Request bodies refused over a size before anything reads them.

FastAPI parses a whole JSON body before a model's length checks run, and a 422 echoed the
input back: a 20 MB POST grew the process by about 1 GB, the size of the machine (audit,
Oct 31). Every body is read here up to its route's limit first, chunked ones included.
"""

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]


class BodyLimit:
    def __init__(self, app: App, default: int, limits: dict[str, int] | None = None):
        self.app = app
        self.default = default
        self.limits = limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        limit = self.limits.get(scope["path"], self.default)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None:
            try:
                declared = int(length[:20])
            except ValueError:
                declared = limit + 1
            if declared > limit:
                await _refuse(send, limit)
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return  # the client went away
            body += message.get("body", b"")
            if len(body) > limit:
                await _refuse(send, limit)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


async def _refuse(send: Send, limit: int) -> None:
    payload = json.dumps({"detail": f"request body over {limit:,} bytes"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
