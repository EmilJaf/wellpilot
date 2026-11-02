"""Security headers on every response.

The page runs only its own bundle (no inline script, no eval, nothing from another origin)
and talks only to this server, so a strict policy costs nothing: should anything ever slip
text into the page as markup, the browser still runs none of it and sends nothing elsewhere.
Inline styles stay allowed (charts set them). Framing is allowed: nothing on the page acts
for a signed-in user. The microphone is for this page only (Ask by voice).
"""

from wellpilot.api.bodylimit import App, Message, Receive, Scope, Send

POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "media-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)

HEADERS = [
    (b"content-security-policy", POLICY.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"permissions-policy", b"microphone=(self), camera=(), geolocation=(), payment=()"),
]


class SecurityHeaders:
    def __init__(self, app: App):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [
                    (k, v) for k, v in HEADERS if k not in present
                ]
            await send(message)

        await self.app(scope, receive, with_headers)
