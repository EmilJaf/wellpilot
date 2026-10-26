"""Evidence log: every successful tool call becomes citable evidence with an id.

Each item records what ran (tool, arguments, a hash of the tool's source code) and what
it returned, so any number in the final answer can be traced to the code and data
that produced it.
"""

import hashlib
import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Any


@cache
def code_ref(fn: Callable) -> str:
    """'module.function@sha256[:12]' of the function's source."""
    digest = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:12]
    return f"{fn.__module__}.{fn.__qualname__}@{digest}"


@dataclass(frozen=True)
class Evidence:
    id: str
    tool: str
    arguments: dict
    result: Any
    code: str
    created_at: str
    warnings: tuple[str, ...] = ()


class EvidenceLog:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, tool: str, arguments: dict, result: Any, fn: Callable) -> Evidence:
        warnings = result.get("warnings", ()) if isinstance(result, dict) else ()
        evidence = Evidence(
            id=f"E{len(self._items) + 1}",
            tool=tool,
            arguments=arguments,
            result=result,
            code=code_ref(fn),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            warnings=tuple(warnings),
        )
        self._items[evidence.id] = evidence
        return evidence

    @classmethod
    def restore(cls, items: Iterable[dict]) -> "EvidenceLog":
        """A log rebuilt from saved evidence: a run file or a checker corpus case."""
        log = cls()
        for item in items:
            log._items[item["id"]] = Evidence(
                id=item["id"],
                tool=item["tool"],
                arguments=item["arguments"],
                result=item["result"],
                code=item.get("code", ""),
                created_at=item.get("created_at", ""),
                warnings=tuple(item.get("warnings", ())),
            )
        return log

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
