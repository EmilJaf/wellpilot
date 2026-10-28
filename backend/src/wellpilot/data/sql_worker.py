"""Agent SQL in a child process under a hard memory cap.

DuckDB's memory_limit covers its buffer manager, not the lists scalar functions build:
len(range(2e7)) took 2.4 GB and flatten(list_transform(l, lambda x: l)) over a 10,000-item
list 2.2 GB, in the server's own process. Naming such functions one by one cannot close
that (a lambda that captures a list squares it), so on the public server agent SQL runs in a
worker process holding its own copy of the store, under RLIMIT_DATA on Linux. A query past
the cap fails inside the worker, or kills it, never the server. The worker is started on
first use, stopped after IDLE_S seconds without a query, and at most MAX_LIVE run at once.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import resource
import selectors
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import ClassVar

# Bytes of data segment and private mappings per worker. Allocators reserve far more than
# they touch: the worker needs 640 MB to load (peak RSS 280 MB on a full-table sort).
MEMORY_CAP = 768 * 2**20
IDLE_S = 90.0
MAX_LIVE = 1  # a 1 GB machine: the server takes about 375 MB, a worker up to the cap
START_S = 60.0  # importing pandas and loading the tables, on a cold machine
GRACE_S = 3.0  # on top of the store's own query timeout
STOPPED = (
    "the query used more memory or time than a query may and was stopped: aggregate in "
    "SQL instead of building large lists or strings"
)


class WorkerDied(RuntimeError):
    pass


def _read(fd: int, n: int, deadline: float) -> bytes:
    """Exactly n bytes from fd, or WorkerDied on end of file or the deadline."""
    chunks, got = [], 0
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_READ)
        while got < n:
            left = deadline - time.monotonic()
            if left <= 0 or not sel.select(left):
                raise WorkerDied("timed out")
            chunk = os.read(fd, n - got)
            if not chunk:
                raise WorkerDied("exited")
            chunks.append(chunk)
            got += len(chunk)
    return b"".join(chunks)


class _Worker:
    def __init__(self, processed_dir: Path, as_of: date | None, max_rows: int, timeout_s: float):
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",  # each BLAS thread reserves its own buffers
            OMP_NUM_THREADS="1",
            ARROW_DEFAULT_MEMORY_POOL="system",  # jemalloc reserves past the cap
            MALLOC_ARENA_MAX="2",
        )
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "wellpilot.data.sql_worker", str(processed_dir),
                as_of.isoformat() if as_of else "-", str(max_rows), str(timeout_s),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )  # fmt: skip
        try:
            if _read(self.proc.stdout.fileno(), 6, time.monotonic() + START_S) != b"ready\n":
                raise WorkerDied("bad handshake")
        except WorkerDied:
            self.kill()
            raise

    def ask(self, sql: str, wait_s: float):
        fd = self.proc.stdout.fileno()
        try:
            self.proc.stdin.write(json.dumps(sql).encode() + b"\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise WorkerDied("exited") from e
        deadline = time.monotonic() + wait_s
        size = int.from_bytes(_read(fd, 8, deadline), "big")
        # The worker is our own code: what it pickles is a QueryResult or an error text.
        return pickle.loads(_read(fd, size, deadline))  # noqa: S301

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for pipe in (self.proc.stdin, self.proc.stdout):
            if pipe:
                pipe.close()


class IsolatedSql:
    """Runs ProductionStore.query_untrusted for one store in a capped worker process."""

    _live: ClassVar[OrderedDict[int, IsolatedSql]] = OrderedDict()
    _live_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, processed_dir: Path, as_of: date | None, max_rows: int, timeout_s: float):
        self.args = (processed_dir, as_of, max_rows, timeout_s)
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._worker: _Worker | None = None
        self._idle: threading.Timer | None = None

    def query(self, sql: str):
        from wellpilot.data.store import QueryError

        with self._lock:
            if self._idle:
                self._idle.cancel()
            try:
                if self._worker is None:
                    self._make_room()
                    self._worker = _Worker(*self.args)
                with self._live_lock:
                    self._live[id(self)] = self
                    self._live.move_to_end(id(self))
                kind, value = self._worker.ask(sql, self.timeout_s + GRACE_S)
            except WorkerDied as e:
                self._stop()
                raise QueryError(STOPPED) from e
            finally:
                if self._worker is not None:
                    self._idle = threading.Timer(IDLE_S, self.close)
                    self._idle.daemon = True
                    self._idle.start()
        if kind == "error":
            raise QueryError(value)
        return value

    def _make_room(self) -> None:
        """Stop the least recently used other worker when MAX_LIVE are running."""
        with self._live_lock:
            others = [s for k, s in self._live.items() if k != id(self)]
        for other in others[: max(0, len(others) - MAX_LIVE + 1)]:
            if other._lock.acquire(blocking=False):  # a busy one finishes its query
                try:
                    other._stop()
                finally:
                    other._lock.release()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        with self._live_lock:
            self._live.pop(id(self), None)

    @property
    def running(self) -> bool:
        return self._worker is not None

    def close(self) -> None:
        with self._lock:
            if self._idle:
                self._idle.cancel()
            self._stop()


def _cap_memory() -> None:
    # Linux counts private anonymous mappings in RLIMIT_DATA, so DuckDB's lists are
    # covered; macOS does not enforce it (local development only).
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_DATA, (MEMORY_CAP, MEMORY_CAP))
        # if the machine runs out anyway, the kernel stops the worker, not the server
        with contextlib.suppress(OSError):
            Path("/proc/self/oom_score_adj").write_text("1000")


def main(argv: list[str]) -> None:
    _cap_memory()  # before pandas and duckdb load
    from wellpilot.data.store import ProductionStore, QueryError

    processed_dir, as_of, max_rows, timeout_s = argv
    store = ProductionStore(
        Path(processed_dir),
        as_of=None if as_of == "-" else date.fromisoformat(as_of),
        max_rows=int(max_rows),
        timeout_s=float(timeout_s),
    )
    out = sys.stdout.buffer
    out.write(b"ready\n")
    out.flush()
    for line in sys.stdin.buffer:  # end of file: the server closed the pipe or exited
        try:
            reply = ("ok", store.query_untrusted(json.loads(line)))
        except QueryError as e:
            reply = ("error", str(e))
        except MemoryError:
            reply = ("error", STOPPED)
        data = pickle.dumps(reply, protocol=5)
        out.write(len(data).to_bytes(8, "big") + data)
        out.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
