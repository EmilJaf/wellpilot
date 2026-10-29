"""Shared test setup."""

import os

import pytest


@pytest.fixture(autouse=True)
def no_modal(monkeypatch) -> None:
    """No test starts a Modal Sandbox: where Modal is installed and logged in, an app made
    by a test would start one per run and leave it running on the account. run_python is
    tested with a local stand-in (test_sandbox.py)."""
    monkeypatch.setenv("PYTHON_SANDBOX", "false")
    monkeypatch.setenv("ISOLATE_SQL", "false")  # a worker process per store: test_sql_worker.py
    for where in ("wellpilot.main", "wellpilot.eval", "wellpilot.agent.__main__"):
        monkeypatch.setattr(f"{where}.modal_pool", lambda *_: None)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    """TEST_SHARD=k/n keeps every n-th test from the k-th on (after -m and -k), so CI runs
    the suite as n parallel jobs: a parametrized suite's cases spread over all of them,
    where a split by file would leave the metamorphic tests (a third of the time) in one."""
    shard = os.environ.get("TEST_SHARD")
    if not shard:
        return
    k, n = (int(part) for part in shard.split("/"))
    kept = items[k - 1 :: n]
    config.hook.pytest_deselected(items=[item for i, item in enumerate(items) if i % n != k - 1])
    items[:] = kept
