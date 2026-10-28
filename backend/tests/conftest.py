"""Shared test setup."""

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
