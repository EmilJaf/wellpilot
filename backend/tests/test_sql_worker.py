import resource
import sys
import time

import pandas as pd
import pytest

from wellpilot.data import sql_worker
from wellpilot.data.store import ProductionStore, QueryError
from wellpilot.ingest.production import DAILY_FILE, ISSUES_FILE

# A lambda that captures a list squares it: no list of function names closes this, so
# the worker's memory cap is what stops it (2.2 GB in the server's process before).
SQUARED = (
    "WITH a AS (SELECT list(range) l FROM range(30000)) "
    "SELECT len(flatten(list_transform(l, lambda x: l))) AS n FROM a"
)


@pytest.fixture
def processed_dir(tmp_path):
    dates = pd.date_range("2010-01-01", periods=10, freq="D")
    daily = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "well": ["F-12"] * 10 + ["F-14"] * 10,
            "oil_sm3": [float(1000 - i) for i in range(10)] * 2,
        }
    )
    issues = pd.DataFrame(
        {"date": [dates[2]], "well": ["F-12"], "field": ["bhp_bar"],
         "issue": ["zero_gauge_reading"], "raw_value": [0.0], "action": ["set_missing"]}
    )  # fmt: skip
    daily.to_parquet(tmp_path / DAILY_FILE, index=False)
    issues.to_parquet(tmp_path / ISSUES_FILE, index=False)
    return tmp_path


@pytest.fixture
def isolated(processed_dir):
    store = ProductionStore(processed_dir, as_of=pd.Timestamp("2010-01-05").date(), isolated=True)
    yield store
    store.isolated.close()


def test_isolated_results_equal_the_in_process_ones(processed_dir, isolated) -> None:
    plain = ProductionStore(processed_dir, as_of=pd.Timestamp("2010-01-05").date())
    for sql in (
        "SELECT * FROM production_daily ORDER BY well, date",
        "SELECT well, max(date) AS last, sum(oil_sm3) AS oil FROM production_daily GROUP BY well",
    ):
        a, b = isolated.query_untrusted(sql), plain.query_untrusted(sql)
        pd.testing.assert_frame_equal(a.rows, b.rows)
        assert a.truncated == b.truncated
    assert isolated.query_untrusted("SELECT max(date) AS d FROM production_daily").rows.loc[
        0, "d"
    ] == pd.Timestamp("2010-01-05")  # the worker cuts at the same date


def test_errors_come_back_and_the_worker_stays(isolated) -> None:
    with pytest.raises(QueryError, match="nope"):
        isolated.query_untrusted("SELECT * FROM nope")
    with pytest.raises(QueryError, match="range"):
        isolated.query_untrusted("SELECT len(range(20000000)) AS n")
    pid = isolated.isolated._worker.proc.pid

    assert len(isolated.query_untrusted("SELECT 1 AS one").rows) == 1
    assert isolated.isolated._worker.proc.pid == pid


def test_a_dead_worker_is_an_error_and_the_next_query_starts_another(isolated) -> None:
    isolated.query_untrusted("SELECT 1 AS one")
    isolated.isolated._worker.proc.kill()

    with pytest.raises(QueryError, match="stopped"):
        isolated.query_untrusted("SELECT 1 AS one")
    assert not isolated.isolated.running
    assert len(isolated.query_untrusted("SELECT 2 AS two").rows) == 1


def test_a_worker_past_its_time_is_killed(isolated, monkeypatch) -> None:
    isolated.query_untrusted("SELECT 1 AS one")
    proc = isolated.isolated._worker.proc
    monkeypatch.setattr(isolated.isolated, "timeout_s", -sql_worker.GRACE_S + 0.2)
    started = time.monotonic()

    with pytest.raises(QueryError, match="stopped"):
        isolated.query_untrusted("SELECT count(*) AS n FROM range(100000000000)")

    assert time.monotonic() - started < 3 and proc.poll() is not None


def test_one_worker_runs_at_a_time_and_idle_ones_stop(processed_dir, monkeypatch) -> None:
    monkeypatch.setattr(sql_worker, "IDLE_S", 0.3)
    first = ProductionStore(processed_dir, isolated=True)
    second = ProductionStore(processed_dir, as_of=pd.Timestamp("2010-01-03").date(), isolated=True)
    try:
        first.query_untrusted("SELECT 1 AS one")
        second.query_untrusted("SELECT 1 AS one")
        assert second.isolated.running and not first.isolated.running

        time.sleep(1.0)
        assert not second.isolated.running
    finally:
        first.isolated.close()
        second.isolated.close()


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_DATA is enforced on Linux only")
def test_the_memory_cap_stops_a_squared_list(isolated) -> None:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with pytest.raises(QueryError, match="(?i)memory"):
        isolated.query_untrusted(SQUARED)

    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 100 * 1024  # KB
    assert len(isolated.query_untrusted("SELECT 1 AS one").rows) == 1
