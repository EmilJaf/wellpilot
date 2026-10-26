from datetime import date

import pandas as pd
import pytest

from wellpilot.data.store import ProductionStore, QueryError
from wellpilot.ingest.production import DAILY_FILE, ISSUES_FILE


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
        {
            "date": [dates[2], dates[8]],
            "well": ["F-12", "F-12"],
            "column": ["bhp_bar", "bhp_bar"],
            "issue": ["zero_gauge_reading"] * 2,
            "raw_value": [0.0, 0.0],
            "action": ["set_missing"] * 2,
        }
    )
    daily.to_parquet(tmp_path / DAILY_FILE, index=False)
    issues.to_parquet(tmp_path / ISSUES_FILE, index=False)
    return tmp_path


def test_select_returns_rows(processed_dir) -> None:
    store = ProductionStore(processed_dir)

    result = store.query(
        "SELECT well, count(*) AS n FROM production_daily GROUP BY well ORDER BY well"
    )

    assert result.rows.to_dict("records") == [{"well": "F-12", "n": 10}, {"well": "F-14", "n": 10}]
    assert not result.truncated


def test_as_of_hides_later_rows_in_every_table(processed_dir) -> None:
    store = ProductionStore(processed_dir, as_of=date(2010, 1, 5))

    daily = store.query("SELECT max(date) AS last, count(*) AS n FROM production_daily").rows
    issues = store.query("SELECT count(*) AS n FROM production_issues").rows

    assert daily.loc[0, "last"] == pd.Timestamp("2010-01-05")
    assert daily.loc[0, "n"] == 10
    assert issues.loc[0, "n"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE production_daily",
        "INSERT INTO production_daily (well) VALUES ('X')",
        "SELECT 1; DROP TABLE production_daily",
        "SET enable_external_access = true",
        "ATTACH 'other.db'",
    ],
)
def test_rejects_anything_but_one_select(processed_dir, sql) -> None:
    store = ProductionStore(processed_dir)

    with pytest.raises(QueryError, match="single SELECT"):
        store.query(sql)

    assert store.query("SELECT count(*) AS n FROM production_daily").rows.loc[0, "n"] == 20


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/etc/hosts')",
        "SELECT * FROM '/etc/hosts'",
    ],
)
def test_select_cannot_read_files(processed_dir, sql) -> None:
    store = ProductionStore(processed_dir)

    with pytest.raises(QueryError):
        store.query(sql)


def test_rows_are_capped(processed_dir) -> None:
    store = ProductionStore(processed_dir, max_rows=5)

    result = store.query("SELECT * FROM production_daily")

    assert len(result.rows) == 5
    assert result.truncated


def test_long_query_times_out(processed_dir) -> None:
    store = ProductionStore(processed_dir, timeout_s=0.2)

    with pytest.raises(QueryError, match="longer than"):
        store.query("SELECT sum(hash(i)) FROM range(100000000000) t(i)")


def test_bad_sql_is_a_query_error(processed_dir) -> None:
    store = ProductionStore(processed_dir)

    with pytest.raises(QueryError):
        store.query("SELECT nope FROM production_daily")


def test_well_daily_filters_and_orders(processed_dir) -> None:
    store = ProductionStore(processed_dir)

    rows = store.well_daily("F-14", start=date(2010, 1, 3), end=date(2010, 1, 6))

    assert rows["date"].dt.day.tolist() == [3, 4, 5, 6]
    assert set(rows["well"]) == {"F-14"}
