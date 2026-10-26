"""Read-only access to the processed production data.

The Parquet files are loaded into an in-memory DuckDB database; then file access is
switched off and the configuration locked. Agent-written SQL therefore sees only these
tables, can run only a single SELECT, and never sees rows after the store's `as_of` date,
which is what keeps backtests honest.
"""

import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from wellpilot.ingest.production import DAILY_FILE, ISSUES_FILE

TABLES = {"production_daily": DAILY_FILE, "production_issues": ISSUES_FILE}


class QueryError(ValueError):
    """The SQL was rejected, failed or ran too long."""


@dataclass(frozen=True)
class QueryResult:
    rows: pd.DataFrame
    truncated: bool


class ProductionStore:
    def __init__(
        self,
        processed_dir: Path,
        as_of: date | None = None,
        *,
        max_rows: int = 5000,
        timeout_s: float = 5.0,
    ) -> None:
        self.as_of = as_of
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._con = duckdb.connect(":memory:")
        for table, file in TABLES.items():
            frame = pd.read_parquet(processed_dir / file)
            if as_of is not None:
                frame = frame[frame["date"] <= pd.Timestamp(as_of)]
            self._con.register("_frame", frame)
            self._con.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
            self._con.unregister("_frame")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET memory_limit = '512MB'")
        self._con.execute("SET threads = 2")
        self._con.execute("SET lock_configuration = true")

    def query(self, sql: str, params: list | None = None) -> QueryResult:
        """Run one read-only SELECT, capped at max_rows rows and timeout_s seconds."""
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as e:
            raise QueryError(f"SQL does not parse: {e}") from e
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise QueryError("only a single SELECT statement is allowed")

        with self._lock:
            timer = threading.Timer(self.timeout_s, self._con.interrupt)
            timer.start()
            try:
                cursor = self._con.execute(sql, params)
                columns = [d[0] for d in cursor.description]
                records = cursor.fetchmany(self.max_rows + 1)
            except duckdb.InterruptException as e:
                raise QueryError(f"query ran longer than {self.timeout_s}s") from e
            except duckdb.Error as e:
                raise QueryError(str(e)) from e
            finally:
                timer.cancel()

        truncated = len(records) > self.max_rows
        rows = pd.DataFrame.from_records(records[: self.max_rows], columns=columns)
        return QueryResult(rows=rows, truncated=truncated)

    def well_daily(
        self, well: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """All daily rows for one well, ordered by date, optionally within [start, end]."""
        conditions, params = ["well = ?"], [well]
        if start is not None:
            conditions.append("date >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            conditions.append("date <= ?")
            params.append(pd.Timestamp(end))
        sql = f"SELECT * FROM production_daily WHERE {' AND '.join(conditions)} ORDER BY date"
        with self._lock:
            return self._con.execute(sql, params).df()

    def well_issues(self, well: str) -> pd.DataFrame:
        """Every ingest issue logged for one well, ordered by date (no row cap)."""
        sql = "SELECT * FROM production_issues WHERE well = ? ORDER BY date, field"
        with self._lock:
            return self._con.execute(sql, [well]).df()
