"""Read-only access to the processed field data: production and report facts.

The Parquet files are loaded into an in-memory DuckDB database; then file access is
switched off and the configuration locked. Agent-written SQL therefore sees only these
tables, can run only a single SELECT, and never sees rows after the store's `as_of` date
(nor reports issued after it), which is what keeps backtests honest.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from wellpilot.ingest.production import DAILY_FILE, ISSUES_FILE
from wellpilot.ingest.report_facts import FACTS_FILE, SUMMARIES_FILE
from wellpilot.ingest.reports import PAGES_FILE, REPORTS_BY_ID

TABLES = {"production_daily": DAILY_FILE, "production_issues": ISSUES_FILE}
# Present once the report ingest has run.
REPORT_TABLES = {
    "report_pages": PAGES_FILE,
    "report_facts": FACTS_FILE,
    "report_summaries": SUMMARIES_FILE,
}

logger = logging.getLogger(__name__)

# Agent-written SQL comes from a public question. DuckDB's memory limit does not bound a
# single huge string (repeat('x', 5e7) over 40 rows took 2.85 GB), a recursive CTE can
# double a string every step, and the settings expose local paths. These are refused.
DENIED_FUNCTIONS = {
    "repeat", "rpad", "lpad", "printf", "format", "bar",
    "current_setting", "duckdb_settings", "duckdb_secrets", "duckdb_extensions",
}  # fmt: skip
MAX_TEXT = 2000  # characters of one text value a query may return to the agent


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
        self.processed_dir = processed_dir  # report page images live here
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._con = duckdb.connect(":memory:")
        self.tables: set[str] = set()
        for table, file in TABLES.items():
            frame = pd.read_parquet(processed_dir / file)
            if as_of is not None:
                frame = frame[frame["date"] <= pd.Timestamp(as_of)]
            self._load(table, frame)
        for table, file in REPORT_TABLES.items():
            if not (processed_dir / file).exists():
                continue
            frame = pd.read_parquet(processed_dir / file)
            unknown = set(frame["doc_id"]) - set(REPORTS_BY_ID)
            if unknown:  # a document dropped from the registry: no date to cut it by
                logger.warning("ignoring reports not in REPORTS: %s", sorted(unknown))
                frame = frame[~frame["doc_id"].isin(unknown)]
            if as_of is not None:
                issued = frame["doc_id"].map(lambda d: REPORTS_BY_ID[d].doc_date)
                frame = frame[issued <= as_of.isoformat()]
            self._load(table, frame)
        self._con.execute("SET temp_directory = ''")  # nothing spills to disk
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET memory_limit = '256MB'")
        self._con.execute("SET threads = 2")
        self._con.execute("SET lock_configuration = true")

    def _load(self, table: str, frame: pd.DataFrame) -> None:
        self._con.register("_frame", frame)
        self._con.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
        self._con.unregister("_frame")
        self.tables.add(table)

    def query(self, sql: str, params: list | None = None) -> QueryResult:
        """Run one read-only SELECT, capped at max_rows rows and timeout_s seconds."""
        # Everything touching the connection, parsing included, holds the lock: DuckDB
        # connections are not safe to use from several threads at once.
        with self._lock:
            self._single_select(sql)
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

    def _single_select(self, sql: str) -> None:
        """Raise unless sql is exactly one SELECT (checked by DuckDB's own parser). Call it
        holding the lock."""
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as e:
            raise QueryError(f"SQL does not parse: {e}") from e
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise QueryError("only a single SELECT statement is allowed")

    def query_untrusted(self, sql: str) -> QueryResult:
        """Run SQL the agent wrote: a single SELECT without string-bloating functions,
        recursion or settings, whose text columns come back at most MAX_TEXT characters
        long (cut inside DuckDB, so a huge value never reaches Python)."""
        with self._lock:
            self._single_select(sql)
            tree = json.loads(
                self._con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
            )
            if not tree.get("error"):
                text = json.dumps(tree)
                used = set(re.findall(r'"function_name": "(\w+)"', text))
                denied = sorted(f for f in used if f in DENIED_FUNCTIONS or f.startswith("pragma_"))
                if denied:
                    raise QueryError(f"function(s) not allowed here: {', '.join(denied)}")
                if "RECURSIVE_CTE_NODE" in text:
                    raise QueryError("recursive queries are not allowed")
            try:
                described = self._con.execute(f"DESCRIBE {sql}").fetchall()
            except duckdb.Error as e:
                raise QueryError(str(e)) from e
        names = [row[0] for row in described]
        if len(set(names)) != len(names):
            raise QueryError("give every result column a distinct name (use AS)")
        quoted = [
            (name.replace('"', '""'), row[1]) for name, row in zip(names, described, strict=True)
        ]
        columns = ", ".join(
            f'left("{n}", {MAX_TEXT}) AS "{n}"' if kind == "VARCHAR" else f'"{n}"'
            for n, kind in quoted
        )
        return self.query(f"SELECT {columns} FROM ({sql}) AS _q")

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
