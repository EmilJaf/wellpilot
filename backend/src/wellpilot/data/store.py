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

from wellpilot.data.sql_worker import IsolatedSql
from wellpilot.ingest.operations import OPERATIONS_FILE, STEPS_FILE
from wellpilot.ingest.production import DAILY_FILE, ISSUES_FILE
from wellpilot.ingest.production_logs import LOGS_FILE
from wellpilot.ingest.report_facts import FACTS_FILE, SUMMARIES_FILE
from wellpilot.ingest.reports import PAGES_FILE, REPORTS_BY_ID

TABLES = {"production_daily": DAILY_FILE, "production_issues": ISSUES_FILE}
# Present once the report ingest has run.
REPORT_TABLES = {
    "report_pages": PAGES_FILE,
    "report_facts": FACTS_FILE,
    "report_summaries": SUMMARIES_FILE,
}
# Present once the daily-report and production-log ingests have run: well work by day and
# the logs run in the wells, cut at as_of like production (known from the day of the work).
UNTRUSTED_ROWS = 201  # run_sql shows 200 rows and says when there were more
OPERATION_TABLES = {
    "well_operations": OPERATIONS_FILE,
    "well_operation_steps": STEPS_FILE,
    "production_logs": LOGS_FILE,
}

logger = logging.getLogger(__name__)

# Agent-written SQL comes from a public question. DuckDB's memory limit does not bound a
# single huge string (repeat('x', 5e7) over 40 rows took 2.85 GB), a recursive CTE can
# double a string every step, and the settings expose local paths. These are refused.
DENIED_FUNCTIONS = {
    "repeat", "rpad", "lpad", "printf", "format", "bar",
    "current_setting", "duckdb_settings", "duckdb_secrets", "duckdb_extensions",
    "query", "query_table", "json_execute_serialized_sql", "getvariable",
    "list_resize", "array_resize",
}  # fmt: skip
# As scalars these build a list in one call outside the memory limit (len(range(2e7)) took
# 2.4 GB); in FROM they only make rows. The public server also runs agent SQL in a capped
# worker process (sql_worker.py), which this list does not replace.
LIST_BUILDERS = {"range", "generate_series"}
# Table functions are allowed by name: query('...') runs a string as SQL past every check
# here, and the rest read files, settings or the catalog. These three only make rows.
TABLE_FUNCTIONS = {"range", "generate_series", "unnest"}
MAX_TEXT = 2000  # characters of one text value a query may return to the agent


def _table_functions(node) -> list[str]:
    """Names of the table functions in DuckDB's serialized syntax tree."""
    found = []
    if isinstance(node, dict):
        if node.get("type") == "TABLE_FUNCTION":
            found.append(str((node.get("function") or {}).get("function_name", "?")))
        for value in node.values():
            found += _table_functions(value)
    elif isinstance(node, list):
        for value in node:
            found += _table_functions(value)
    return found


def _scalar_functions(node, in_table: bool = False) -> list[str]:
    """Names of the functions called as scalars (not as a FROM-clause table function)."""
    found = []
    if isinstance(node, dict):
        if node.get("class") == "FUNCTION" and not in_table:
            found.append(str(node.get("function_name", "?")))
        for key, value in node.items():
            table = node.get("type") == "TABLE_FUNCTION" and key == "function"
            found += _scalar_functions(value, table)
    elif isinstance(node, list):
        for value in node:
            found += _scalar_functions(value)
    return found


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
        isolated: bool = False,
    ) -> None:
        self.as_of = as_of
        self.processed_dir = processed_dir  # report page images live here
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        # agent SQL in a capped worker process (the public server); see sql_worker.py
        self.isolated = IsolatedSql(processed_dir, as_of, max_rows, timeout_s) if isolated else None
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
        for table, file in OPERATION_TABLES.items():
            if not (processed_dir / file).exists():
                continue
            frame = pd.read_parquet(processed_dir / file)
            if as_of is not None:
                cut = pd.Timestamp(as_of)
                if "report_date" in frame:  # a day's work is reported the next morning
                    frame = frame[frame["report_date"] <= cut]
                elif table == "well_operation_steps":  # lines of those same reports
                    frame = frame[frame["date"] < cut]
                else:
                    frame = frame[frame["date"] <= cut]
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

    def query(
        self, sql: str, params: list | None = None, max_rows: int | None = None
    ) -> QueryResult:
        """Run one read-only SELECT, capped at max_rows rows and timeout_s seconds."""
        limit = min(self.max_rows, max_rows) if max_rows else self.max_rows
        # Everything touching the connection, parsing included, holds the lock: DuckDB
        # connections are not safe to use from several threads at once.
        with self._lock:
            self._single_select(sql)
            timer = threading.Timer(self.timeout_s, self._con.interrupt)
            timer.start()
            try:
                cursor = self._con.execute(sql, params)
                columns = [d[0] for d in cursor.description]
                records = cursor.fetchmany(limit + 1)
            except duckdb.InterruptException as e:
                raise QueryError(f"query ran longer than {self.timeout_s}s") from e
            except duckdb.Error as e:
                raise QueryError(str(e)) from e
            finally:
                timer.cancel()

        truncated = len(records) > limit
        rows = pd.DataFrame.from_records(records[:limit], columns=columns)
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
        if self.isolated is not None:
            return self.isolated.query(sql)
        with self._lock:
            self._single_select(sql)
            tree = json.loads(
                self._con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
            )
            if tree.get("error"):  # unchecked SQL never runs
                raise QueryError(f"SQL could not be checked: {tree.get('error_message', '')}")
            text = json.dumps(tree)
            used = set(re.findall(r'"function_name": "(\w+)"', text))
            denied = sorted(f for f in used if f in DENIED_FUNCTIONS or f.startswith("pragma_"))
            denied += sorted(set(_table_functions(tree)) - TABLE_FUNCTIONS - set(denied))
            denied += sorted(set(_scalar_functions(tree)) & LIST_BUILDERS)
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
        # The agent sees UNTRUSTED_ROWS rows: fetching thousands of wide ones only to drop
        # them cost hundreds of MB in the worker's reply (audit, Oct 31).
        return self.query(f"SELECT {columns} FROM ({sql}) AS _q", max_rows=UNTRUSTED_ROWS)

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
