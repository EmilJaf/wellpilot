"""Runs inside the sandbox; imports nothing from wellpilot.

    python sandbox_runner.py prepare <as_of|none>   # once per run
    python sandbox_runner.py run <code>              # each run_python call

prepare copies the field's tables from the image (VOLVE_DIR) to DATA_DIR, cut at the run's
as-of date, and deletes the full copy, so code written for a question about 2010 cannot
read 2011. run executes the agent's code with pandas, numpy, scipy and two helpers over
those tables, sql() and daily(), and prints the value it left in `result` as one JSON line
after a marker, with what it printed and how many data rows it read.
"""

import contextlib
import io
import json
import linecache
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

VOLVE_DIR = Path(os.environ.get("VOLVE_DIR", "/volve"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TABLES = ("production_daily", "production_issues", "production_logs")
MARKER = "__wellpilot_result__"
PRINTED_CHARS = 2000
SOURCE = "<agent code>"


def prepare(as_of: str) -> None:
    import duckdb

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for table in TABLES:
        source = VOLVE_DIR / f"{table}.parquet"
        if not source.exists():
            continue
        cut = "" if as_of == "none" else f" WHERE date <= DATE '{as_of}'"
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{source}'){cut}) "
            f"TO '{DATA_DIR / f'{table}.parquet'}' (FORMAT parquet)"
        )
    shutil.rmtree(VOLVE_DIR, ignore_errors=True)


def jsonable(value):
    """The result as JSON: tables column-wise, numpy and pandas values as Python."""
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        frame = value.reset_index() if not isinstance(value.index, pd.RangeIndex) else value
        return {
            "columns": [str(c) for c in frame.columns],
            "rows": jsonable(frame.astype(object).values.tolist()),
        }
    if isinstance(value, pd.Series):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | np.ndarray):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, bool | int | str):
        return value
    if value is pd.NaT:
        return None
    return str(value)


def run(code: str, nonce: str = "") -> None:
    import duckdb
    import numpy as np
    import pandas as pd

    con = duckdb.connect()
    for table in TABLES:
        path = DATA_DIR / f"{table}.parquet"
        if path.exists():
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    read = {"rows": 0}

    def sql(query: str, params: list | None = None) -> pd.DataFrame:
        """One DuckDB query over production_daily, production_issues, production_logs."""
        frame = con.execute(query, params or []).df()
        read["rows"] += len(frame)
        return frame

    def daily(well: str) -> pd.DataFrame:
        """One well's daily production, as production_daily holds it."""
        return sql("SELECT * FROM production_daily WHERE well = ? ORDER BY date", [well])

    namespace = {"pd": pd, "np": np, "sql": sql, "daily": daily, "__name__": "__agent__"}
    # The result line carries a nonce from the server: a line the code prints that looks
    # like it (or one printed before os._exit) is not taken for the result. The code is not
    # handed the nonce; the result stays an inference all the same.
    del sys.argv[1:]
    printed = io.StringIO()
    # Tracebacks quote the agent's own lines, and only its frames: what it needs to fix.
    linecache.cache[SOURCE] = (len(code), None, code.splitlines(True), SOURCE)
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(code, SOURCE, "exec"), namespace)
    except Exception as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == SOURCE]
        sys.stderr.write(
            "Traceback (your code):\n"
            + "".join(traceback.format_list(frames))
            + "".join(traceback.format_exception_only(e))
        )
        sys.exit(1)
    out = {
        "result": jsonable(namespace.get("result")),
        "printed": printed.getvalue()[-PRINTED_CHARS:],
        "rows_read": read["rows"],
    }
    print(MARKER + nonce + json.dumps(out))


if __name__ == "__main__":
    mode, argument = sys.argv[1], sys.argv[2]
    prepare(argument) if mode == "prepare" else run(argument, *sys.argv[3:4])
