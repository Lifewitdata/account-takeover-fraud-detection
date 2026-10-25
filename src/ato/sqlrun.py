"""Execute the DuckDB port of the feature SQL and prove it against pandas.

Why this module exists: `sql/fraud_features.sql` was authored as BigQuery and never run
against a warehouse, which is a fair criticism of the project. This harness makes the SQL
executable on the committed CSVs and checks it row by row against the authoritative pandas
implementation in `src/ato/features.py`.

The honest claim that this supports is: *ported and dialect-verified with sqlglot, executed
on a DuckDB mirror of the committed CSVs, never run on a live BigQuery warehouse.* Nothing
here talks to GCP - there are no credentials in this checkout.

Run it from the repo root:

    PYTHONPATH=src python -m ato.sqlrun
    python src/ato/sqlrun.py        # zero-config variant, bootstraps sys.path itself
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # package execution: PYTHONPATH=src python -m ato.sqlrun
    from . import config as C
    from . import data as D
    from . import features as F
except ImportError:  # direct script execution: python src/ato/sqlrun.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ato import config as C
    from ato import data as D
    from ato import features as F

import duckdb
import numpy as np
import pandas as pd
import sqlglot

LOGIN_SQL = C.SQL / "duckdb" / "login_features.sql"
BIGQUERY_SQL = C.SQL / "fraud_features.sql"

# Committed CSVs and the view names they are exposed under. login_attempts also
# carries `file_row_number` (the CSV line number) because pandas chains rows with
# identical timestamps in file order, so a faithful SQL mirror needs the same
# deterministic tie-break inside its window. See sql/README.md.
CSV_VIEWS = [
    ("users", "users.csv"),
    ("devices", "devices.csv"),
    ("login_attempts", "login_attempts.csv"),
    ("password_resets", "password_resets.csv"),
    ("transactions", "transactions.csv"),
]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

# Derived numerics are compared with a stated relative tolerance plus a small absolute
# floor; integer/boolean flags and counts are compared exactly.
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-9
FLOAT_FEATURES = {"distance_km", "hours_since_last", "speed_kmh"}

# Features that are floats because they are derived from measurements, not flags.
PARITY_ORDER = list(F.LOGIN_FEATURES)


def _quote(path: Path) -> str:
    """DuckDB reads forward-slash paths on Windows; backslashes escape inside literals."""
    return str(path).replace("\\", "/")


def connect(duckdb_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open DuckDB and register the five committed CSVs as views.

    `duckdb_path=None` gives an in-memory database, which is all the parity check needs;
    pass a file path to poke at the same data from a DuckDB shell afterwards.
    """
    conn = duckdb.connect(str(duckdb_path) if duckdb_path is not None else ":memory:")
    # A CSV row ordinal is needed for the tie-break documented in sql/README.md, and it
    # is only reproducible if the scan is sequential: single thread plus
    # preserve_insertion_order makes row_number() OVER () equal the CSV line number
    # (asserted over all 162,778 rows by tests/test_sql_parity.py).
    conn.execute("SET threads TO 1")
    conn.execute("SET preserve_insertion_order = true")
    for view, filename in CSV_VIEWS:
        # read_csv with auto_detect rather than read_csv_auto because the ordinal column
        # has to be added around the scan.
        sql = (
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT *, CAST(row_number() OVER () AS BIGINT) AS file_row_number "
            f"FROM read_csv("
            f"'{_quote(C.DATA / filename)}', header=true, auto_detect=true, "
            f"timestampformat='{TIMESTAMP_FORMAT}', dateformat='{DATE_FORMAT}')"
        )
        conn.execute(sql)
    conn.execute("CREATE SCHEMA IF NOT EXISTS ato")
    return conn


def split_statements(sql_text: str) -> list[str]:
    """Split a SQL file on ';' at statement level.

    Deliberately small: it ignores semicolons inside single/double quoted literals,
    inside `--` line comments and inside `/* */` block comments (which is how sqlglot
    re-renders the `--` headers of fraud_features.sql, and the header text itself says
    "written carefully; not run live"). It does not try to be a parser; sqlglot already
    exists for that.
    """
    statements: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(sql_text)
    while i < n:
        ch = sql_text[i]
        pair = sql_text[i:i + 2]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                # SQL escapes a quote by doubling it; keep scanning if that happened.
                if i + 1 < n and sql_text[i + 1] == quote:
                    buf.append(sql_text[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if pair == "--":
            end = sql_text.find("\n", i)
            end = n if end == -1 else end
            buf.append(sql_text[i:end])
            i = end
            continue
        if pair == "/*":
            end = sql_text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            buf.append(sql_text[i:end])
            i = end
            continue
        if ch == ";":
            statements.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf)
    if tail.strip():
        statements.append(tail)
    return [s.strip() for s in statements if s.strip()]


def run_file(conn: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """Execute every statement in a file, returning the last SELECT's result."""
    path = Path(path)
    result = pd.DataFrame()
    for statement in split_statements(path.read_text(encoding="utf-8")):
        executed = conn.execute(statement)
        frame = executed.fetchdf() if executed.description is not None else pd.DataFrame()
        if not frame.empty:
            result = frame
    return result


def transpile_bigquery_to_duckdb(sql_text: str) -> str:
    """Re-write BigQuery Standard SQL as DuckDB SQL with sqlglot (parse level, no execution)."""
    transpiled = sqlglot.transpile(sql_text, read="bigquery", write="duckdb")
    return ";\n".join(transpiled) + ";\n"


def _to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="raise").astype("float64").to_numpy()


def _compare_feature(pandas_values: pd.Series, sql_values: pd.Series) -> tuple[float, int]:
    """Row-level absolute difference and mismatch count for one joined feature column."""
    feature = pandas_values.name
    a = _to_float(pandas_values)
    b = _to_float(sql_values)
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    # NULL on one side and a value on the other is always a mismatch; NULL on both
    # sides is agreement (pandas leaves distance_km NaN on the first login of a user).
    nan_disagreement = int((a_nan != b_nan).sum())
    finite = ~(a_nan | b_nan)
    diff = np.abs(a[finite] - b[finite])
    if feature in FLOAT_FEATURES:
        bad = int((diff > FLOAT_ATOL + FLOAT_RTOL * np.abs(b[finite])).sum())
    else:
        bad = int((diff != 0.0).sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    return max_abs, bad + nan_disagreement


def login_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pandas reference, SQL mirror) login feature frames, built from the same rows."""
    reference = F.add_login_features(D.load_silver()["logins"])
    mirror = run_file(connect(), LOGIN_SQL)
    return reference, mirror


def parity_report(reference: pd.DataFrame, mirror: pd.DataFrame) -> pd.DataFrame:
    """Join the two frames on attempt_id and report each of the 15 features.

    Every number here is a per-row difference: the frames are indexed by attempt_id and
    compared row by row, not by comparing aggregate summaries, which would hide per-row
    errors. `rows_compared` is the number of attempts present in both frames.
    """
    ref = reference.set_index("attempt_id", drop=False)
    mir = mirror.set_index("attempt_id", drop=False)
    shared = ref.index.intersection(mir.index)
    rows = []
    for feature in PARITY_ORDER:
        max_abs, mismatches = _compare_feature(
            ref.loc[shared, feature].rename(feature), mir.loc[shared, feature].rename(feature))
        if mismatches > 0:
            status = "mismatch"
        elif max_abs > 0:
            status = "tolerated"
        else:
            status = "match"
        rows.append({"feature": feature, "rows_compared": int(len(shared)),
                     "max_abs_diff": max_abs, "mismatch_count": int(mismatches),
                     "status": status})
    report = pd.DataFrame(rows, columns=["feature", "rows_compared", "max_abs_diff",
                                         "mismatch_count", "status"])
    report.attrs["row_count_reference"] = int(len(ref))
    report.attrs["row_count_sql"] = int(len(mir))
    report.attrs["attempt_ids_only_in_reference"] = int(len(ref.index.difference(mir.index)))
    report.attrs["attempt_ids_only_in_sql"] = int(len(mir.index.difference(ref.index)))
    return report


def verify_login_parity() -> pd.DataFrame:
    """Per-feature row-level parity report for the 15 login features."""
    reference, mirror = login_frames()
    return parity_report(reference, mirror)


def _ascii(text: str) -> str:
    """Console codepage here is cp1252 and the artifacts must stay plain ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


def _first_code_line(sql: str) -> str:
    """First non-blank, non-comment line of a statement, for labeling it."""
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("--", "/*", "*/", "*")):
            return stripped[:78]
    return sql.strip()[:78]


def _describe_statement(sql: str, index: int) -> dict:
    """Parse one statement as BigQuery and transpile it to DuckDB, capturing failure."""
    record = {"index": index, "statement": _first_code_line(sql), "kind": "",
              "parse_ok": False, "transpile_ok": False, "error": ""}
    try:
        parsed = sqlglot.parse_one(sql, read="bigquery")
        record["parse_ok"] = True
        record["kind"] = str(parsed.key).upper()
    except Exception as exc:  # noqa: BLE001 - the failure is the point of this artifact
        record["error"] = f"parse {type(exc).__name__}: {exc}"
        return record
    try:
        sqlglot.transpile(sql, read="bigquery", write="duckdb")
        record["transpile_ok"] = True
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"transpile {type(exc).__name__}: {exc}"
    return record


def run_transpiled_bigquery(
        path: Path = BIGQUERY_SQL) -> tuple[dict, list[dict], pd.DataFrame | None]:
    """Execute the sqlglot-transpiled BigQuery file against a DuckDB catalog shim.

    `project.dataset.login_attempts` is materialised as an attachment so the original
    table references resolve. This runs the *transpiled text on DuckDB*; it is still
    not a BigQuery run. Returns ({index: row_count_or_None}, [failure notes], the
    login_features view re-read as a frame, or None).
    """
    statements = split_statements(path.read_text(encoding="utf-8"))
    conn = connect()
    conn.execute("ATTACH ':memory:' AS project")
    conn.execute("CREATE SCHEMA IF NOT EXISTS project.dataset")
    # Qualified: an unqualified login_attempts here would resolve back to this same
    # view and DuckDB reports infinite recursion.
    conn.execute("CREATE OR REPLACE VIEW project.dataset.login_attempts "
                 "AS SELECT * FROM memory.main.login_attempts")
    results: dict[int, object] = {}
    notes: list[dict] = []
    for index, statement in enumerate(statements, start=1):
        try:
            duck_sql = sqlglot.transpile(statement, read="bigquery", write="duckdb")[0]
        except Exception as exc:  # noqa: BLE001
            notes.append({"index": index, "stage": "transpile",
                          "error": f"{type(exc).__name__}: {exc}"})
            results[index] = None
            continue
        try:
            executed = conn.execute(duck_sql)
        except Exception as exc:  # noqa: BLE001
            notes.append({"index": index, "stage": "execute",
                          "error": f"{type(exc).__name__}: {exc}"})
            results[index] = None
            continue
        results[index] = executed.fetchdf() if executed.description is not None else "no result set"
    try:
        frame = conn.execute("SELECT * FROM project.dataset.login_features").fetchdf()
    except Exception as exc:  # noqa: BLE001
        notes.append({"index": 0, "stage": "read login_features",
                      "error": f"{type(exc).__name__}: {exc}"})
        frame = None
    return results, notes, frame


def _bigquery_reference_section(reference: pd.DataFrame, path: Path = BIGQUERY_SQL) -> list[str]:
    """Quantify how the transpiled BigQuery original differs from the pandas frame."""
    lines = [
        "",
        "APPENDIX - the transpiled text was ALSO executed on DuckDB (never on BigQuery).",
        "The original's table reference was shimmed as project.dataset.login_attempts over",
        "the committed CSV, and outputs were joined to pandas on attempt_id row by row.",
        "This is the measurement behind the difference list in sql/README.md.",
        "",
    ]
    try:
        results, notes, frame = run_transpiled_bigquery(path)
    except Exception as exc:  # noqa: BLE001
        return lines + [f"transpiled run failed outright: {type(exc).__name__}: {exc}"]
    for index, value in results.items():
        rows = "no result set" if isinstance(value, str) else (
            "FAILED" if value is None else f"{len(value)} rows")
        lines.append(f"   statement {index}: {rows}")
    for note in notes:
        lines.append(f"   statement {note['index']} {note['stage']} error: {note['error']}")
    if frame is None or frame.empty:
        lines.append("   login_features view produced no frame; no per-feature comparison.")
        return lines
    lines.append(f"   login_features view re-read from the shim: {len(frame)} rows")
    shared = reference["attempt_id"].tolist()
    ref = reference.set_index("attempt_id").loc[shared]
    mir = frame.set_index("attempt_id").loc[shared]
    lines += ["", "   per-feature divergence of the BigQuery original vs pandas:",
              f"   {'feature':<20}{'rows':>8}{'max_abs_diff':>16}{'rows_differing':>16}",
              "   " + "-" * 60]
    for feature in PARITY_ORDER:
        max_abs, mismatches = _compare_feature(
            ref[feature].rename(feature), mir[feature].rename(feature))
        lines.append(f"   {feature:<20}{len(shared):>8}{max_abs:>16.6g}{mismatches:>16}")
    lines += [
        "",
        "   Interpretation: distance_km differs everywhere because the notebook's",
        "   haversine_km scales the longitude term by cos(lat2) ** 2 while the BigQuery",
        "   file writes the textbook cos(lat1) * cos(lat2). hours_since_last / speed_kmh /",
        "   impossible_travel differ because BigQuery's TIMESTAMP_DIFF(..., MINUTE) floors",
        "   the gap to whole minutes. dow differs on every row: DAYOFWEEK is Sunday = 1..7,",
        "   pandas dayofweek is Monday = 0..6 (sqlglot renders it as DuckDB dayofweek,",
        "   Sunday = 0..6, a third scale). new_device / new_country differ on 6 rows only,",
        "   which is the unqualified LAG ORDER BY timestamp over 36 tied (user_id,",
        "   timestamp) pairs - the original has no deterministic tie-break, so those flags",
        "   are not reproducible. u_att / u_failed / i_att / i_failed / i_users differ",
        "   because sqlglot renders BigQuery's 600 * DIV(UNIX_SECONDS(ts), 600) as",
        "   600 * CAST(EPOCH(ts) AS BIGINT) // 600, and in DuckDB * and // bind left to",
        "   right, so the multiplication applies before the floor division and the",
        "   10-minute bucket collapses to per-second. Transpiled text needs review, not",
        "   blind execution; sql/duckdb/login_features.sql is the version that is proven.",
    ]
    return lines


def bigquery_dialect_report(path: Path = BIGQUERY_SQL,
                           reference: pd.DataFrame | None = None) -> str:
    """Text artifact backing the 'dialect-verified' claim, including any parse failure."""
    statements = split_statements(path.read_text(encoding="utf-8"))
    lines = [
        "BigQuery dialect check for sql/fraud_features.sql",
        "generated by src/ato/sqlrun.py",
        "",
        "WHAT THIS IS: sqlglot parsed every statement as read='bigquery' and transpiled it",
        "to write='duckdb'. That is a dialect/syntax check, plus (appendix) an execution of",
        "the transpiled text against a DuckDB mirror of the committed CSVs.",
        "WHAT THIS IS NOT: sql/fraud_features.sql was NEVER run on a live BigQuery",
        "warehouse - this checkout has no GCP credentials and nothing here contacts one.",
        "The executable proof of the feature logic is the DuckDB mirror",
        "sql/duckdb/login_features.sql, run against the CSVs; its row-level agreement with",
        "pandas is outputs/sql/sql_parity.csv.",
        "",
        f"source file      : {path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()}",
        f"sqlglot version  : {sqlglot.__version__}",
        f"duckdb version   : {duckdb.__version__}",
        f"statements found : {len(statements)}",
        "",
    ]
    records = [_describe_statement(sql, i) for i, sql in enumerate(statements, start=1)]
    for rec in records:
        lines += [
            f"-- statement {rec['index']}  ({rec['kind'] or 'unparsed'})",
            f"   head       : {rec['statement']}",
            f"   parse ok   : {rec['parse_ok']}",
            f"   transpile  : {rec['transpile_ok']}",
            f"   error      : {rec['error'] or 'none'}",
        ]
    parsed = sum(1 for r in records if r["parse_ok"])
    transpiled = sum(1 for r in records if r["transpile_ok"])
    lines += [
        "",
        f"summary: {parsed}/{len(records)} statements parse as BigQuery Standard SQL, "
        f"{transpiled}/{len(records)} transpile to DuckDB.",
        "",
        "NOTE: a naive split on ';' finds more than 3 statements here - the header comment",
        "says 'written carefully; not run live', and split_statements() therefore skips",
        "comment text before splitting.",
        "",
        "Transpiled DuckDB text:",
        "",
    ]
    try:
        lines.append(transpile_bigquery_to_duckdb(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"transpile of the whole file failed: {type(exc).__name__}: {exc}")
    if reference is not None:
        lines += _bigquery_reference_section(reference, path)
    return _ascii("\n".join(lines) + "\n")


def main() -> None:
    C.ensure_dirs()
    print("[sqlrun] building the pandas reference frame (about a minute) ...")
    reference, mirror = login_frames()
    print(f"[sqlrun] sql rows={len(mirror)} cols={len(mirror.columns)} "
          f"pandas rows={len(reference)}")
    C.write_csv(mirror, C.SQL_DIR, "login_features_duckdb.csv")

    report = parity_report(reference, mirror)
    C.write_csv(report, C.SQL_DIR, "sql_parity.csv")
    print(report.to_string(index=False))
    print(f"[sqlrun] attempts only in pandas={report.attrs['attempt_ids_only_in_reference']} "
          f"only in sql={report.attrs['attempt_ids_only_in_sql']} "
          f"total mismatched rows={int(report['mismatch_count'].sum())}")

    text = bigquery_dialect_report(reference=reference)
    C.write_text(text, C.SQL_DIR, "bigquery_dialect_check.txt")
    print("[sqlrun] wrote outputs/sql/login_features_duckdb.csv, "
          "outputs/sql/sql_parity.csv, outputs/sql/bigquery_dialect_check.txt")


if __name__ == "__main__":
    main()
