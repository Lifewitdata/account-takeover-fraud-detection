"""Parity tests between the DuckDB SQL mirror and the pandas feature implementation.

These assert the agreement, they do not merely execute the SQL: the SQL output is joined
to `ato.features.add_login_features` on attempt_id and compared row by row, per feature.

Run from the repo root:  python -m pytest tests/test_sql_parity.py -q

pyproject.toml configures `testpaths` but no `sys.path` entry for `src`, and this test is
the only file in scope that can fix that, so it prepends <root>/src itself (the same
bootstrap `src/ato/sqlrun.py` uses when executed directly).

`ato.sqlrun.login_frames()` builds the pandas reference frame, which costs a couple of
minutes on 162k login attempts, so the frames are built once per module and reused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ato import features as F  # noqa: E402
from ato import sqlrun as S  # noqa: E402

BIGQUERY_STATEMENTS = 3


@pytest.fixture(scope="module")
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pandas reference, DuckDB mirror) login feature frames."""
    return S.login_frames()


@pytest.fixture(scope="module")
def report(frames) -> pd.DataFrame:
    return S.parity_report(*frames)


def test_sql_row_count_matches_pandas(frames) -> None:
    reference, mirror = frames
    assert len(mirror) == len(reference), "DuckDB and pandas must emit one row per attempt"


def test_attempt_id_set_matches(frames) -> None:
    reference, mirror = frames
    assert set(mirror["attempt_id"]) == set(reference["attempt_id"])


def test_parity_is_computed_by_row_join_not_aggregates(report) -> None:
    # rows_compared is the size of the attempt_id intersection, so a per-row diff would
    # have caught anything an aggregate summary could hide.
    assert int(report["rows_compared"].iloc[0]) > 0
    assert set(report["rows_compared"]) == {int(report["rows_compared"].iloc[0])}
    assert report.attrs["attempt_ids_only_in_reference"] == 0
    assert report.attrs["attempt_ids_only_in_sql"] == 0


def test_every_login_feature_is_reported(report) -> None:
    assert list(report["feature"]) == list(F.LOGIN_FEATURES)


def test_no_feature_mismatches(report) -> None:
    bad = report.loc[report["status"] == "mismatch"]
    assert bad.empty, f"features failing row-level parity:\n{bad.to_string(index=False)}"
    assert int(report["mismatch_count"].sum()) == 0
    assert set(report["status"]) <= {"match", "tolerated"}


def test_integer_flags_and_counts_are_bit_exact(report) -> None:
    exact = report.loc[~report["feature"].isin(S.FLOAT_FEATURES)]
    assert not exact.empty
    assert (exact["max_abs_diff"] == 0.0).all(), (
        "counts and flags are compared with exact equality; a nonzero diff here means the "
        "SQL window/bucket semantics drifted from pandas, not a rounding issue"
    )


def test_derived_floats_agree_to_double_precision(report) -> None:
    floats = report.loc[report["feature"].isin(S.FLOAT_FEATURES)]
    assert not floats.empty
    # Distance is in km and can reach ~18,000, so this is relative-precision noise, not a
    # definition difference; anything above this scale is a real divergence.
    assert (floats["max_abs_diff"] <= 1e-6).all()


def test_bigquery_file_parses_as_bigquery() -> None:
    statements = S.split_statements(S.BIGQUERY_SQL.read_text(encoding="utf-8"))
    assert len(statements) == BIGQUERY_STATEMENTS, (
        "the header comment contains a semicolon ('written carefully; not run live'), so a "
        "naive split on ';' over-counts; split_statements() must skip comment text"
    )
    import sqlglot

    for statement in statements:
        parsed = sqlglot.parse_one(statement, read="bigquery")
        assert parsed is not None


def test_bigquery_file_transpiles_to_duckdb() -> None:
    transpiled = S.transpile_bigquery_to_duckdb(S.BIGQUERY_SQL.read_text(encoding="utf-8"))
    assert "DATE_DIFF" in transpiled  # TIMESTAMP_DIFF -> DuckDB, i.e. it really transpiled
    assert len(S.split_statements(transpiled)) == BIGQUERY_STATEMENTS


def test_views_are_registered_over_the_committed_csvs() -> None:
    conn = S.connect()
    for view, _ in S.CSV_VIEWS:
        rows = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchdf().iloc[0, 0]
        assert int(rows) > 0
    kinds = conn.execute(
        "SELECT column_name, column_type FROM (DESCRIBE login_attempts) "
        "WHERE column_name = 'timestamp'").fetchdf()
    assert kinds.iloc[0, 1] == "TIMESTAMP", "timestamp must not land in DuckDB as VARCHAR"


def test_file_row_number_reproduces_csv_line_order() -> None:
    """The window tie-break relies on this: SQL row ordinal == pandas parse order."""
    conn = S.connect()
    got = conn.execute(
        "SELECT file_row_number, attempt_id FROM login_attempts ORDER BY file_row_number"
    ).fetchdf()
    expected = pd.read_csv(S.C.DATA / "login_attempts.csv", usecols=["attempt_id"])
    assert len(got) == len(expected)
    assert (got["file_row_number"].to_numpy()
            == range(1, len(expected) + 1)).all()
    assert (got["attempt_id"].tolist() == expected["attempt_id"].tolist())


def test_mirror_exposes_the_pandas_feature_columns(frames) -> None:
    _, mirror = frames
    for feature in F.LOGIN_FEATURES:
        assert feature in mirror.columns
