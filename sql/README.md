# sql/ — the feature SQL, executed and measured

Two files, two different claims. Do not conflate them.

| file | dialect | what was actually done to it |
| --- | --- | --- |
| `fraud_features.sql` | BigQuery Standard SQL | parsed and transpiled with sqlglot (dialect check). **Never executed on a live warehouse** — this checkout has no GCP credentials. Its transpiled text was additionally executed on a DuckDB shim (see "Reference run" below). |
| `duckdb/login_features.sql` | DuckDB SQL | **executed** against the committed CSVs and compared row by row to the pandas implementation in `src/ato/features.py`. |

The honest sentence to use: *the feature SQL is ported and dialect-verified, executed on a
DuckDB mirror of the committed CSVs, and proven to agree with the pandas pipeline on all 15
login features at row level; it has never been run on BigQuery.*

## Commands

```bash
# from the repo root, Python 3.11+, deps from requirements.txt + requirements-dev.txt
PYTHONPATH=src python -m ato.sqlrun          # ~5 min, dominated by the pandas reference build
python src/ato/sqlrun.py                     # same thing, bootstraps sys.path itself
python -m pytest tests/test_sql_parity.py -q # the parity assertions (~5 min, same reference build)
```

`ato` is not installed in site-packages and `pyproject.toml` sets no pytest `pythonpath`, so
plain `python -m ato.sqlrun` from the repo root needs `PYTHONPATH=src`; `src/ato/sqlrun.py`
and `tests/test_sql_parity.py` each prepend `<root>/src` instead, so those two work with no
environment setup at all.

Writes:

- `outputs/sql/login_features_duckdb.csv` — the executed SQL output (162,778 rows; the 15
  features plus keys, and a `distance_km_textbook` diagnostic column, see difference 1).
- `outputs/sql/sql_parity.csv` — the row-level parity report below.
- `outputs/sql/bigquery_dialect_check.txt` — sqlglot parse/transpile result for every
  statement of `fraud_features.sql` plus the reference-run appendix. Any parse failure lands
  in this file as text instead of being swallowed.

## Parity result (`outputs/sql/sql_parity.csv`)

Frames are joined on `attempt_id` and compared per row — not by comparing aggregates — over
all 162,778 attempts (0 attempts present on only one side). Integer counts and flags are
compared with exact equality; `distance_km`, `hours_since_last` and `speed_kmh` with
`FLOAT_RTOL = 1e-6` relative + `FLOAT_ATOL = 1e-9` absolute (module constants in
`src/ato/sqlrun.py`, not inline numbers).

| feature | rows_compared | max_abs_diff | mismatch_count | status |
| --- | --- | --- | --- | --- |
| distance_km | 162778 | 2.728484e-12 | 0 | tolerated |
| hours_since_last | 162778 | 0.0 | 0 | match |
| speed_kmh | 162778 | 2.842171e-14 | 0 | tolerated |
| impossible_travel | 162778 | 0.0 | 0 | match |
| new_device | 162778 | 0.0 | 0 | match |
| new_country | 162778 | 0.0 | 0 | match |
| u_att | 162778 | 0.0 | 0 | match |
| u_failed | 162778 | 0.0 | 0 | match |
| i_att | 162778 | 0.0 | 0 | match |
| i_failed | 162778 | 0.0 | 0 | match |
| i_users | 162778 | 0.0 | 0 | match |
| device_n_users | 162778 | 0.0 | 0 | match |
| ip_n_users | 162778 | 0.0 | 0 | match |
| hour | 162778 | 0.0 | 0 | match |
| dow | 162778 | 0.0 | 0 | match |

`tolerated` is double-precision noise at the 1e-12 level on values up to ~18,000 km, i.e.
relative error ~1e-16; nothing had to be rounded away and no feature needed an xfail.

## What the DuckDB file reads

`ato.sqlrun.connect()` registers the five committed CSVs as views (`users`, `devices`,
`login_attempts`, `password_resets`, `transactions`) with explicit
`timestampformat='%Y-%m-%d %H:%M:%S'` / `dateformat='%Y-%m-%d'` so `timestamp` lands as
`TIMESTAMP` rather than `VARCHAR`. `login_features.sql` reads `login_attempts` and `users`.

Two mechanical things it depends on:

- **`file_row_number`** — `connect()` adds a CSV-line ordinal to each view (`row_number()
  OVER ()` with `SET threads TO 1` + `preserve_insertion_order`, which was verified to equal
  the pandas parse order for all 162,778 rows; `test_file_row_number_reproduces_csv_line_order`
  asserts it on every run). pandas `sort_values(["user_id", "timestamp"])` keeps rows with an
  identical timestamp in file order, and `groupby.shift(1)` chains them in that order, so
  `LAG(...) OVER (PARTITION BY user_id ORDER BY timestamp, file_row_number)` is required to
  match. Ordering by `attempt_id` instead moved `hours_since_last` on 32 rows.
- **the `users` join** — it mirrors the silver foreign-key filter in `src/ato/data.py`
  (attempts whose `user_id` is not a known account are quarantined before features are built).
  On the committed CSVs `dq_audit` drops zero rows, so it is a no-op guard; it matters if the
  CSVs ever stop being clean, because the velocity and sharing counts would otherwise be
  computed over rows the pipeline never sees.

## pandas vs `fraud_features.sql`: the differences, and which side each one followed

pandas (`src/ato/features.py`, i.e. the notebook) is authoritative, because the pipeline, the
rules and the model all consume it. So the DuckDB port follows pandas in every case, including
case 1 where pandas is the one that is mathematically wrong.

1. **`distance_km` — the pandas haversine is wrong and the port mirrors it.**
   `features.py:54` computes
   `a = sin(dlat/2)^2 + cos(lat1) * cos(lat2)**2 * sin(dlon/2)^2` — note the `** 2` on
   `cos(lat2)`, which is not the great-circle formula (correct is `cos(lat1)*cos(lat2)`).
   `fraud_features.sql:42` writes the correct product. Measured on the reference run: the two
   definitions disagree on ~149k of 162,778 rows, up to 11,124 km, because the pandas version
   under-reports the longitude term wherever `cos(lat) < 1` (roughly 40% short at 50 degrees
   latitude, which is where most of this data sits). Followed: **pandas**, so parity holds; the
   correct value is emitted next to it as `distance_km_textbook`. Consequence to flag to a
   reviewer: `speed_kmh` and the `impossible_travel` thresholds (500 km / 800 km/h) are tuned
   against the shrunken distance, so `R04_impossible_travel` fires less often than the rule
   text implies. Fixing the formula means re-fitting the rule, which is out of scope here.
2. **`hours_since_last` — BigQuery floors the gap to whole minutes.** `fraud_features.sql:45`
   uses `TIMESTAMP_DIFF(timestamp, prev_ts, MINUTE) / 60.0`; `features.py:69` uses
   `total_seconds() / 3600`. Up to 60 s of error per gap, and gaps under one minute become
   exactly 0 h, which `NULLIF(..., 0)` then turns into `speed_kmh = 0`. Reference run: differs
   on ~152k rows, and it changes `impossible_travel` on 83 of them. Followed: **pandas**
   (exact seconds, `epoch_ns` difference / 3.6e12).
3. **`dow` — three different scales.** `fraud_features.sql:60` uses `EXTRACT(DAYOFWEEK ...)`
   (Sunday = 1 .. Saturday = 7, as its own comment says); `features.py:90` uses
   `dt.dayofweek` (Monday = 0 .. Sunday = 6); sqlglot's DuckDB rendering gives Sunday = 0 ..
   Saturday = 6, a third scale. The reference run differs on all 162,778 rows. Followed:
   **pandas** (`CAST(DATE_PART('isodow', timestamp) AS INTEGER) - 1`). `hour` (line 59) is the
   same on all three readings — 0 differing rows — because the CSV timestamps are naive.
4. **The BigQuery window has no tie-break, so its output is not reproducible.**
   `fraud_features.sql:34` orders `LAG(...) OVER (PARTITION BY user_id ORDER BY timestamp)`
   only. The CSVs contain 36 `(user_id, timestamp)` pairs with identical second-level
   timestamps, so which of the two rows is "previous" is up to the engine. In the reference run
   that flipped `new_device` on 6 rows and `new_country` on 6 rows — the count itself moves
   between runs, which is the point. Followed: **pandas** (deterministic file order, via
   `file_row_number`).
5. **`device_n_users` / `ip_n_users` are whole-period counts, by design, in all three.**
   `features.py:86-87` and `fraud_features.sql:80-89` both count distinct users per device/IP
   over the entire frame, so they are **not past-only**, whatever the "15 past-only features"
   header at `fraud_features.sql:15` and the docstring of `add_login_features` claim. Kept as
   is (they are structural facts about the device/IP, and `R02_shared_device_5u` and the
   incident graph both rely on that reading) and called out here rather than silently converted
   into a running count.
6. **The velocity features are whole-bucket, also not strictly past-only.**
   `u_att / u_failed / i_att / i_failed / i_users` (`features.py:76-84`,
   `fraud_features.sql:63-79`) count every attempt in the same 10-minute bucket, including the
   current row and rows that arrive later in the same bucket. The bucket definition itself
   agrees: `dt.floor("10min")` and `600 * DIV(UNIX_SECONDS(ts), 600)` are the same 600-second
   grid, and the port keeps the integer epoch grid (`epoch_ns // 6e11`) so no time-zone or
   timestamp rounding can creep in. Followed: **both** (they already agree); the
   "past-only" wording in the header is what is wrong.
7. **Source relation.** The BigQuery file reads the raw `login_attempts` table
   (`fraud_features.sql:23`); the pipeline builds features on silver (`load_silver`, i.e.
   bronze minus the DQ quarantine). Addressed by the `users` join above; zero rows differ on
   the committed data.
8. **Statement splitting needs comment awareness** — `fraud_features.sql:3` contains
   "written carefully; not run live", a `;` inside a `--` comment. A naive `split(";")`
   over-counts statements; `split_statements()` skips `--` and `/* */` comment text and quoted
   literals, and `test_bigquery_file_parses_as_bigquery` pins the count at 3.

## Reference run (`outputs/sql/bigquery_dialect_check.txt`, appendix)

The transpiled text was executed on DuckDB against a `project.dataset.login_attempts` shim so
that the original file's own references resolve: statement 1 creates the view (162,778 rows),
statement 2 (`account_risk_rollup`) returns 5,100 rows, statement 3 (`ip_watchlist`) returns
**0 rows — and that is a transpilation defect, not a data finding**. sqlglot renders BigQuery's
`600 * DIV(UNIX_SECONDS(ts), 600)` as `600 * CAST(EPOCH(ts) AS BIGINT) // 600`, and in DuckDB
`*` and `//` bind left to right, so the multiplication applies *before* the floor division and
the 10-minute bucket collapses to a per-second key. With the bucket written correctly
(`600 * (epoch // 600)`) the watchlist query matches 48 windows (max 360 attempts / 60 distinct
users in one window), which is the shape `R01_ip_velocity_5u10m` describes. Lesson recorded
honestly: transpiled SQL is a starting point, not proof; the version that is proven is
`duckdb/login_features.sql`.
