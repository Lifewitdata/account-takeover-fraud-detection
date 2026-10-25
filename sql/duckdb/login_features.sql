-- ============================================================================
-- login_features.sql - DuckDB SQL, executable mirror of
-- src/ato/features.py::add_login_features (the authoritative implementation).
--
--   run with : python -m ato.sqlrun          (see sql/README.md)
--   reads    : views login_attempts and users, registered by ato.sqlrun.connect
--   returns  : one row per login attempt, the 15 LOGIN_FEATURES columns
--
-- This is NOT a rewrite of sql/fraud_features.sql. That BigQuery file stays as
-- authored (parsed and dialect-checked, never executed against a warehouse).
-- Where the two disagree, the pandas/notebook definition wins here because the
-- parity target is the pandas frame; each divergence is listed in
-- sql/README.md:
--   * hours_since_last is exact seconds / 3600, not BigQuery's MINUTE-truncated diff
--   * dow follows pandas dayofweek (Monday = 0 ... Sunday = 6), not DAYOFWEEK (Sunday = 1)
--   * distance_km keeps the pandas cos(lat2) ** 2 form (see movement, item 1)
--   * device_n_users / ip_n_users stay WHOLE-PERIOD counts, on purpose (see below)
--   * u_att / u_failed / i_att / i_failed / i_users are whole-10-minute-bucket
--     counts in pandas too, so the bucket including the current row is faithful,
--     not a leak introduced by this file
--
-- Past-only semantics: every prev_* value is a LAG inside PARTITION BY user_id
-- ordered by event time, so a row only ever sees earlier rows of its own
-- account; the first row of a user gets NULL, exactly like pandas shift(1).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- login_features: one row per login attempt, the 15 features of LOGIN_FEATURES.
-- ----------------------------------------------------------------------------
WITH base AS (
  SELECT
    la.attempt_id,
    la.user_id,
    la.timestamp,
    la.ip_address,
    la.device_id,
    la.ip_country,
    la.ip_lat,
    la.ip_lon,
    la.login_result,
    la.file_row_number,
    -- 10-minute bucket as a whole-number epoch key, exactly pandas'
    -- .dt.floor("10min") (floor to a multiple of 600 s). Keeping it an integer
    -- removes any timestamp/time-zone rounding from the group-and-join below.
    CAST(epoch_ns(la.timestamp) // 600000000000 AS BIGINT) * 600000000000 AS t10
  -- The join mirrors the silver foreign-key filter in src/ato/data.py: attempts
  -- whose user_id is not a known account are quarantined before features are
  -- built. On the committed CSVs dq_audit reports zero violations, so this is a
  -- no-op guard, not a row-reducing step.
  FROM login_attempts AS la
  INNER JOIN users AS us USING (user_id)
),
with_prev AS (
  SELECT
    base.*,
    LAG(timestamp)   OVER w AS prev_ts,
    LAG(ip_lat)      OVER w AS prev_lat,
    LAG(ip_lon)      OVER w AS prev_lon,
    LAG(device_id)   OVER w AS prev_device,
    LAG(ip_country)  OVER w AS prev_country
  FROM base
  -- Timestamp ties need an explicit, deterministic tie-break because pandas
  -- resolves them by input order (read_csv file order, then a stable multi-key
  -- sort), and groupby.shift(1) therefore chains tied rows in file order. The
  -- CSVs hold 36 tied (user_id, timestamp) pairs; ordering by file_row_number -
  -- the CSV line number that ato.sqlrun.connect exposes on the view - reproduces
  -- that chaining exactly, which attempt_id ordering does not (it swapped the
  -- pair members and moved hours_since_last on 32 rows).
  WINDOW w AS (PARTITION BY user_id ORDER BY timestamp, file_row_number)
),
movement AS (
  SELECT
    with_prev.*,
    -- distance_km, same 6371 km mean radius and arcsin(sqrt(a)) shape as
    -- features.haversine_km, with point 1 = the previous login of this account
    -- and point 2 = this one.
    --
    -- FAITHFUL MIRROR, INCLUDING A KNOWN DEFECT: features.py line 54 reads
    --     np.cos(lat1) * np.cos(lat2) ** 2 * np.sin(dlon / 2) ** 2
    -- so the longitude term is scaled by cos(lat2) SQUARED rather than by
    -- cos(lat1) * cos(lat2). That is not the great-circle distance: it shrinks the
    -- longitudinal component by an extra factor cos(lat) (~0.6x at 50 degrees, which
    -- is where most of this data sits), so any hop with a longitude change comes out
    -- short. The BigQuery original writes the textbook product instead. pandas is the
    -- parity authority, so distance_km below keeps the ** 2 and matches the
    -- pipeline row for row; distance_km_textbook carries the corrected value as
    -- a diagnostic and is deliberately excluded from the parity comparison.
    6371 * 2 * ASIN(SQRT(
      POW(SIN((RADIANS(ip_lat) - RADIANS(prev_lat)) / 2), 2) +
      COS(RADIANS(prev_lat)) * POW(COS(RADIANS(ip_lat)), 2) *
      POW(SIN((RADIANS(ip_lon) - RADIANS(prev_lon)) / 2), 2)
    )) AS distance_km,
    6371 * 2 * ASIN(SQRT(
      POW(SIN((RADIANS(ip_lat) - RADIANS(prev_lat)) / 2), 2) +
      COS(RADIANS(prev_lat)) * COS(RADIANS(ip_lat)) *
      POW(SIN((RADIANS(ip_lon) - RADIANS(prev_lon)) / 2), 2)
    )) AS distance_km_textbook,
    -- exact elapsed hours between consecutive logins of the account
    -- (pandas: (timestamp - prev_ts).dt.total_seconds() / 3600)
    (epoch_ns(timestamp) - epoch_ns(prev_ts)) / 3600000000000.0 AS hours_since_last
  FROM with_prev
),
kinematics AS (
  SELECT
    movement.*,
    -- pandas: (distance_km / hours_since_last.replace(0, nan)).fillna(0)
    -- A NULL hours_since_last (first row of an account) and a NULL distance both
    -- collapse to 0 the same way fillna(0) does.
    COALESCE(distance_km / NULLIF(hours_since_last, 0), 0) AS speed_kmh
  FROM movement
),
flags AS (
  SELECT
    kinematics.*,
    -- pandas compares a possibly-NaN distance; NaN > 500 is False there, and a
    -- NULL comparison falls through this CASE to 0 here.
    CASE WHEN distance_km > 500 AND speed_kmh > 800 THEN 1 ELSE 0 END AS impossible_travel,
    CASE WHEN prev_device  IS NOT NULL AND device_id  <> prev_device  THEN 1 ELSE 0 END AS new_device,
    CASE WHEN prev_country IS NOT NULL AND ip_country <> prev_country THEN 1 ELSE 0 END AS new_country,
    CAST(DATE_PART('hour', timestamp) AS INTEGER) AS hour,
    -- pandas dayofweek: Monday = 0 ... Sunday = 6
    CAST(DATE_PART('isodow', timestamp) AS INTEGER) - 1 AS dow
  FROM kinematics
),
user_vel AS (
  SELECT user_id, t10,
         CAST(COUNT(*) AS BIGINT) AS u_att,
         CAST(SUM(CASE WHEN login_result = 'failed' THEN 1 ELSE 0 END) AS BIGINT) AS u_failed
  FROM base GROUP BY user_id, t10
),
ip_vel AS (
  SELECT ip_address, t10,
         CAST(COUNT(*) AS BIGINT) AS i_att,
         CAST(SUM(CASE WHEN login_result = 'failed' THEN 1 ELSE 0 END) AS BIGINT) AS i_failed,
         CAST(COUNT(DISTINCT user_id) AS BIGINT) AS i_users
  FROM base GROUP BY ip_address, t10
),
-- Structural, whole-period counts, by design. They are identical in the pandas
-- join, in the notebook and in the BigQuery original: they describe how shared
-- a device or an IP is across the whole observation window, so they are NOT
-- past-only. Stated here rather than quietly converted into a running count.
device_sharing AS (
  SELECT device_id, CAST(COUNT(DISTINCT user_id) AS BIGINT) AS device_n_users
  FROM base GROUP BY device_id
),
ip_sharing AS (
  SELECT ip_address, CAST(COUNT(DISTINCT user_id) AS BIGINT) AS ip_n_users
  FROM base GROUP BY ip_address
)
SELECT
  f.attempt_id,
  f.user_id,
  f.timestamp,
  f.device_id,
  f.ip_address,
  f.distance_km,
  f.hours_since_last,
  f.speed_kmh,
  f.impossible_travel,
  f.new_device,
  f.new_country,
  uv.u_att,
  uv.u_failed,
  iv.i_att,
  iv.i_failed,
  iv.i_users,
  ds.device_n_users,
  ish.ip_n_users,
  f.hour,
  f.dow,
  -- diagnostic only, not one of the 15 features and excluded from the parity report
  f.distance_km_textbook
FROM flags AS f
LEFT JOIN user_vel AS uv        ON uv.user_id = f.user_id AND uv.t10 = f.t10
LEFT JOIN ip_vel AS iv          ON iv.ip_address = f.ip_address AND iv.t10 = f.t10
LEFT JOIN device_sharing AS ds  ON ds.device_id = f.device_id
LEFT JOIN ip_sharing AS ish     ON ish.ip_address = f.ip_address
ORDER BY f.user_id, f.timestamp, f.attempt_id;
