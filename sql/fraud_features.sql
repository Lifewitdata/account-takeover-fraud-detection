-- ============================================================================
-- fraud_features.sql — Trust & Safety: login-level fraud features
-- Dialect: BigQuery Standard SQL (written carefully; not run live — no warehouse creds)
-- Mirrors the feature engineering in notebooks/account_takeover_fraud_detection.ipynb
--
-- Source table : `project.dataset.login_attempts`
--                (attempt_id, user_id, timestamp, ip_address, device_id,
--                 ip_country, ip_city, ip_lat, ip_lon, login_result,
--                 failure_reason, is_fraud_attempt)
-- NOTE: is_fraud_attempt is a synthetic ground-truth label for evaluation only.
--       It is deliberately EXCLUDED from every feature below.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Query 1: login_features — one row per login attempt, 15 past-only features
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW `project.dataset.login_features` AS
WITH base AS (
  SELECT
    *,
    -- 10-minute bucket for velocity features
    TIMESTAMP_SECONDS(600 * DIV(UNIX_SECONDS(timestamp), 600)) AS t10
  FROM `project.dataset.login_attempts`
),
with_prev AS (
  SELECT
    b.*,
    LAG(timestamp)  OVER w AS prev_ts,
    LAG(ip_lat)     OVER w AS prev_lat,
    LAG(ip_lon)     OVER w AS prev_lon,
    LAG(device_id)  OVER w AS prev_device,
    LAG(ip_country) OVER w AS prev_country
  FROM base b
  WINDOW w AS (PARTITION BY user_id ORDER BY timestamp)
),
movement AS (
  SELECT
    *,
    -- haversine distance (km) between consecutive logins of the same user
    6371 * 2 * ASIN(SQRT(
      POW(SIN((RADIANS(ip_lat) - RADIANS(prev_lat)) / 2), 2) +
      COS(RADIANS(prev_lat)) * COS(RADIANS(ip_lat)) *
      POW(SIN((RADIANS(ip_lon) - RADIANS(prev_lon)) / 2), 2)
    )) AS distance_km,
    TIMESTAMP_DIFF(timestamp, prev_ts, MINUTE) / 60.0 AS hours_since_last
  FROM with_prev
),
flags AS (
  SELECT
    *,
    COALESCE(distance_km / NULLIF(hours_since_last, 0), 0) AS speed_kmh,
    CASE
      WHEN COALESCE(distance_km, 0) > 500
       AND COALESCE(distance_km / NULLIF(hours_since_last, 0), 0) > 800
      THEN 1 ELSE 0
    END AS impossible_travel,
    CASE WHEN prev_device IS NOT NULL AND device_id != prev_device THEN 1 ELSE 0 END AS new_device,
    CASE WHEN prev_country IS NOT NULL AND ip_country != prev_country THEN 1 ELSE 0 END AS new_country,
    EXTRACT(HOUR FROM timestamp) AS hour,
    EXTRACT(DAYOFWEEK FROM timestamp) AS dow  -- 1 = Sunday … 7 = Saturday
  FROM movement
),
user_vel AS (
  SELECT
    user_id, t10,
    COUNT(*) AS u_att,
    COUNTIF(login_result = 'failed') AS u_failed
  FROM base
  GROUP BY 1, 2
),
ip_vel AS (
  SELECT
    ip_address, t10,
    COUNT(*) AS i_att,
    COUNTIF(login_result = 'failed') AS i_failed,
    COUNT(DISTINCT user_id) AS i_users
  FROM base
  GROUP BY 1, 2
),
device_sharing AS (
  SELECT device_id, COUNT(DISTINCT user_id) AS device_n_users
  FROM base
  GROUP BY 1
),
ip_sharing AS (
  SELECT ip_address, COUNT(DISTINCT user_id) AS ip_n_users
  FROM base
  GROUP BY 1
)
SELECT
  f.* EXCEPT (t10, prev_ts, prev_lat, prev_lon, prev_device, prev_country),
  uv.u_att,
  uv.u_failed,
  iv.i_att,
  iv.i_failed,
  iv.i_users,
  ds.device_n_users,
  ish.ip_n_users
FROM flags f
LEFT JOIN user_vel uv      USING (user_id, t10)
LEFT JOIN ip_vel iv        USING (ip_address, t10)
LEFT JOIN device_sharing ds  USING (device_id)
LEFT JOIN ip_sharing ish     USING (ip_address);

-- ----------------------------------------------------------------------------
-- Query 2: account_risk_rollup — one row per user for the review queue
-- (heuristic triage score; the calibrated model score lives in the notebook)
-- ----------------------------------------------------------------------------
SELECT
  user_id,
  COUNT(*) AS n_attempts,
  SUM(impossible_travel) AS n_impossible_travel,
  SUM(new_device) AS n_new_device,
  SUM(new_country) AS n_new_country,
  MAX(i_users) AS max_users_per_ip_window,
  MAX(device_n_users) AS max_users_per_device,
  ( SUM(impossible_travel) * 3
    + SUM(new_device) * 1
    + COUNTIF(i_users >= 5) * 2
    + COUNTIF(device_n_users >= 5) * 2
  ) AS heuristic_score
FROM `project.dataset.login_features`
GROUP BY user_id
ORDER BY heuristic_score DESC;

-- ----------------------------------------------------------------------------
-- Query 3: ip_watchlist — credential-stuffing shaped windows (20+ attempts,
-- 5+ distinct users per 10 minutes from a single IP)
-- ----------------------------------------------------------------------------
SELECT
  ip_address,
  TIMESTAMP_SECONDS(600 * DIV(UNIX_SECONDS(timestamp), 600)) AS t10,
  COUNT(*) AS attempts,
  COUNTIF(login_result = 'failed') AS failed,
  COUNT(DISTINCT user_id) AS users,
  COUNT(DISTINCT device_id) AS devices
FROM `project.dataset.login_attempts`
GROUP BY 1, 2
HAVING attempts >= 20 AND users >= 5
ORDER BY attempts DESC;
