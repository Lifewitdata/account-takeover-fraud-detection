"""Feature engineering.

`add_login_features` is a verbatim port of `add_features` in
notebooks/account_takeover_fraud_detection.ipynb so the packaged pipeline reproduces the
notebook's metrics exactly. `add_payment_features` is the transaction-level layer added in v2:
the original project scored logins and only described payments, which is why payment fraud read
as secondary.

Every feature is past-only within the frame it is computed on: a row may only use information
from earlier events for the same account. Feature frames are always built inside a time split,
never across the train/test boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_NS = 86_400_000_000_000

LOGIN_FEATURES = [
    "distance_km", "hours_since_last", "speed_kmh", "impossible_travel",
    "new_device", "new_country", "u_att", "u_failed",
    "i_att", "i_failed", "i_users", "device_n_users", "ip_n_users",
    "hour", "dow",
]

PAYMENT_NUMERIC_FEATURES = [
    "log_amount", "amount_zscore", "amount_vs_user_median", "history_n",
    "hours_since_prev_txn", "txn_count_user_24h", "txn_sum_user_24h",
    "hours_since_reset", "reset_within_24h", "device_never_used_by_user",
    "ip_never_used_by_user", "is_first_category", "high_convertible",
    "device_n_users", "ip_n_users", "hour", "dow",
    "hours_since_new_device_login",
]

PAYMENT_FEATURES = PAYMENT_NUMERIC_FEATURES

# The generator gives every legitimate transaction a fresh random IP address (60,321 distinct
# transaction IPs across 5,000 users) while an attacker re-uses the IP they already authenticated
# from. So `ip_never_used_by_user` reads 1.000 for legit and 0.028 for fraud - a perfect separator
# that is an artefact of how the synthetic data was written, not a signal to ship. It stays in the
# feature frame so the artefact is measurable, and scoring reports an artefact-free variant next to
# the headline number.
GENERATOR_ARTEFACTS = ["ip_never_used_by_user"]

# Categories that convert stolen value into something spendable or resellable quickly.
CONVERTIBLE_CATEGORIES = ("gift_cards", "jewelry", "digital_goods")
NO_PRIOR_EVENT = 9.99e6


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) ** 2 * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_login_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 15 past-only login features, identical to the notebook implementation."""
    f = df.sort_values(["user_id", "timestamp"]).copy().reset_index(drop=True)
    g = f.groupby("user_id")
    f["prev_ts"] = g["timestamp"].shift(1)
    f["prev_lat"] = g["ip_lat"].shift(1)
    f["prev_lon"] = g["ip_lon"].shift(1)
    f["prev_device"] = g["device_id"].shift(1)
    f["prev_country"] = g["ip_country"].shift(1)

    f["distance_km"] = haversine_km(f["prev_lat"], f["prev_lon"], f["ip_lat"], f["ip_lon"])
    f["hours_since_last"] = (f["timestamp"] - f["prev_ts"]).dt.total_seconds() / 3600
    f["speed_kmh"] = (f["distance_km"] / f["hours_since_last"].replace(0, np.nan)).fillna(0)
    f["impossible_travel"] = ((f["distance_km"] > 500) & (f["speed_kmh"] > 800)).astype(int)
    f["new_device"] = ((f["device_id"] != f["prev_device"]) & f["prev_device"].notna()).astype(int)
    f["new_country"] = ((f["ip_country"] != f["prev_country"]) & f["prev_country"].notna()).astype(int)

    f["t10"] = f["timestamp"].dt.floor("10min")
    u10 = (f.groupby(["user_id", "t10"])
           .agg(u_att=("attempt_id", "count"),
                u_failed=("login_result", lambda s: (s == "failed").sum())).reset_index())
    i10 = (f.groupby(["ip_address", "t10"])
           .agg(i_att=("attempt_id", "count"),
                i_failed=("login_result", lambda s: (s == "failed").sum()),
                i_users=("user_id", "nunique")).reset_index())
    f = f.merge(u10, on=["user_id", "t10"], how="left")
    f = f.merge(i10, on=["ip_address", "t10"], how="left")

    f = f.join(f.groupby("device_id")["user_id"].nunique().rename("device_n_users"), on="device_id")
    f = f.join(f.groupby("ip_address")["user_id"].nunique().rename("ip_n_users"), on="ip_address")

    f["hour"] = f["timestamp"].dt.hour
    f["dow"] = f["timestamp"].dt.dayofweek
    return f


def _trailing_stats(ts: np.ndarray, value: np.ndarray, lookback_days: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count / sum / sum-of-squares of strictly-earlier rows inside a lookback, per row.

    `ts` must be ascending. searchsorted on the account's own event times makes the window
    boundary exact rather than bucket-approximate.
    """
    idx = np.arange(len(ts))
    start = np.searchsorted(ts, ts - lookback_days * DAY_NS, side="left")
    cum = np.concatenate([[0.0], np.cumsum(value)])
    cum2 = np.concatenate([[0.0], np.cumsum(value.astype(float) ** 2)])
    return idx - start, cum[idx] - cum[start], cum2[idx] - cum2[start]


def _first_seen(events: pd.DataFrame, key: str) -> pd.DataFrame:
    """Earliest time each (user, key) pair was observed - the leak-free basis for 'never seen before'."""
    return (events.groupby(["user_id", key], as_index=False)["timestamp"]
            .min().rename(columns={"timestamp": f"first_{key}"}))


def add_payment_features(txns: pd.DataFrame, login_features: pd.DataFrame, resets: pd.DataFrame,
                         history_days: int = 90, min_history: int = 5) -> pd.DataFrame:
    """Past-only transaction features, including the reset-to-payout and device-continuity chains.

    `login_features` must come from `add_login_features` - the payment layer reads the login-level
    `new_device` flag rather than re-deriving it, because that is what carries the takeover signal
    forward to the moment money moves. Sharing features (device_n_users, ip_n_users) are structural
    counts over the frame passed in, matching the convention the login features already use.
    """
    logins = login_features
    t = txns.sort_values(["user_id", "timestamp"]).copy().reset_index(drop=True)
    amount = t["amount_usd"].to_numpy(dtype=float)
    # pandas 3 parses these columns at microsecond resolution; cast explicitly so DAY_NS means a day.
    ts_ns = t["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")

    n_prior = np.zeros(len(t))
    mean_prior = np.full(len(t), np.nan)
    std_prior = np.full(len(t), np.nan)
    n24 = np.zeros(len(t))
    s24 = np.zeros(len(t))

    for pos in t.groupby("user_id", sort=False).indices.values():
        pos = np.sort(pos)
        n, s, sq = _trailing_stats(ts_ns[pos], amount[pos], history_days)
        n_prior[pos] = n
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_prior[pos] = np.divide(s, n, out=np.full(n.shape, np.nan), where=n > 0)
            var = np.divide(sq, n, out=np.full(n.shape, np.nan), where=n > 0) - mean_prior[pos] ** 2
        std_prior[pos] = np.sqrt(np.maximum(var, 0.0))
        c24, sum24, _ = _trailing_stats(ts_ns[pos], amount[pos], 1.0)
        n24[pos] = c24 + 1
        s24[pos] = sum24 + amount[pos]

    t["history_n"] = n_prior
    t["amount_zscore"] = np.where(
        (n_prior >= min_history) & (std_prior > 0), (amount - mean_prior) / std_prior, 0.0)
    t["txn_count_user_24h"] = n24
    t["txn_sum_user_24h"] = s24
    t["log_amount"] = np.log1p(amount)

    g = t.groupby("user_id")
    t["hours_since_prev_txn"] = (t["timestamp"] - g["timestamp"].shift(1)).dt.total_seconds() / 3600
    prev_median = g["amount_usd"].transform(lambda s: s.shift(1).expanding().median())
    t["amount_vs_user_median"] = np.where(
        prev_median.to_numpy() > 0, amount / prev_median.to_numpy(), 0.0)

    rst = (resets.sort_values("request_timestamp")[["user_id", "request_timestamp", "status"]]
           .rename(columns={"status": "last_reset_status"}))
    t = pd.merge_asof(t.sort_values("timestamp"), rst, left_on="timestamp",
                      right_on="request_timestamp", by="user_id", direction="backward")
    t["hours_since_reset"] = (t["timestamp"] - t["request_timestamp"]).dt.total_seconds() / 3600
    t["hours_since_reset"] = t["hours_since_reset"].fillna(NO_PRIOR_EVENT)
    t["reset_within_24h"] = t["hours_since_reset"].between(0, 24).astype(int)

    for key, flag in (("device_id", "device_never_used_by_user"),
                      ("ip_address", "ip_never_used_by_user")):
        fs = _first_seen(logins[["user_id", key, "timestamp"]], key)
        # A plain equi-join, not merge_asof: the first-seen instant is a fixed property of the
        # (user, key) pair, so the test is simply whether it falls after this transaction.
        t = t.merge(fs, on=["user_id", key], how="left")
        first_col = f"first_{key}"
        t[flag] = (t[first_col].isna() | (t[first_col] > t["timestamp"])).astype(int)
        t = t.drop(columns=[first_col])
    t["paid_on_new_device"] = t["device_never_used_by_user"]

    # R06 measures device continuity at payment time and fails for a causal reason: an attacker who
    # authenticates on a new device first makes that device "known" before they spend. Carrying the
    # login-level flag forward instead recovers the signal.
    nd = (logins.loc[(logins["new_device"] == 1) & (logins["login_result"] == "success"),
                     ["user_id", "timestamp"]]
          .rename(columns={"timestamp": "nd_login_ts"}).sort_values("nd_login_ts"))
    t = pd.merge_asof(t.sort_values("timestamp"), nd, left_on="timestamp", right_on="nd_login_ts",
                      by="user_id", direction="backward", tolerance=pd.Timedelta("24h"))
    t["hours_since_new_device_login"] = ((t["timestamp"] - t["nd_login_ts"])
                                         .dt.total_seconds() / 3600).fillna(NO_PRIOR_EVENT)
    t = t.drop(columns=["nd_login_ts"])

    # First occurrence in time of this category for this account; txns are already ordered by time
    # within account, so cumcount 0 is exactly "no earlier transaction in this category".
    t["is_first_category"] = (t.groupby(["user_id", "merchant_category"]).cumcount() == 0).astype(int)

    t["high_convertible"] = t["merchant_category"].isin(CONVERTIBLE_CATEGORIES).astype(int)
    t = t.join(logins.groupby("device_id")["user_id"].nunique().rename("device_n_users"), on="device_id")
    t = t.join(logins.groupby("ip_address")["user_id"].nunique().rename("ip_n_users"), on="ip_address")
    t[["device_n_users", "ip_n_users"]] = t[["device_n_users", "ip_n_users"]].fillna(0)
    t["hour"] = t["timestamp"].dt.hour
    t["dow"] = t["timestamp"].dt.dayofweek
    return t.sort_values("timestamp").reset_index(drop=True)
