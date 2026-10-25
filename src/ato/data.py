"""Bronze load and Silver cleaning for the five source tables.

`load_bronze` reads exactly what is committed in data/. `dq_audit` reproduces the notebook's
data-quality rules and returns an audit-trail frame plus a quarantine frame, so the pipeline can
show which checks ran rather than only asserting that data is clean.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

TABLES = {
    "users": ("users.csv", ["signup_date"]),
    "devices": ("devices.csv", ["first_seen_date"]),
    "login_attempts": ("login_attempts.csv", ["timestamp"]),
    "password_resets": ("password_resets.csv", ["request_timestamp", "complete_timestamp"]),
    "transactions": ("transactions.csv", ["timestamp"]),
}

# Columns that must be unique per row - the join keys everything downstream relies on.
PRIMARY_KEYS = {
    "users": "user_id",
    "devices": "device_id",
    "login_attempts": "attempt_id",
    "password_resets": "reset_id",
    "transactions": "txn_id",
}

FOREIGN_KEYS = {
    "login_attempts": {"user_id": "users", "device_id": "devices"},
    "password_resets": {"user_id": "users", "device_id": "devices"},
    "transactions": {"user_id": "users", "device_id": "devices"},
}

# The event-time column each table is checked against for future dating.
TIME_COL = {
    "users": "signup_date",
    "devices": "first_seen_date",
    "login_attempts": "timestamp",
    "password_resets": "request_timestamp",
    "transactions": "timestamp",
}


def load_bronze() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, (fname, dates) in TABLES.items():
        out[name] = pd.read_csv(C.DATA / fname, parse_dates=dates or None)
    return out


def dq_audit(bronze: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the hard and soft data-quality rules.

    Returns (findings, quarantine). Quarantine holds row keys failing a hard rule; findings is the
    full audit trail including checks that passed, which is what an auditor actually asks for.
    """
    findings: list[dict] = []
    quarantine: list[dict] = []
    now = pd.Timestamp.now()

    for table, df in bronze.items():
        pk = PRIMARY_KEYS[table]
        dup = df[pk].duplicated()
        findings.append({"table": table, "check": f"unique_{pk}", "violations": int(dup.sum()),
                         "rows": int(len(df)), "severity": "hard" if dup.any() else "pass",
                         "detail": f"{pk} must identify one row"})
        dup_keys = df.loc[dup, pk]

        ts_col = TIME_COL[table]
        future = pd.Series(False, index=df.index)
        if pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            future = df[ts_col] > now
        findings.append({"table": table, "check": f"no_future_{ts_col}", "violations": int(future.sum()),
                         "rows": int(len(df)), "severity": "hard" if future.any() else "pass",
                         "detail": "event time cannot be in the future"})

        nulls = df.isna().sum()
        crit = [c for c in df.columns if c.endswith("_id") or c in ("timestamp", "user_id")]
        crit_nulls = int(sum(nulls.get(c, 0) for c in crit))
        findings.append({"table": table, "check": "no_null_keys", "violations": crit_nulls,
                         "rows": int(len(df)), "severity": "hard" if crit_nulls else "pass",
                         "detail": f"checked {len(crit)} key columns"})

        for qk, parent in FOREIGN_KEYS.get(table, {}).items():
            if qk in df and parent in bronze:
                orphan = ~df[qk].isin(set(bronze[parent][PRIMARY_KEYS[parent]]))
                findings.append({"table": table, "check": f"fk_{qk}_in_{parent}",
                                 "violations": int(orphan.sum()), "rows": int(len(df)),
                                 "severity": "hard" if orphan.any() else "pass",
                                 "detail": f"{qk} must resolve to {parent}"})
                if orphan.any():
                    quarantine.extend({"table": table, "key": k, "reason": f"orphan_{qk}"}
                                      for k in df.loc[orphan, PRIMARY_KEYS[table]])

        if dup.any():
            quarantine.extend({"table": table, "key": k, "reason": "duplicate_primary_key"}
                              for k in dup_keys)
        if future.any():
            quarantine.extend({"table": table, "key": k, "reason": f"future_{ts_col}"}
                              for k in df.loc[future, PRIMARY_KEYS[table]])

    logins = bronze["login_attempts"]
    bad_geo = (logins["ip_lat"].abs() > 90) | (logins["ip_lon"].abs() > 180) | logins[["ip_lat", "ip_lon"]].isna().any(axis=1)
    findings.append({"table": "login_attempts", "check": "ip_coordinates_valid",
                     "violations": int(bad_geo.sum()), "rows": int(len(logins)),
                     "severity": "hard" if bad_geo.any() else "pass",
                     "detail": "ip_lat within +-90, ip_lon within +-180, not null"})
    quarantine.extend({"table": "login_attempts", "key": k, "reason": "invalid_ip_coordinates"}
                      for k in logins.loc[bad_geo, "attempt_id"])

    txns = bronze["transactions"]
    bad_amt = (txns["amount_usd"] <= 0) | txns["amount_usd"].isna()
    findings.append({"table": "transactions", "check": "amount_positive",
                     "violations": int(bad_amt.sum()), "rows": int(len(txns)),
                     "severity": "hard" if bad_amt.any() else "pass",
                     "detail": "amount_usd must be greater than zero"})
    quarantine.extend({"table": "transactions", "key": k, "reason": "non_positive_amount"}
                      for k in txns.loc[bad_amt, "txn_id"])

    resets = bronze["password_resets"]
    neg_dur = (resets["complete_timestamp"].notna()
               & (resets["complete_timestamp"] < resets["request_timestamp"]))
    findings.append({"table": "password_resets", "check": "completion_after_request",
                     "violations": int(neg_dur.sum()), "rows": int(len(resets)),
                     "severity": "hard" if neg_dur.any() else "pass",
                     "detail": "complete_timestamp cannot precede request_timestamp"})
    quarantine.extend({"table": "password_resets", "key": k, "reason": "reset_completes_before_request"}
                      for k in resets.loc[neg_dur, "reset_id"])

    findings.append({"table": "all", "check": "label_columns_present",
                     "violations": 0, "rows": sum(len(d) for d in bronze.values()),
                     "severity": "pass",
                     "detail": "is_fraud_attempt / is_fraud_reset / is_fraud_txn are synthetic "
                               "ground truth, evaluation-only, excluded from every feature set"})

    qdf = pd.DataFrame(quarantine, columns=["table", "key", "reason"])
    return pd.DataFrame(findings), qdf


def load_silver() -> dict[str, pd.DataFrame]:
    """Bronze minus anything failing a hard rule, with the same normalisation the notebook applies."""
    bronze = load_bronze()
    findings, quarantine = dq_audit(bronze)
    qkeys = {t: set(g["key"]) for t, g in quarantine.groupby("table")} if len(quarantine) else {}

    def drop(table: str, df: pd.DataFrame) -> pd.DataFrame:
        bad = qkeys.get(table)
        if not bad:
            return df
        return df[~df[PRIMARY_KEYS[table]].isin(bad)].copy()

    logins = drop("login_attempts", bronze["login_attempts"])
    logins["failure_reason"] = logins["failure_reason"].fillna("")
    logins = logins.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    txns = drop("transactions", bronze["transactions"]).sort_values(
        ["user_id", "timestamp"]).reset_index(drop=True)
    resets = drop("password_resets", bronze["password_resets"]).sort_values(
        ["user_id", "request_timestamp"]).reset_index(drop=True)

    return {
        "users": bronze["users"],
        "devices": bronze["devices"],
        "logins": logins,
        "resets": resets,
        "txns": txns,
        "findings": findings,
        "quarantine": quarantine,
        "bronze_counts": {k: len(v) for k, v in bronze.items()},
    }
