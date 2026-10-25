"""Model training and evaluation for both detection surfaces.

Login side reproduces the notebook exactly: split by time first, compute features inside each split,
logistic regression and random forest, ROC-AUC / average precision / precision@k / threshold table.
Payment side is new in v2 and uses the same discipline on transaction rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import features as F
from . import rules as R


def _split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = raw[raw["timestamp"] < C.TRAIN_END].copy()
    test = raw[raw["timestamp"] >= C.TRAIN_END].copy()
    return train, test


def precision_at_k(scores: np.ndarray, labels: np.ndarray, ks: list[int]) -> dict[int, float]:
    order = np.argsort(scores)[::-1]
    ranked = labels[order]
    return {k: float(ranked[:k].mean()) for k in ks if k <= len(ranked)}


def threshold_table(scores: np.ndarray, labels: np.ndarray, grid: list[float],
                    n_days: int) -> pd.DataFrame:
    rows = []
    for thr in grid:
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        alerts = int(pred.sum())
        rows.append({
            "threshold": thr, "alerts": alerts, "true_positives": tp, "false_positives": fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / labels.sum() if labels.sum() else 0.0,
            "alerts_per_day": alerts / n_days if n_days else float("nan"),
            "missed_fraud": int(labels.sum()) - tp,
        })
    return pd.DataFrame(rows)


def _fit_pair(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=C.RANDOM_STATE)
    lr.fit(X, y)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                random_state=C.RANDOM_STATE)
    rf.fit(X, y)
    return {"logreg": lr, "rf": rf}


def score_logins(silver: dict, ks: list[int] | None = None,
                 grid: list[float] | None = None) -> dict:
    """Train on Jan-Sep, evaluate on Oct-Dec, and return frames plus every published metric."""
    mon = C.load_yaml("monitoring.yaml")
    ks = ks or [100, 500, 1000, 2000, 5000]
    grid = grid or mon["thresholds"]["grid"]

    train_raw, test_raw = _split(silver["logins"])
    train, test = F.add_login_features(train_raw), F.add_login_features(test_raw)
    X_train, y_train = train[F.LOGIN_FEATURES].fillna(0), train["is_fraud_attempt"]
    X_test, y_test = test[F.LOGIN_FEATURES].fillna(0), test["is_fraud_attempt"]

    models = _fit_pair(X_train, y_train)
    scores = models["rf"].predict_proba(X_test)[:, 1]
    labels = y_test.to_numpy()
    n_days = int(pd.Series(test["timestamp"].dt.date).nunique())

    metrics = pd.DataFrame([
        {"surface": "login", "model": name,
         "roc_auc": roc_auc_score(labels, m.predict_proba(X_test)[:, 1]),
         "average_precision": average_precision_score(labels, m.predict_proba(X_test)[:, 1]),
         "train_rows": len(X_train), "test_rows": len(X_test),
         "test_fraud_rate": float(labels.mean())}
        for name, m in models.items()])

    p_at_k = pd.DataFrame([{"surface": "login", "k": k, "precision": v,
                            "fraud_rows_in_window": int(labels.sum())}
                           for k, v in precision_at_k(scores, labels, ks).items()])

    test = test.assign(risk_score=scores)
    accounts = (test.groupby("user_id")
                .agg(n_attempts=("attempt_id", "count"),
                     max_score=("risk_score", "max"),
                     fraud_attempts=("is_fraud_attempt", "sum"),
                     last_ts=("timestamp", "max"))
                .reset_index().sort_values("max_score", ascending=False))

    importances = (pd.Series(models["rf"].feature_importances_, index=F.LOGIN_FEATURES)
                   .sort_values(ascending=False).rename("importance").reset_index()
                   .rename(columns={"index": "feature"}))
    importances.insert(0, "surface", "login")

    return {"train": train, "test": R.apply_rules(test, "login"),
            "models": models, "metrics": metrics,
            "thresholds": threshold_table(scores, labels, grid, n_days),
            "precision_at_k": p_at_k, "accounts": accounts,
            "importances": importances, "n_days": n_days}


def score_payments(silver: dict, ks: list[int] | None = None,
                   grid: list[float] | None = None,
                   login_features: pd.DataFrame | None = None) -> dict:
    """Score transactions with the same time discipline. Payments were described, never scored, in v1.

    Features are built once over the full transaction history and only then split by time. Every
    payment feature is past-only by construction - trailing windows look at strictly earlier rows of
    the same account, and the device/IP continuity flags compare against the first instant that pair
    was ever seen - so truncating history at the split boundary would not remove leakage, it would
    just make a September device look brand new in October. The one exception is the structural
    sharing counts (device_n_users, ip_n_users), which are whole-period by design, the same caveat
    the login features already carry.
    """
    mon = C.load_yaml("monitoring.yaml")
    grid = grid or mon["thresholds"]["grid"]
    defaults = C.load_yaml("detection_rules.yaml")["defaults"]

    lg = F.add_login_features(silver["logins"]) if login_features is None else login_features
    full = F.add_payment_features(
        silver["txns"], lg, silver["resets"],
        history_days=defaults["personal_history_days"],
        min_history=defaults["min_history_for_zscore"])
    train, test = _split(full)
    ks = ks or [25, 50, 100, 250, 500]

    X_train = train[F.PAYMENT_FEATURES].astype(float).fillna(0)
    X_test = test[F.PAYMENT_FEATURES].astype(float).fillna(0)
    y_train, y_test = train["is_fraud_txn"], test["is_fraud_txn"]

    models = _fit_pair(X_train, y_train)
    scores = models["rf"].predict_proba(X_test)[:, 1]
    labels = y_test.to_numpy()
    n_days = int(pd.Series(test["timestamp"].dt.date).nunique())

    metric_rows = [
        {"surface": "payment", "model": name, "feature_set": "all",
         "roc_auc": roc_auc_score(labels, m.predict_proba(X_test)[:, 1]),
         "average_precision": average_precision_score(labels, m.predict_proba(X_test)[:, 1]),
         "train_rows": len(X_train), "test_rows": len(X_test),
         "test_fraud_rate": float(labels.mean())}
        for name, m in models.items()]

    # The generator gives every legitimate transaction a fresh IP, which makes
    # ip_never_used_by_user a near-perfect separator. Re-fit without it so a reader can see how much
    # of the separation is data-generation artefact rather than account behaviour.
    af = [c for c in F.PAYMENT_FEATURES if c not in F.GENERATOR_ARTEFACTS]
    X_train_af, X_test_af = X_train[af], X_test[af]
    rf_af = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                   random_state=C.RANDOM_STATE).fit(X_train_af, y_train)
    s_af = rf_af.predict_proba(X_test_af)[:, 1]
    metric_rows.append({"surface": "payment", "model": "rf", "feature_set": "artefact_free",
                        "roc_auc": roc_auc_score(labels, s_af),
                        "average_precision": average_precision_score(labels, s_af),
                        "train_rows": len(X_train_af), "test_rows": len(X_test_af),
                        "test_fraud_rate": float(labels.mean())})
    metrics = pd.DataFrame(metric_rows)

    p_at_k = pd.DataFrame([{"surface": "payment", "k": k, "precision": v,
                            "fraud_rows_in_window": int(labels.sum())}
                           for k, v in precision_at_k(scores, labels, ks).items()])
    importances = (pd.Series(models["rf"].feature_importances_, index=F.PAYMENT_FEATURES)
                   .sort_values(ascending=False).rename("importance").reset_index()
                   .rename(columns={"index": "feature"}))
    importances.insert(0, "surface", "payment")
    importances["generator_artefact"] = importances["feature"].isin(F.GENERATOR_ARTEFACTS)

    test = test.assign(txn_risk_score=scores)
    return {"train": train, "test": R.apply_rules(test, "payment"), "models": models,
            "metrics": metrics, "thresholds": threshold_table(scores, labels, grid, n_days),
            "precision_at_k": p_at_k, "importances": importances, "n_days": n_days}


def rule_efficacy(f: pd.DataFrame, surface: str, label_col: str) -> pd.DataFrame:
    """How much of the labelled fraud each rule actually catches, and what it drags in with it.

    This is the table that turns a rule from a config entry into something a reviewer can argue
    about: recall share, precision, and volume cost measured on the same held-out window.
    """
    y = f[label_col].to_numpy()
    n_fraud = int(y.sum())
    rows = []
    for rid in R.rule_ids(surface):
        hit = f[f"flag_{rid}"].to_numpy()
        tp = int((hit & (y == 1)).sum())
        fp = int((hit & (y == 0)).sum())
        rows.append({
            "rule_id": rid, "surface": surface, "hits": int(hit.sum()),
            "fraud_hits": tp, "legit_hits": fp,
            "recall_share": tp / n_fraud if n_fraud else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
        })
    return pd.DataFrame(rows)
