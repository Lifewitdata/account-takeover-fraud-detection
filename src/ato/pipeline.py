"""End-to-end run: data audit -> features -> scoring -> RCA -> exposure -> compliance -> monitoring.

`python -m ato.pipeline` (or the `ato-pipeline` entry point) rebuilds every file under outputs/
from data/*.csv and config/*. The scheduled workflow runs exactly this command, so what CI checks and
what the dashboard reads are the same bytes. Output is deterministic for a given input: no wall-clock
timestamps are written, and every model is seeded.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from . import compliance as CP
from . import config as C
from . import data as D
from . import exposure as E
from . import features as F
from . import incidents as I
from . import monitoring as M
from . import scoring as S

RCA_FILES = {
    "incidents": "incidents.csv", "chains": "incident_chains.csv",
    "root_cause_findings": "root_cause_findings.csv", "root_cause_by_stage": "root_cause_by_stage.csv",
    "primary_root_causes": "primary_root_causes.csv", "root_cause_coverage": "root_cause_coverage.csv",
    "campaigns": "stuffing_campaigns.csv", "campaign_loss_link": "campaign_loss_link.csv",
    "rings": "fraud_rings.csv", "reconciliation": "loss_reconciliation.csv",
    "edge_basis_test": "ring_edge_basis_test.csv",
}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)


def _rel(path) -> str:
    return path.relative_to(C.ROOT).as_posix()


def run(quiet: bool = False) -> dict:
    t0 = time.perf_counter()
    log = (lambda m: None) if quiet else (lambda m: _log(m, t0))
    C.ensure_dirs()
    written: dict[str, int] = {}

    def out(df: pd.DataFrame, directory, name: str) -> None:
        written[_rel(C.write_csv(df, directory, name))] = len(df)

    cm, mon = C.load_yaml("cost_model.yaml"), C.load_yaml("monitoring.yaml")
    silver = D.load_silver()
    out(silver["findings"], C.OUTPUTS, "dq_findings.csv")
    log(f"silver: {len(silver['logins']):,} logins, {len(silver['txns']):,} transactions, "
        f"{len(silver['quarantine'])} quarantined rows")

    login_features = F.add_login_features(silver["logins"])
    login = S.score_logins(silver)
    payment = S.score_payments(silver, login_features=login_features)
    scored = {"login": login, "payment": payment}
    out(pd.concat([login["metrics"], payment["metrics"]], ignore_index=True),
        C.SCORED_DIR, "model_metrics.csv")
    out(pd.concat([login["precision_at_k"], payment["precision_at_k"]], ignore_index=True),
        C.SCORED_DIR, "precision_at_k.csv")
    out(pd.concat([login["thresholds"].assign(surface="login"),
                   payment["thresholds"].assign(surface="payment")], ignore_index=True),
        C.SCORED_DIR, "threshold_table.csv")
    out(pd.concat([login["importances"], payment["importances"]], ignore_index=True),
        C.SCORED_DIR, "feature_importance.csv")
    out(pd.concat([S.rule_efficacy(login["test"], "login", "is_fraud_attempt"),
                   S.rule_efficacy(payment["test"], "payment", "is_fraud_txn")], ignore_index=True),
        C.SCORED_DIR, "rule_efficacy.csv")
    out(login["accounts"].head(500), C.SCORED_DIR, "login_account_queue.csv")
    pay_cols = ["txn_id", "user_id", "timestamp", "amount_usd", "merchant_category",
                "txn_risk_score", "is_fraud_txn"] + [c for c in payment["test"] if c.startswith("flag_")]
    out(payment["test"].sort_values("txn_risk_score", ascending=False)[pay_cols].head(500),
        C.SCORED_DIR, "payment_alert_queue.csv")
    log("scored login and payment surfaces")

    rca = I.build(silver, login_features, payment["test"])
    for key, name in RCA_FILES.items():
        out(rca[key], C.INCIDENT_DIR, name)
    log(f"RCA: {len(rca['incidents'])} takeovers, {len(rca['rings'])} rings, "
        f"{len(rca['campaigns'])} stuffing campaigns")

    exposure = E.build(scored, rca)
    for key, df in exposure.items():
        out(df, C.EXPOSURE_DIR, f"{key}.csv")
    log("exposure model written")

    for key, df in CP.build(scored, rca, exposure, cm).items():
        out(df, C.COMPLIANCE_DIR, f"{key}.csv")

    drift = M.feature_drift(scored, mon)
    weekly = M.rule_kpi_weekly(scored, mon)
    decay = M.rule_decay(weekly, mon)
    out(drift, C.MONITOR_DIR, "drift_features.csv")
    out(weekly, C.MONITOR_DIR, "rule_kpi_weekly.csv")
    out(decay, C.MONITOR_DIR, "rule_decay.csv")

    # kpi_latest is itself evidence for one obligation (STD-05), so it is counted as written before
    # the status check runs, then written for real once K08 can read that check.
    kpi_path = _rel(C.MONITOR_DIR / "kpi_latest.csv")
    obligations = CP.obligation_status({**written, kpi_path: 1})
    kpis = M.kpi_latest(weekly, decay, drift, exposure, rca["chains"], obligations, scored, mon, cm)
    out(kpis, C.MONITOR_DIR, "kpi_latest.csv")
    out(M.alerts(decay, drift, kpis, mon), C.MONITOR_DIR, "alerts.csv")
    obligations = CP.obligation_status(written)
    out(obligations, C.COMPLIANCE_DIR, "obligation_status.csv")
    log("compliance evidence and monitoring KPIs written")

    out(pd.DataFrame(sorted(written.items()), columns=["artifact", "rows"]),
        C.OUTPUTS, "run_manifest.csv")
    failures = obligations[obligations["check"].str.startswith("FAIL")]
    if len(failures):
        raise RuntimeError(f"compliance evidence missing for {failures['obligation_id'].tolist()}")
    log(f"done: {len(written)} artifacts")
    return {"scored": scored, "rca": rca, "exposure": exposure, "kpis": kpis,
            "obligations": obligations, "written": written}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-alert", action="store_true",
                        help="exit 2 if any KPI is not ok (for a scheduled run that should page)")
    args = parser.parse_args(argv)
    result = run(quiet=args.quiet)
    kpis = result["kpis"]
    print(kpis[["kpi_id", "value", "unit", "status"]].to_string(index=False))
    if args.fail_on_alert and (~kpis["status"].isin(["ok", "stable"])).any():
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
