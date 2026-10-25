"""Compliance evidence: turn config/compliance_matrix.csv from a mapping into files that exist.

Each obligation in the matrix names a control and the artifact that evidences it. This module writes
those artifacts from the same scored frames, incident chains and exposure tables the rest of the
pipeline produces, then checks every row of the matrix against what was actually written. A row
marked `evidence_generated` whose file is missing or empty fails the check, so the matrix cannot
quietly overstate coverage.

What this is not: legal advice or a filing. Statutory day-counts that change by circular are kept
in the matrix's verification_note rather than computed here. The two clocks that are computed - the
GDPR 72 hours and the US SAR 30 calendar days - are fixed in the instrument text itself.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

SAR_DAYS = 30
GDPR_HOURS = 72


def str_candidates(incidents: pd.DataFrame, rings: pd.DataFrame) -> pd.DataFrame:
    """One case per confirmed takeover or ring, with the detection time the filing clock runs from."""
    ato = pd.DataFrame({
        "case_id": incidents["incident_id"], "typology": "account_takeover",
        "subjects": incidents["user_id"], "subject_count": 1,
        "detection_ts": incidents["first_signal_ts"].fillna(incidents["first_attack_ts"]),
        "fraud_value_usd": incidents["fraud_value_usd"],
        "first_signal_rule": incidents["first_signal_rule"],
    })
    ring = pd.DataFrame({
        "case_id": rings["ring_id"], "typology": "coordinated_ring",
        "subjects": rings["member_accounts"], "subject_count": rings["accounts"],
        "detection_ts": rings["first_fraud_ts"], "fraud_value_usd": rings["fraud_value_usd"],
        "first_signal_rule": "R02_shared_device_5u",
    })
    out = pd.concat([ato, ring], ignore_index=True)
    out["detection_ts"] = pd.to_datetime(out["detection_ts"])
    out["us_sar_due_date"] = (out["detection_ts"] + pd.Timedelta(days=SAR_DAYS)).dt.date
    out["in_fiu_str_due"] = "per current FIU-IND guidance - see matrix IN-01"
    return out.sort_values("detection_ts").reset_index(drop=True)


def rekyc_queue(chains: pd.DataFrame) -> pd.DataFrame:
    """Accounts whose identifiers can no longer be trusted, and whether a reset actually closed the door."""
    q = chains[["incident_id", "user_id", "reset_status", "reset_completed_ts", "first_loss_ts",
                "last_loss_ts", "paid_on_attack_device", "fraud_value_usd"]].copy()
    after_reset = (q["reset_completed_ts"].notna()
                   & (pd.to_datetime(q["last_loss_ts"]) > pd.to_datetime(q["reset_completed_ts"])))
    q["loss_continued_after_reset"] = after_reset
    q["trigger"] = q["paid_on_attack_device"].map(
        {1: "payout on attacker device", 0: "takeover confirmed"}).fillna("takeover confirmed")
    q["priority"] = (after_reset.map({True: "P1 session survived reset", False: "P2"}))
    q["action"] = "invalidate all sessions, re-verify identity, re-bind trusted device"
    return q.sort_values(["priority", "fraud_value_usd"], ascending=[True, False]).reset_index(drop=True)


def fraud_escalation_log(chains: pd.DataFrame, cm: dict) -> pd.DataFrame:
    target = cm["assumptions"]["enforcement_latency"]["target_minutes_to_action"]
    log = chains[["incident_id", "incident_type", "user_id", "first_attack_ts", "first_signal_ts",
                  "first_loss_ts", "fraud_value_usd", "preventable_value_usd",
                  "action_window_min", "first_signal_rule"]].copy()
    log["tier"] = pd.cut(log["fraud_value_usd"], [-1, 250, 1000, float("inf")],
                         labels=["T3 analyst", "T2 team lead", "T1 fraud nodal officer"]).astype(str)
    # The window is how long the first payout waited after the first signal; acting inside the SLA
    # only stops that payout if the window is at least as long as the SLA.
    log["stoppable_at_target_sla"] = log["action_window_min"] >= target
    log["customer_notice_required"] = True
    return log.sort_values("fraud_value_usd", ascending=False).reset_index(drop=True)


def payment_fraud_report(pay_all: pd.DataFrame) -> pd.DataFrame:
    """Monthly payment-fraud pack by merchant category: volume, value and fraud rate."""
    p = pay_all.assign(month=pay_all["timestamp"].dt.to_period("M").astype(str),
                       fraud_value=pay_all["amount_usd"].where(pay_all["is_fraud_txn"] == 1, 0.0))
    out = (p.groupby(["month", "merchant_category"])
           .agg(txns=("txn_id", "count"), fraud_txns=("is_fraud_txn", "sum"),
                value_usd=("amount_usd", "sum"), fraud_value_usd=("fraud_value", "sum"))
           .reset_index())
    out["fraud_rate"] = (out["fraud_txns"] / out["txns"]).round(4)
    out["fraud_value_share"] = (out["fraud_value_usd"] / out["value_usd"]).round(4)
    for c in ("value_usd", "fraud_value_usd"):
        out[c] = out[c].round(2)
    return out


def breach_assessment(campaigns: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Credential-stuffing campaigns as a personal-data breach question: who was authenticated as."""
    b = campaigns.merge(link[["campaign_id", "breached_with_fraud_txn", "already_counted_as_takeover"]],
                        on="campaign_id", how="left")
    b["detected_ts"] = pd.to_datetime(b["end_ts"])
    b["gdpr_art33_deadline"] = b["detected_ts"] + pd.Timedelta(hours=GDPR_HOURS)
    b["notify_subjects_assessment"] = b["breached_accounts"].map(
        lambda n: "assess high risk - accounts authenticated by attacker" if n else "no access gained")
    b["dpdp_board_intimation"] = b["breached_accounts"].gt(0).map(
        {True: "required - format per DPDP Rules", False: "not triggered"})
    return b[["campaign_id", "ip_address", "ip_country", "start_ts", "end_ts", "attempts",
              "targeted_accounts", "breached_accounts", "breach_rate", "breached_with_fraud_txn",
              "detected_ts", "gdpr_art33_deadline", "notify_subjects_assessment",
              "dpdp_board_intimation"]]


def control_evidence_index(control_cost: pd.DataFrame, scored: dict) -> pd.DataFrame:
    """Proof that each documented control operated in the period, not just that it is configured."""
    reg = {r["id"]: r for r in C.load_yaml("detection_rules.yaml")["rules"]}
    last_fired = {}
    for surface in ("login", "payment"):
        f = scored[surface]["test"]
        for col in [c for c in f.columns if c.startswith("flag_")]:
            ts = f.loc[f[col], "timestamp"]
            last_fired[col[5:]] = ts.max() if len(ts) else pd.NaT
    idx = control_cost[["rule_id", "surface", "disposition", "action_units", "fraud_units",
                        "precision", "verdict"]].copy()
    idx["control_name"] = idx["rule_id"].map(lambda r: reg[r]["name"])
    idx["stage"] = idx["rule_id"].map(lambda r: reg[r]["stage"])
    idx["last_fired_ts"] = idx["rule_id"].map(last_fired)
    idx["operated_in_period"] = idx["action_units"] > 0
    idx["evidence_files"] = "outputs/exposure/control_cost.csv|outputs/monitoring/rule_kpi_weekly.csv"
    return idx


def sca_decision_log(scored: dict, cm: dict) -> pd.DataFrame:
    """Monthly step-up decisions: prompts issued, who they hit, and the friction they cost."""
    reg = {r["id"]: r for r in C.load_yaml("detection_rules.yaml")["rules"]}
    fr = cm["assumptions"]["friction"]
    rows = []
    for surface, label in (("login", "is_fraud_attempt"), ("payment", "is_fraud_txn")):
        f = scored[surface]["test"]
        for rid in [r for r, s in reg.items() if s["surface"] == surface
                    and s["disposition"] == "step_up"]:
            hit = f[f[f"flag_{rid}"]]
            for month, g in hit.groupby(hit["timestamp"].dt.to_period("M").astype(str)):
                legit = int((g[label] == 0).sum())
                rows.append({"month": month, "rule_id": rid, "surface": surface,
                             "step_up_prompts": len(g), "prompts_on_fraud": int((g[label] == 1).sum()),
                             "prompts_on_legitimate": legit,
                             "est_abandoned_sessions": round(legit * fr["step_up_abandon_rate"], 1),
                             "exemption_applied": "none"})
    return pd.DataFrame(rows)


def stuffing_throttle_report(login_test: pd.DataFrame) -> pd.DataFrame:
    """IPs the velocity throttle acted on: attempts, accounts tried and how many it let through."""
    hit = login_test[login_test["flag_R01_ip_velocity_5u10m"]]
    out = (hit.groupby("ip_address")
           .agg(attempts=("attempt_id", "count"), accounts_tried=("user_id", "nunique"),
                failed=("login_result", lambda s: int((s != "success").sum())),
                first_ts=("timestamp", "min"), last_ts=("timestamp", "max"))
           .reset_index())
    ok = login_test[login_test["ip_address"].isin(out["ip_address"])
                    & (login_test["login_result"] == "success")]
    out = out.merge(ok.groupby("ip_address")["user_id"].nunique().rename("accounts_authenticated"),
                    on="ip_address", how="left").fillna({"accounts_authenticated": 0})
    out["accounts_authenticated"] = out["accounts_authenticated"].astype(int)
    out["failed_share"] = (out["failed"] / out["attempts"]).round(4)
    out["nist_800_63b_action"] = "rate-limit IP, force breached-password check on targeted accounts"
    return out.sort_values("attempts", ascending=False).reset_index(drop=True)


def build(scored: dict, rca: dict, exposure: dict, cm: dict) -> dict:
    pay_all = pd.concat([scored["payment"]["train"], scored["payment"]["test"]], ignore_index=True)
    return {
        "str_candidates": str_candidates(rca["incidents"], rca["rings"]),
        "rekyc_queue": rekyc_queue(rca["chains"]),
        "fraud_escalation_log": fraud_escalation_log(rca["chains"], cm),
        "payment_fraud_report": payment_fraud_report(pay_all),
        "breach_assessment": breach_assessment(rca["campaigns"], rca["campaign_loss_link"]),
        "control_evidence_index": control_evidence_index(exposure["control_cost"], scored),
        "sca_decision_log": sca_decision_log(scored, cm),
        "stuffing_throttle_report": stuffing_throttle_report(scored["login"]["test"]),
    }


def obligation_status(written: dict[str, int]) -> pd.DataFrame:
    """Check every matrix row against the artifacts this run wrote (path -> row count)."""
    m = pd.read_csv(C.CONFIG / "compliance_matrix.csv", encoding="utf-8")
    m = m.rename(columns={"status": "matrix_status"})
    m["artifact_rows"] = m["evidence_artifact"].map(lambda p: written.get(p, 0))
    m["evidence_present"] = m["artifact_rows"] > 0
    m["check"] = "ok"
    m.loc[(m["matrix_status"] == "evidence_generated") & ~m["evidence_present"], "check"] = \
        "FAIL: matrix claims evidence that was not generated"
    m.loc[m["matrix_status"] == "design_mapped", "check"] = "mapped only, no evidence claimed"
    m.loc[m["matrix_status"] == "out_of_scope", "check"] = "out of scope, listed for breadth"
    return m
