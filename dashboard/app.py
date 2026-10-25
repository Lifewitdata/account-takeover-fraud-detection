# Trust & Safety - account takeover and payment fraud. Reads only what `python -m ato.pipeline` writes.
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="ATO & Payment Fraud Risk", layout="wide")
st.title("Account takeover and payment fraud")


@st.cache_data
def read(sub: str, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(OUT / sub / name, **kw)


if not (OUT / "run_manifest.csv").exists():
    st.error("No pipeline outputs found. Run `python -m ato.pipeline` from the repo root.")
    st.stop()

kpis = read("monitoring", "kpi_latest.csv")
alerts = read("monitoring", "alerts.csv")
summary = read("exposure", "exposure_summary.csv").set_index("line")

st.caption(f"Data window ends {kpis['as_of'].iloc[0]}. Every number below is recomputed from data/ "
           "by the pipeline; lines marked 'assumption' come from config/cost_model.yaml.")
c = st.columns(5)
c[0].metric("Fraud value, whole period", f"${summary.loc['fraud_value_whole_period', 'value']:,.0f}")
c[1].metric("Total fraud cost", f"${summary.loc['total_fraud_cost', 'value']:,.0f}")
c[2].metric("Control cost (deduplicated)",
            f"${summary.loc['programme_control_cost_deduplicated', 'value']:,.0f}")
c[3].metric("Review cases / day", f"{kpis.set_index('kpi_id').loc['K03_reviewer_load', 'value']:.1f}")
c[4].metric("Open alerts", len(alerts))

tabs = st.tabs(["Monitoring", "Incidents & RCA", "Exposure", "Payments", "Compliance",
                "Review queue"])

with tabs[0]:
    st.subheader("KPIs")
    st.dataframe(kpis[["kpi_id", "name", "value", "unit", "threshold", "status", "detail"]],
                 hide_index=True, use_container_width=True)
    st.subheader("Open alerts")
    st.dataframe(alerts, hide_index=True, use_container_width=True)
    weekly = read("monitoring", "rule_kpi_weekly.csv", parse_dates=["week_start"])
    rule = st.selectbox("Rule", sorted(weekly["rule_id"].unique()))
    w = weekly[(weekly["rule_id"] == rule) & (~weekly["below_min_events"])]
    st.plotly_chart(px.line(w, x="week_start", y=["precision", "recall"], color_discrete_sequence=None,
                            title=f"{rule}: weekly precision and recall (baseline Jan-Sep, current Oct-Dec)"),
                    use_container_width=True)
    st.subheader("Rule decay vs baseline")
    st.dataframe(read("monitoring", "rule_decay.csv"), hide_index=True, use_container_width=True)
    drift = read("monitoring", "drift_features.csv")
    pop = st.radio("Drift population", ["legitimate_only", "all_rows"], horizontal=True)
    d = drift[drift["population"] == pop]
    st.plotly_chart(px.bar(d, x="psi", y="feature", color="psi_status", facet_col="surface",
                           orientation="h", title="Feature PSI, current vs baseline"),
                    use_container_width=True)

with tabs[1]:
    rec = read("incidents", "loss_reconciliation.csv")
    st.subheader("Where every fraud dollar is attributed")
    st.dataframe(rec, hide_index=True, use_container_width=True)
    st.subheader("Primary root cause (earliest-stage gap, additive)")
    st.dataframe(read("incidents", "primary_root_causes.csv"), hide_index=True,
                 use_container_width=True)
    st.subheader("All root causes by kill-chain stage (overlapping)")
    st.dataframe(read("incidents", "root_cause_by_stage.csv"), hide_index=True,
                 use_container_width=True)
    chains = read("incidents", "incident_chains.csv")
    st.subheader(f"Confirmed incidents ({len(chains)})")
    pick = st.selectbox("Incident", chains.sort_values("fraud_value_usd", ascending=False)["incident_id"])
    st.json(chains.set_index("incident_id").loc[pick].dropna().astype(str).to_dict())
    findings = read("incidents", "root_cause_findings.csv")
    st.dataframe(findings[findings["incident_id"] == pick], hide_index=True, use_container_width=True)
    st.subheader("Fraud rings and stuffing campaigns")
    st.dataframe(read("incidents", "fraud_rings.csv"), hide_index=True, use_container_width=True)
    st.dataframe(read("incidents", "campaign_loss_link.csv"), hide_index=True,
                 use_container_width=True)

with tabs[2]:
    st.subheader("Exposure summary")
    st.dataframe(summary.reset_index(), hide_index=True, use_container_width=True)
    cc = read("exposure", "control_cost.csv")
    st.subheader("Per-rule cost and verdict (standalone view)")
    st.dataframe(cc[["rule_id", "surface", "disposition", "action_units", "precision",
                     "total_control_cost_usd", "benefit_usd", "benefit_basis", "net_position_usd",
                     "verdict"]], hide_index=True, use_container_width=True)
    lat = read("exposure", "enforcement_latency.csv")
    st.plotly_chart(px.line(lat, x="act_within_minutes", y="value_stoppable_usd", markers=True,
                            log_x=True, title="Fraud value still stoppable if the first signal is "
                                              "acted on within N minutes"), use_container_width=True)
    st.subheader("Threshold economics")
    st.dataframe(read("exposure", "threshold_economics.csv"), hide_index=True,
                 use_container_width=True)
    st.subheader("Assumption ledger")
    st.dataframe(read("exposure", "assumption_ledger.csv"), hide_index=True,
                 use_container_width=True)

with tabs[3]:
    report = read("compliance", "payment_fraud_report.csv")
    monthly = report.groupby("month", as_index=False)[["fraud_value_usd", "fraud_txns"]].sum()
    st.plotly_chart(px.bar(monthly, x="month", y="fraud_value_usd", title="Payment fraud value by month"),
                    use_container_width=True)
    cat = report.groupby("merchant_category", as_index=False)[["fraud_value_usd", "value_usd"]].sum()
    cat["fraud_value_share"] = cat["fraud_value_usd"] / cat["value_usd"]
    st.plotly_chart(px.bar(cat.sort_values("fraud_value_usd"), x="fraud_value_usd",
                           y="merchant_category", orientation="h",
                           title="Payment fraud value by merchant category"), use_container_width=True)
    st.subheader("Payment alert queue (top 500 by model score, test window)")
    st.dataframe(read("scored", "payment_alert_queue.csv"), hide_index=True,
                 use_container_width=True)
    st.subheader("Rule efficacy, both surfaces")
    st.dataframe(read("scored", "rule_efficacy.csv"), hide_index=True, use_container_width=True)
    st.dataframe(read("scored", "model_metrics.csv"), hide_index=True, use_container_width=True)

with tabs[4]:
    ob = read("compliance", "obligation_status.csv")
    juris = st.multiselect("Jurisdiction", sorted(ob["jurisdiction"].unique()),
                           default=sorted(ob["jurisdiction"].unique()))
    st.dataframe(ob[ob["jurisdiction"].isin(juris)][
        ["obligation_id", "jurisdiction", "instrument", "control_name", "evidence_artifact",
         "matrix_status", "artifact_rows", "check", "verification_note"]],
        hide_index=True, use_container_width=True)
    art = st.selectbox("Evidence artifact", sorted(p.name for p in (OUT / "compliance").glob("*.csv")))
    st.dataframe(read("compliance", art), hide_index=True, use_container_width=True)

with tabs[5]:
    accts = read("scored", "login_account_queue.csv")
    thr = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
    q = accts[accts["max_score"] >= thr]
    st.write(f"{len(q):,} accounts at or above threshold (top 500 by score are kept)")
    st.dataframe(q, hide_index=True, use_container_width=True)
