"""Scheduled monitoring: feature drift, weekly rule KPIs, rule decay and one KPI row per tracked metric.

v1 ended with "re-tune thresholds monthly". This module is what makes that a mechanism: every run
compares the held-out window against the window the rules were tuned on, and writes a dated alert
with an owner and a due date whenever a configured threshold in config/monitoring.yaml is crossed.
The scheduled GitHub Actions workflow runs it weekly; locally it is `python -m ato.pipeline`.

Two measurement choices worth knowing before reading the outputs:

* Drift is reported twice per feature - on every row, and on legitimate rows only. Attack traffic
  arrives in bursts, so the all-rows PSI moves whenever a campaign lands. The legitimate-only PSI is
  the one that says the customer population itself has shifted, which is what invalidates a threshold.
* R09 is the model-score rule and the model was fitted on the baseline weeks, so it has no honest
  baseline-window precision. Its baseline cells are left empty rather than filled in-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import features as F
from . import rules as R

SURFACES = {
    "login": {"label": "is_fraud_attempt", "features": F.LOGIN_FEATURES, "score": "risk_score"},
    "payment": {"label": "is_fraud_txn", "score": "txn_risk_score",
                "features": [c for c in F.PAYMENT_FEATURES if c not in F.GENERATOR_ARTEFACTS]},
}
IN_SAMPLE_RULES = {"R09_model_score"}
ACTIONING = ("auto_block", "review", "step_up", "hold")


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with baseline-quantile bins; discrete features bin by value."""
    baseline, current = baseline[~np.isnan(baseline)], current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return float("nan")
    values = np.unique(baseline)
    if len(values) <= bins:
        cats = np.union1d(values, np.unique(current))
        b = np.array([(baseline == v).mean() for v in cats])
        c = np.array([(current == v).mean() for v in cats])
    else:
        edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1))[1:-1])
        b = np.bincount(np.searchsorted(edges, baseline, side="right"), minlength=len(edges) + 1)
        c = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1)
        b, c = b / b.sum(), c / c.sum()
    b, c = np.clip(b, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - b) * np.log(c / b)))


def feature_drift(scored: dict, mon: dict) -> pd.DataFrame:
    """PSI and KS per feature, baseline (train) window vs current (test) window.

    Payment features use trailing per-account windows, so the first `personal_history_days` of data
    have structurally short histories. Those weeks are dropped from the payment baseline; otherwise
    the warm-up reads as drift that no threshold change could fix.
    """
    cfg = mon["drift"]
    warmup = pd.Timedelta(days=C.load_yaml("detection_rules.yaml")["defaults"]["personal_history_days"])
    rows = []
    for surface, spec in SURFACES.items():
        base, cur = scored[surface]["train"], scored[surface]["test"]
        if surface == "payment":
            base = base[base["timestamp"] >= base["timestamp"].min() + warmup]
        baseline_from = base["timestamp"].min().date().isoformat()
        for population in ("all_rows", "legitimate_only"):
            if population == "legitimate_only":
                b_f, c_f = base[base[spec["label"]] == 0], cur[cur[spec["label"]] == 0]
            else:
                b_f, c_f = base, cur
            for feat in spec["features"]:
                b = b_f[feat].astype(float).to_numpy()
                c = c_f[feat].astype(float).to_numpy()
                value = psi(b, c, cfg["psi"]["bins"])
                ks = stats.ks_2samp(b[~np.isnan(b)], c[~np.isnan(c)])
                status = ("severe" if value >= cfg["psi"]["severe_from"] else
                          "moderate" if value >= cfg["psi"]["moderate_from"] else "stable")
                rows.append({
                    "surface": surface, "population": population, "feature": feat,
                    "baseline_from": baseline_from, "baseline_rows": len(b), "current_rows": len(c),
                    "baseline_mean": round(float(np.nanmean(b)), 4),
                    "current_mean": round(float(np.nanmean(c)), 4),
                    "psi": round(value, 4), "psi_status": status,
                    "ks_stat": round(float(ks.statistic), 4), "ks_p_value": float(ks.pvalue),
                    "ks_alert": bool(ks.statistic > cfg["ks"]["alert_above"]
                                     and ks.pvalue < cfg["ks"]["p_value_alert_below"]),
                })
    return pd.DataFrame(rows).sort_values(["surface", "population", "psi"],
                                          ascending=[True, True, False]).reset_index(drop=True)


def _flagged(scored: dict, surface: str) -> pd.DataFrame:
    """Baseline and current rows with rule flags; baseline R09 is masked, see module docstring."""
    spec = SURFACES[surface]
    base = scored[surface]["train"].copy()
    base[spec["score"]] = np.nan
    base = R.apply_rules(base, surface).assign(window="baseline")
    cur = scored[surface]["test"].assign(window="current")
    both = pd.concat([base, cur], ignore_index=True)
    for rid in IN_SAMPLE_RULES & set(R.rule_ids(surface)):
        both[f"flag_{rid}"] = both[f"flag_{rid}"].astype(object)
        both.loc[both["window"] == "baseline", f"flag_{rid}"] = np.nan
    return both


def rule_kpi_weekly(scored: dict, mon: dict) -> pd.DataFrame:
    w = mon["windows"]
    disposition = {r["id"]: r["disposition"] for r in R.load_rules()["rules"]}
    rows = []
    for surface, spec in SURFACES.items():
        f = _flagged(scored, surface)
        f["week_start"] = f["timestamp"].dt.to_period(w["period"]).dt.start_time
        y = f[spec["label"]] == 1
        actioning = [r for r in R.rule_ids(surface)
                     if disposition[r] in ACTIONING and r not in IN_SAMPLE_RULES]
        f["flag_ANY_ACTION"] = f[[f"flag_{r}" for r in actioning]].astype(bool).any(axis=1)
        for rid in R.rule_ids(surface) + ["ANY_ACTION"]:
            col = f"flag_{rid}"
            for (week, window), g in f.groupby(["week_start", "window"]):
                gy = y.loc[g.index]
                masked = g[col].isna().all()
                hit = g[col].fillna(False).astype(bool)
                fraud_events = int(gy.sum())
                tp, fp = int((hit & gy).sum()), int((hit & ~gy).sum())
                rows.append({
                    "surface": surface, "rule_id": rid, "window": window,
                    "week_start": week.date().isoformat(), "events": len(g),
                    "fraud_events": fraud_events,
                    "alerts": np.nan if masked else tp + fp,
                    "fraud_hits": np.nan if masked else tp,
                    "legit_hits": np.nan if masked else fp,
                    "precision": np.nan if masked or not tp + fp else round(tp / (tp + fp), 4),
                    "recall": np.nan if masked or not fraud_events else round(tp / fraud_events, 4),
                    "below_min_events": len(g) < w["min_weekly_events"],
                })
    return pd.DataFrame(rows).sort_values(["surface", "rule_id", "week_start"]).reset_index(drop=True)


def rule_decay(weekly: pd.DataFrame, mon: dict) -> pd.DataFrame:
    """Pooled precision and recall per window, and the relative drop the SLA is written against."""
    d = mon["drift"]["rule_decay"]
    use = weekly[~weekly["below_min_events"]]
    rows = []
    for (surface, rid), g in use.groupby(["surface", "rule_id"]):
        rec = {"surface": surface, "rule_id": rid}
        for window in ("baseline", "current"):
            s = g[g["window"] == window]
            if s["alerts"].isna().all():
                rec[f"{window}_precision"] = rec[f"{window}_recall"] = np.nan
                rec[f"{window}_alerts"] = np.nan
                continue
            tp, alerts, fraud = s["fraud_hits"].sum(), s["alerts"].sum(), s["fraud_events"].sum()
            rec[f"{window}_alerts"] = int(alerts)
            rec[f"{window}_precision"] = round(tp / alerts, 4) if alerts else np.nan
            rec[f"{window}_recall"] = round(tp / fraud, 4) if fraud else np.nan
        worst, worst_metric = np.nan, ""
        for metric in ("precision", "recall"):
            b, c = rec[f"baseline_{metric}"], rec[f"current_{metric}"]
            if pd.notna(b) and pd.notna(c) and b > 0:
                drop = round((b - c) / b, 4)
                rec[f"{metric}_relative_drop"] = drop
                if pd.isna(worst) or drop > worst:
                    worst, worst_metric = drop, metric
            else:
                rec[f"{metric}_relative_drop"] = np.nan
        if pd.isna(worst):
            status = "no out-of-sample baseline" if rid in IN_SAMPLE_RULES else "no baseline signal"
        elif worst >= d["relative_drop_stop"]:
            status = "escalate"
        elif worst >= d["relative_drop_alert"]:
            status = "review threshold"
        else:
            status = "ok"
        rec.update({"worst_metric": worst_metric, "worst_relative_drop": worst, "status": status})
        rows.append(rec)
    return pd.DataFrame(rows)


def _due(as_of: pd.Timestamp, business_days: int) -> str:
    return str(np.busday_offset(as_of.date(), business_days, roll="forward"))


def kpi_latest(weekly: pd.DataFrame, decay: pd.DataFrame, drift: pd.DataFrame,
               exposure: dict, chains: pd.DataFrame, obligations: pd.DataFrame,
               scored: dict, mon: dict, cm: dict) -> pd.DataFrame:
    w = mon["windows"]
    cur = weekly[(weekly["window"] == "current") & (~weekly["below_min_events"])]
    last_week = cur["week_start"].max()
    any_login = cur[(cur["rule_id"] == "ANY_ACTION") & (cur["surface"] == "login")]
    detected_last = int(any_login.loc[any_login["week_start"] == last_week, "fraud_hits"].sum())
    detected_mean = round(float(any_login["fraud_hits"].mean()), 1)

    acting = decay[decay["rule_id"] != "ANY_ACTION"].dropna(subset=["current_precision"])
    acting = acting[acting["current_precision"] > 0]
    worst_prec = acting.sort_values("current_precision").iloc[0]

    prog = exposure["programme_cost"]
    every = prog[prog["surface"] == "ALL"].iloc[0]
    n_days = scored["login"]["n_days"]
    n_weeks = cur["week_start"].nunique()
    cases_per_day = round(float(every["review_cases_opened"]) / n_days, 2)
    capacity = mon["thresholds"]["review_capacity_cases_per_day"]

    legit_drift = drift[drift["population"] == "legitimate_only"].sort_values("psi").iloc[-1]
    all_drift = drift[drift["population"] == "all_rows"].sort_values("psi").iloc[-1]

    pay = scored["payment"]["test"]
    disposition = {r["id"]: r["disposition"] for r in R.load_rules()["rules"]}
    preventive = [r for r in R.rule_ids("payment") if disposition[r] in ("hold", "step_up")]
    stopped = pay[pay[[f"flag_{r}" for r in preventive]].any(axis=1) & (pay["is_fraud_txn"] == 1)]
    stopped_value = round(float(stopped["amount_usd"].sum()), 2)
    pay_fraud_value = round(float(pay.loc[pay["is_fraud_txn"] == 1, "amount_usd"].sum()), 2)

    window = chains["action_window_min"].dropna()
    target = cm["assumptions"]["enforcement_latency"]["target_minutes_to_action"]
    in_scope = obligations[obligations["matrix_status"] == "evidence_generated"]
    evidenced = round(float(in_scope["evidence_present"].mean()), 4) if len(in_scope) else np.nan

    p = mon["drift"]["psi"]
    rows = [
        ("K01_detection_rate", detected_last, "count", f"weekly mean {detected_mean}",
         "ok" if detected_last >= 0.5 * detected_mean else "review",
         f"fraud login attempts hit by an actioning rule, week of {last_week}"),
        ("K02_rule_precision", float(worst_prec["current_precision"]), "ratio",
         f"decay alert at -{mon['drift']['rule_decay']['relative_drop_alert']:.0%} vs baseline",
         "escalate" if (decay["status"] == "escalate").any() else
         "review" if (decay["status"] == "review threshold").any() else "ok",
         f"lowest current precision: {worst_prec['rule_id']}"),
        ("K03_reviewer_load", cases_per_day, "cases/day", f"capacity {capacity}",
         "ok" if cases_per_day <= capacity else "over capacity",
         "deduplicated review cases opened per day, test window"),
        ("K04_feature_psi", float(legit_drift["psi"]), "index", f"severe from {p['severe_from']}",
         "severe" if legit_drift["psi"] >= p["severe_from"] else
         "moderate" if legit_drift["psi"] >= p["moderate_from"] else "stable",
         f"legitimate-only max: {legit_drift['surface']}.{legit_drift['feature']}; all-rows max "
         f"{all_drift['psi']} on {all_drift['surface']}.{all_drift['feature']}"),
        ("K05_false_positive_cost", round(float(every["wrongly_actioned_cost_usd"]) / n_weeks, 2),
         "USD/week", "trend only", "ok", "programme wrongly-actioned cost / weeks in window"),
        ("K06_loss_prevented", stopped_value, "USD", f"of {pay_fraud_value} payment fraud",
         "ok" if pay_fraud_value and stopped_value / pay_fraud_value >= 0.5 else "review",
         "fraud payments flagged by a hold or step-up rule before completion, test window"),
        ("K07_time_to_action", float(window.median()) if len(window) else np.nan, "minutes",
         f"action SLA {target}",
         "ok" if len(window) and (window >= target).mean() >= 0.9 else "breach",
         f"median signal-to-first-payout window; {(window >= target).mean():.0%} of chains leave "
         f"at least the {target}-minute SLA, so loss past it is an enforcement gap, not detection"),
        ("K08_compliance_evidence", evidenced, "ratio", "1.0",
         "ok" if evidenced == 1.0 else "gap",
         "obligations marked evidence_generated whose artifact exists and is non-empty"),
    ]
    names = {k["id"]: k["name"] for k in mon["kpis"]}
    sources = {k["id"]: k["source"] for k in mon["kpis"]}
    return pd.DataFrame([{"kpi_id": k, "name": names[k], "value": v, "unit": u, "threshold": t,
                          "status": s, "detail": d, "source": sources[k],
                          "as_of": w["current_end"]} for k, v, u, t, s, d in rows])


def alerts(decay: pd.DataFrame, drift: pd.DataFrame, kpis: pd.DataFrame, mon: dict) -> pd.DataFrame:
    """Every crossed threshold as one owned row with an SLA due date - the triage queue."""
    as_of = pd.Timestamp(mon["windows"]["current_end"])
    owner = mon["schedule"]["owner"]
    out = []
    for _, r in decay[decay["status"].isin(["review threshold", "escalate"])].iterrows():
        out.append({"source": "rule_decay", "subject": r["rule_id"], "severity": r["status"],
                    "evidence": f"{r['worst_metric']} fell {r['worst_relative_drop']:.0%} vs baseline",
                    "due_by": _due(as_of, 5 if r["status"] == "escalate" else 2)})
    legit = drift[(drift["population"] == "legitimate_only") & (drift["psi_status"] == "severe")]
    for _, r in legit.iterrows():
        out.append({"source": "feature_drift", "subject": f"{r['surface']}.{r['feature']}",
                    "severity": "severe", "evidence": f"PSI {r['psi']} on legitimate rows",
                    "due_by": _due(as_of, 2)})
    for _, r in kpis[~kpis["status"].isin(["ok", "stable"])].iterrows():
        out.append({"source": "kpi", "subject": r["kpi_id"], "severity": r["status"],
                    "evidence": f"{r['value']} {r['unit']} vs {r['threshold']}",
                    "due_by": _due(as_of, 2)})
    frame = pd.DataFrame(out, columns=["source", "subject", "severity", "evidence", "due_by"])
    frame.insert(0, "raised_on", as_of.date().isoformat())
    frame["owner"] = owner
    frame["decision"] = ""
    return frame
