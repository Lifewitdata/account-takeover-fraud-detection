"""Risk exposure in business terms: what fraud costs, what the controls cost, what a false positive does.

v1 reported an AUC, precision at k and a review-queue size, then suggested thresholds be "re-tuned
monthly". That is an accuracy report, not a risk report. It never says what the fraud actually costs,
what running the controls costs, what a false positive does to a real customer, or which number belongs
in a funding paper. This module does that arithmetic, and keeps the two kinds of input visibly apart:
every summary line carries a `basis` of measured / derived / assumption, and
outputs/exposure/assumption_ledger.csv lists every planning input with its source so a reader can
substitute their own and re-run.

One deliberate limit, stated rather than buried: no reputation cost and no customer lifetime value is
invented. A wrongly blocked customer is reported as a *count* of customers lost alongside the dollars
that are genuinely priced in config/cost_model.yaml, because a made-up LTV would move the headline
number more than any measurement in this repo does.

Units matter more than they usually get credit for here. A rule that fires on 400 login attempts from
12 accounts does not create 400 cases: an auto-block and a step-up act on each attempt, while a review
opens one case per account. Each disposition is therefore priced on its own action unit, and both the
row count and the unit count are reported so nobody has to take that choice on trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import rules as R

SURFACE_SCORE_COL = {"login": "risk_score", "payment": "txn_risk_score"}
SURFACE_LABEL_COL = {"login": "is_fraud_attempt", "payment": "is_fraud_txn"}

# Does the disposition act per attempt, or open one case per account?
UNIT_OF_ACTION = {"auto_block": "row", "step_up": "row", "hold": "row",
                  "review": "account_on_login", "monitor": "row"}

# What a false positive costs, indexed by cost_model.yaml's `actions.<disposition>.wrongly_actioned`.
# A rule that only queues a case or records a signal does nothing to a legitimate customer, and saying
# so out loud is the point: it is why "just raise the threshold" is not automatically the cheap option.
WRONGED_COMPONENTS = {
    "hard_block": ("appeal", "support", "abandon", "churn_count"),
    "delayed_funds": ("appeal", "delayed_value"),
    "friction": ("abandon",),
    "queue_only": (),
    "none": (),
}
PREVENTIVE = {"auto_block", "hold", "step_up"}
LATENCY_POINTS_MIN = [0, 15, 30, 60, 120, 240, 480, 1440]


def _cost(minutes: float, rate_per_hour: float) -> float:
    return round(minutes / 60.0 * rate_per_hour, 2)


def assumption_ledger(cm: dict) -> pd.DataFrame:
    """Flatten every planning input into one table, so no downstream number is untraceable."""
    rows = []
    for section, body in cm["assumptions"].items():
        for key, val in body.items():
            rows.append({"path": f"assumptions.{section}.{key}", "value": val,
                         "kind": "assumption", "note": "planning input, not observed in data/"})
    for disp, body in cm["actions"].items():
        for key, val in body.items():
            if key == "description":
                continue
            rows.append({"path": f"actions.{disp}.{key}", "value": val, "kind": "config",
                         "note": "disposition pricing, keyed to detection_rules.yaml"})
    rows.append({"path": "decision.capacity_share_of_queue",
                 "value": cm["decision"]["capacity_share_of_queue"], "kind": "config",
                 "note": "the only judgement threshold in the verdict column"})
    return pd.DataFrame(rows)


def _account_fraud_value(txns: pd.DataFrame) -> pd.Series:
    """Per-account fraudulent value, used when a login-surface rule interrupts an account, not a payment."""
    fraud = txns[txns["is_fraud_txn"] == 1]
    return fraud.groupby("user_id")["amount_usd"].sum()


def rule_costs(scored: dict, cm: dict, mon: dict, acct_value: pd.Series) -> pd.DataFrame:
    """Price every registry rule on the held-out window: staff time, customer harm, value touched.

    Two things this deliberately does *not* charge to a rule, because rules overlap and doing so would
    multiply the same case across them. Dispute handling and account remediation are consequences of
    confirmed fraud, not of which rule surfaced it, so they live once in loss_model(). And the value a
    rule "recovers" is labelled *touched*: R01, R02, R09 and R10 all stand on the same accounts, so the
    touched column is not additive and build() deduplicates it for the programme-level line.
    """
    rv, fr, lo = (cm["assumptions"][k] for k in ("review", "friction", "loss"))
    act, reg = cm["actions"], R.load_rules()["rules"]
    disposition = {r["id"]: r["disposition"] for r in reg}
    capacity = mon["thresholds"]["review_capacity_cases_per_day"]
    queue_share = cm["decision"]["capacity_share_of_queue"]
    rate = rv["cost_per_hour_usd"]

    rows = []
    for surface, score_col in SURFACE_SCORE_COL.items():
        f = scored[surface]["test"]
        n_days = scored[surface]["n_days"]
        label_col = SURFACE_LABEL_COL[surface]
        is_fraud = f[label_col].to_numpy() == 1
        for rid in R.rule_ids(surface):
            disp = disposition[rid]
            hit = f[f"flag_{rid}"].to_numpy().astype(bool)
            unit = UNIT_OF_ACTION[disp]
            if unit == "account_on_login":
                keys = f["user_id"]
                mask = pd.Series(hit, index=f.index).groupby(keys).max().astype(bool)
                fraud_mask = pd.Series(is_fraud, index=f.index).groupby(keys).max().astype(bool)
                alerts = int(mask.sum())
                fraud_units = int((mask & fraud_mask).sum())
                legit_units = alerts - fraud_units
            else:
                alerts = int(hit.sum())
                fraud_units = int((hit & is_fraud).sum())
                legit_units = alerts - fraud_units
            comps = WRONGED_COMPONENTS[act[disp]["wrongly_actioned"]]

            if surface == "payment":
                gross = round(float(f.loc[hit & is_fraud, "amount_usd"].sum()), 2)
                delayed = round(float(f.loc[hit & ~is_fraud, "amount_usd"].sum()), 2)
                value_basis = "amount on the flagged fraudulent rows"
            else:
                flagged_accounts = f.loc[hit & is_fraud, "user_id"].unique()
                gross = round(float(acct_value.reindex(flagged_accounts).fillna(0.0).sum()), 2)
                delayed = float("nan")
                value_basis = "fraud value of the accounts whose login rows were flagged"

            analyst = _cost(alerts * act[disp]["analyst_minutes"], rate)
            appeal = _cost(legit_units * rv["appeal_rate_on_false_positive"] * rv["appeal_minutes"],
                           rate) if "appeal" in comps else 0.0
            support = _cost(legit_units * fr["hard_block_support_minutes"], rate) \
                if "support" in comps else 0.0
            abandon = round(legit_units * fr["step_up_abandon_rate"] * fr["session_value_usd"], 2) \
                if "abandon" in comps else 0.0
            churned = round(legit_units * fr["hard_block_churn_rate"], 1) \
                if "churn_count" in comps else 0.0
            wrongly = round(appeal + support + abandon, 2)
            total_cost = round(analyst + wrongly, 2)
            touched_recovered = round(gross * lo["recovery_rate_on_confirmed_fraud"], 2)
            # A control that stops the payout is worth the payout; one that only queues a case is worth
            # what recovery claws back afterwards. Upper bound for step-up, which an attacker may pass.
            preventive = disp in PREVENTIVE
            benefit = gross if preventive else (0.0 if disp == "monitor" else touched_recovered)
            per_day = round(alerts / n_days, 2) if n_days else float("nan")

            if fraud_units == 0:
                verdict = "no measured benefit"
            elif disp == "monitor":
                verdict = "signal only, no action taken"
            elif total_cost > benefit:
                verdict = "costs more than it saves"
            elif disp == "review" and per_day > capacity * queue_share:
                verdict = "consumes too much queue capacity"
            else:
                verdict = "keep"

            rows.append({
                "rule_id": rid, "surface": surface, "disposition": disp,
                "action_unit": unit, "period": "test_window_Oct-Dec",
                "alerts_rows": int(hit.sum()), "fraud_rows": int((hit & is_fraud).sum()),
                "legit_rows": int((hit & ~is_fraud).sum()),
                "action_units": alerts, "fraud_units": fraud_units, "legit_units": legit_units,
                "precision": round(fraud_units / alerts, 4) if alerts else float("nan"),
                "recall_share": round(int((hit & is_fraud).sum()) / int(is_fraud.sum()), 4)
                                 if is_fraud.sum() else float("nan"),
                "alerts_per_day": per_day,
                "analyst_cost_usd": analyst, "appeal_cost_usd": appeal,
                "support_cost_usd": support, "abandoned_revenue_usd": abandon,
                "wrongly_actioned_cost_usd": wrongly, "total_control_cost_usd": total_cost,
                "customers_churned_count": churned, "delayed_legit_value_usd": delayed,
                "fraud_value_touched_usd": gross, "value_basis": value_basis,
                "recovered_value_touched_usd": touched_recovered,
                "benefit_usd": round(benefit, 2),
                "benefit_basis": ("payout prevented" if preventive else
                                  "none, no action" if disp == "monitor" else "recovered after review"),
                "net_position_usd": round(total_cost - benefit, 2),
                "cost_per_fraud_case_usd": round(total_cost / fraud_units, 2)
                                           if fraud_units else float("nan"),
                "wronged_components": "|".join(comps) if comps else "none",
                "verdict": verdict,
            })
    out = pd.DataFrame(rows).sort_values(["surface", "rule_id"]).reset_index(drop=True)
    out["costing"] = ("standalone: sums across these rows overstate the queue, because one account "
                      "opened by two review rules is still one case - see programme_cost")
    return out


# How badly a wrongly-fired disposition treats a legitimate customer, worst first. A row hit by both
# an auto-block and a step-up is hard-blocked, so the harm is charged once, at the top of this order.
HARM_PRIORITY = ["hard_block", "delayed_funds", "friction", "queue_only", "none"]


def programme_cost(scored: dict, cm: dict, acct_value: pd.Series) -> pd.DataFrame:
    """What the whole rule set costs when it runs together, deduplicated the way a queue actually does.

    The per-rule table is a standalone view; this is the additive one, and the difference is not small.
    Four login rules stand on the same 353 accounts, so charging each of them a 12-minute case would
    bill the programme for work it does once. Here staff time is charged at the most expensive
    disposition that fired on each row, review at one case per account however many rules touched it,
    and customer harm once per affected row at the worst action taken on it.
    """
    rv, fr, lo = (cm["assumptions"][k] for k in ("review", "friction", "loss"))
    act, reg = cm["actions"], R.load_rules()["rules"]
    disposition = {r["id"]: r["disposition"] for r in reg}
    rate = rv["cost_per_hour_usd"]
    no_harm_rank = len(HARM_PRIORITY)

    rows = []
    for surface in SURFACE_SCORE_COL:
        f = scored[surface]["test"]
        is_fraud = f[SURFACE_LABEL_COL[surface]].to_numpy() == 1
        n = len(f)
        row_minutes = np.zeros(n)
        harm_rank = np.full(n, no_harm_rank)
        account_case = np.zeros(n, dtype=bool)
        any_hit = np.zeros(n, dtype=bool)
        for rid in R.rule_ids(surface):
            disp = disposition[rid]
            fired = f[f"flag_{rid}"].to_numpy().astype(bool)
            any_hit |= fired
            rank = HARM_PRIORITY.index(act[disp]["wrongly_actioned"])
            harm_rank = np.minimum(harm_rank, np.where(fired, rank, no_harm_rank))
            if UNIT_OF_ACTION[disp] == "account_on_login":
                account_case |= fired
            else:
                row_minutes = np.maximum(row_minutes, fired * act[disp]["analyst_minutes"])

        cases = int(pd.Series(account_case, index=f.index).groupby(f["user_id"]).max().sum())
        review_minutes = cases * act["review"]["analyst_minutes"]
        legit_rank = harm_rank[~is_fraud & any_hit]
        counts = {h: int((legit_rank == i).sum()) for i, h in enumerate(HARM_PRIORITY)}
        appeal_units = counts["hard_block"] + counts["delayed_funds"]
        wrongly = round(
            _cost(appeal_units * rv["appeal_rate_on_false_positive"] * rv["appeal_minutes"], rate)
            + _cost(counts["hard_block"] * fr["hard_block_support_minutes"], rate)
            + (counts["hard_block"] + counts["friction"]) * fr["step_up_abandon_rate"]
            * fr["session_value_usd"], 2)
        if surface == "payment":
            touched_value = float(f.loc[any_hit & is_fraud, "amount_usd"].sum())
        else:
            touched_accounts = f.loc[any_hit & is_fraud, "user_id"].unique()
            touched_value = float(acct_value.reindex(touched_accounts).fillna(0.0).sum())
        analyst = round(_cost(float(row_minutes.sum()) + review_minutes, rate), 2)
        rows.append({
            "surface": surface, "period": "test_window_Oct-Dec",
            "rows_scored": n, "rows_flagged_by_any_rule": int(any_hit.sum()),
            "review_cases_opened": cases, "review_cases_cost_usd": _cost(review_minutes, rate),
            "row_action_analyst_cost_usd": analyst,
            "fraud_rows_touched": int((any_hit & is_fraud).sum()),
            "legit_rows_hard_blocked": counts["hard_block"],
            "legit_rows_held": counts["delayed_funds"],
            "legit_rows_stepped_up": counts["friction"],
            "legit_rows_queued_only": counts["queue_only"],
            "wrongly_actioned_cost_usd": wrongly,
            "total_control_cost_usd": round(analyst + wrongly, 2),
            "fraud_value_touched_usd": round(touched_value, 2),
            "recovered_value_usd": round(touched_value * lo["recovery_rate_on_confirmed_fraud"], 2),
            "customers_churned_count": round(counts["hard_block"] * fr["hard_block_churn_rate"], 1),
        })
    frame = pd.DataFrame(rows)
    total = {"surface": "ALL", "period": frame["period"].iloc[0]}
    for col in frame.columns[2:]:
        total[col] = round(float(frame[col].sum()), 2)
    return pd.concat([frame, pd.DataFrame([total])], ignore_index=True)


def threshold_economics(scored: dict, cm: dict, mon: dict, acct_value: pd.Series) -> pd.DataFrame:
    """Cost the model's action threshold on each surface rather than only scoring it."""
    rv, lo = cm["assumptions"]["review"], cm["assumptions"]["loss"]
    capacity = mon["thresholds"]["review_capacity_cases_per_day"]
    rows = []
    for surface, score_col in SURFACE_SCORE_COL.items():
        f = scored[surface]["test"]
        n_days = scored[surface]["n_days"]
        label_col = SURFACE_LABEL_COL[surface]
        scores, is_fraud = f[score_col].to_numpy(), f[label_col].to_numpy() == 1
        accounts = f["user_id"].to_numpy()
        # Login rows carry no amount: the value at stake behind one is the account's fraud value, so
        # the two surfaces price a threshold off different columns and say which.
        amounts = f["amount_usd"].to_numpy() if surface == "payment" else None
        total_fraud_value = (round(float(amounts[is_fraud].sum()), 2) if surface == "payment"
                             else round(float(acct_value.sum()), 2))
        for thr in mon["thresholds"]["grid"]:
            pred = scores >= thr
            fraud_rows = int((pred & is_fraud).sum())
            legit_rows = int((pred & ~is_fraud).sum())
            alerts = fraud_rows + legit_rows
            flagged_accounts = pd.unique(accounts[pred])
            accounts_in_alerts = int(len(flagged_accounts))
            caught_accounts = pd.unique(accounts[pred & is_fraud])
            missed_accounts = pd.unique(accounts[(~pred) & is_fraud])
            if surface == "payment":
                caught = round(float(amounts[pred & is_fraud].sum()), 2)
                missed = round(float(amounts[(~pred) & is_fraud].sum()), 2)
            else:
                caught = round(float(acct_value.reindex(caught_accounts).fillna(0.0).sum()), 2)
                missed = round(float(acct_value.reindex(missed_accounts).fillna(0.0).sum()), 2)
            accounts_per_day = round(accounts_in_alerts / n_days, 2) if n_days else float("nan")
            analyst = _cost(alerts * rv["minutes_per_case"], rv["cost_per_hour_usd"])
            wrongly = _cost(legit_rows * rv["appeal_rate_on_false_positive"] * rv["appeal_minutes"],
                            rv["cost_per_hour_usd"])
            recovered = round(caught * lo["recovery_rate_on_confirmed_fraud"], 2)
            rows.append({
                "surface": surface, "threshold": thr, "period": "test_window_Oct-Dec",
                "alerts_rows": alerts, "fraud_rows": fraud_rows, "legit_rows": legit_rows,
                "accounts_in_alerts": accounts_in_alerts,
                "precision_on_rows": round(fraud_rows / alerts, 4) if alerts else 0.0,
                "recall_on_rows": round(fraud_rows / int(is_fraud.sum()), 4)
                                  if is_fraud.sum() else 0.0,
                "alerts_per_day": round(alerts / n_days, 2) if n_days else float("nan"),
                "accounts_per_day": accounts_per_day,
                "queue_utilisation": round(accounts_per_day / capacity, 3) if capacity else float("nan"),
                "within_capacity": bool(accounts_per_day <= capacity),
                "fraud_value_caught_usd": caught, "fraud_value_missed_usd": missed,
                "fraud_value_total_usd": total_fraud_value,
                "recovered_value_usd": recovered,
                "unrecovered_loss_usd": round(missed * (1 - lo["recovery_rate_on_confirmed_fraud"]), 2),
                "analyst_cost_usd": analyst, "appeal_cost_usd": wrongly,
                "net_exposure_usd": round(analyst + wrongly
                                          + missed * (1 - lo["recovery_rate_on_confirmed_fraud"])
                                          - recovered, 2),
                "is_default_threshold": thr == mon["thresholds"]["default"],
            })
    return pd.DataFrame(rows)


def enforcement_latency(rca_chains: pd.DataFrame, fraud_txns: pd.DataFrame, cm: dict) -> pd.DataFrame:
    """What value a control could still stop if it acted within N minutes of its own first signal.

    Measured, not modelled, and it needs no cost assumption: every ATO incident has a first qualifying
    signal and a sequence of payouts, so a payout is stoppable at N when it moved more than N minutes
    after that signal. This is the table that funds faster triage.

    Two denominators are carried on every row because they answer different questions. A payout being
    stoppable is a property of the takeover chains, so `share_of_chain_fraud_value` says how much of
    the money those 120 accounts took was still in the account when a rule had already fired - the
    operational number. `share_of_all_fraud_value` puts it against the whole ledger, which is smaller
    and is the one a board sees.
    """
    signals = rca_chains[["user_id", "first_signal_ts"]].dropna(subset=["first_signal_ts"])
    joined = fraud_txns.merge(signals, on="user_id", how="inner")
    joined = joined.assign(
        min_from_signal=(joined["timestamp"] - joined["first_signal_ts"]).dt.total_seconds() / 60.0)
    chain_total = round(float(joined["amount_usd"].sum()), 2)
    all_total = round(float(fraud_txns["amount_usd"].sum()), 2)
    target = cm["assumptions"]["enforcement_latency"]["target_minutes_to_action"]

    rows, previous = [], None
    for point in LATENCY_POINTS_MIN:
        stoppable = joined[joined["min_from_signal"] > point]
        value = round(float(stoppable["amount_usd"].sum()), 2)
        if previous is None:
            marginal = float("nan")
        else:
            steps = (point - previous[0]) / 15.0
            marginal = round((previous[1] - value) / steps, 2)
        previous = (point, value)
        rows.append({
            "act_within_minutes": point,
            "incidents_with_a_stoppable_payout": int(stoppable["user_id"].nunique()),
            "payouts_stoppable": int(len(stoppable)),
            "value_stoppable_usd": value,
            "share_of_chain_fraud_value": round(value / chain_total, 4) if chain_total else float("nan"),
            "share_of_all_fraud_value": round(value / all_total, 4) if all_total else float("nan"),
            "chain_fraud_value_usd": chain_total, "all_fraud_value_usd": all_total,
            "value_lost_per_extra_15min_of_delay_usd": marginal,
            "within_current_target": bool(point <= target),
        })
    return pd.DataFrame(rows)


def loss_model(rca_chains: pd.DataFrame, test_txns: pd.DataFrame, fraud_txns: pd.DataFrame,
               all_txns: pd.DataFrame, cm: dict) -> pd.DataFrame:
    """The realised cost of the fraud, and how much of it was still leaving after a signal."""
    lo, rv = cm["assumptions"]["loss"], cm["assumptions"]["review"]
    gross_all = round(float(fraud_txns["amount_usd"].sum()), 2)
    gross_test = round(float(test_txns.loc[test_txns["is_fraud_txn"] == 1, "amount_usd"].sum()), 2)
    after_signal = round(float(rca_chains["preventable_value_usd"].sum()), 2)
    recovery = lo["recovery_rate_on_confirmed_fraud"]
    n_incidents = int(len(rca_chains))
    n_accounts = int(fraud_txns["user_id"].nunique())

    net_loss = round(gross_all * (1 - recovery), 2)
    dispute = round(n_accounts * lo["dispute_cost_per_case_usd"], 2)
    remediation = _cost(n_incidents * lo["account_remediation_minutes"], rv["cost_per_hour_usd"])
    non_financial = lo["non_financial_harm_usd_per_incident"] * n_incidents
    total_cost = round(net_loss + dispute + remediation + non_financial, 2)

    return pd.DataFrame([
        {"line": "fraud_value_whole_period", "metric": "Labelled fraudulent transaction value, "
                "Jan-Dec 2025", "value": gross_all, "unit": "USD", "basis": "measured",
         "source_or_assumption": "data/transactions.csv where is_fraud_txn = 1"},
        {"line": "fraud_value_test_window", "metric": "Labelled fraudulent value inside the held-out "
                "Oct-Dec window", "value": gross_test, "unit": "USD", "basis": "measured",
         "source_or_assumption": "scored payment test frame"},
        {"line": "accounts_affected", "metric": "Accounts whose transactions are labelled fraud",
         "value": n_accounts, "unit": "count", "basis": "measured",
         "source_or_assumption": "distinct user_id on labelled rows"},
        {"line": "value_still_moving_after_first_signal", "metric": "Value that left an ATO account "
                "after a registry rule had already fired on it", "value": after_signal, "unit": "USD",
         "basis": "measured", "source_or_assumption": "incident preventable_value_usd"},
        {"line": "expected_recovery", "metric": "Value a recovery process claws back",
         "value": round(gross_all * recovery, 2), "unit": "USD", "basis": "derived",
         "source_or_assumption": "assumptions.loss.recovery_rate_on_confirmed_fraud"},
        {"line": "net_financial_loss", "metric": "Fraud value left after recovery", "value": net_loss,
         "unit": "USD", "basis": "derived", "source_or_assumption": "gross x (1 - recovery rate)"},
        {"line": "dispute_handling_cost", "metric": "Chargeback handling on affected accounts",
         "value": dispute, "unit": "USD", "basis": "derived",
         "source_or_assumption": "one dispute per account x dispute_cost_per_case_usd"},
        {"line": "account_remediation_cost", "metric": "Re-verification and document re-collection",
         "value": remediation, "unit": "USD", "basis": "derived",
         "source_or_assumption": "incidents x account_remediation_minutes at analyst rate"},
        {"line": "non_financial_harm", "metric": "Reputation and distress cost",
         "value": round(non_financial, 2), "unit": "USD", "basis": "assumption",
         "source_or_assumption": "held at zero deliberately; no invented reputation figure"},
        {"line": "total_fraud_cost", "metric": "Cost of the fraud itself, before any detection spend",
         "value": total_cost, "unit": "USD", "basis": "derived",
         "source_or_assumption": "net loss + dispute + remediation + non-financial"},
        {"line": "cost_per_confirmed_incident", "metric": "Total fraud cost per confirmed ATO incident",
         "value": round(total_cost / n_incidents, 2) if n_incidents else float("nan"),
         "unit": "USD/incident", "basis": "derived", "source_or_assumption": f"{n_incidents} incidents"},
        {"line": "median_fraud_case", "metric": "Median fraudulent transaction",
         "value": round(float(fraud_txns["amount_usd"].median()), 2), "unit": "USD",
         "basis": "measured", "source_or_assumption": "labelled rows"},
        {"line": "legit_txn_value_in_period", "metric": "Total legitimate payment volume, the "
                "denominator any hold or block puts at risk of delay", "value": round(float(
                    all_txns.loc[all_txns["is_fraud_txn"] == 0, "amount_usd"].sum()), 2),
         "unit": "USD", "basis": "measured", "source_or_assumption": "unlabelled rows in the period"},
    ])


def build(scored: dict, rca: dict) -> dict:
    """Assemble every exposure artifact, keyed by the filename the pipeline writes them to."""
    cm, mon = C.load_yaml("cost_model.yaml"), C.load_yaml("monitoring.yaml")
    txns = pd.concat([scored["payment"]["train"], scored["payment"]["test"]], ignore_index=True)
    fraud_txns = txns[txns["is_fraud_txn"] == 1]
    acct_value = _account_fraud_value(scored["payment"]["test"])

    control = rule_costs(scored, cm, mon, acct_value)
    programme = programme_cost(scored, cm, acct_value)
    thresholds = threshold_economics(scored, cm, mon, acct_value)
    latency = enforcement_latency(rca["chains"], fraud_txns, cm)
    loss = loss_model(rca["chains"], scored["payment"]["test"], fraud_txns, txns, cm)

    every = programme[programme["surface"] == "ALL"].iloc[0]
    spend = float(every["total_control_cost_usd"])
    wronged = float(every["wrongly_actioned_cost_usd"])
    churned = float(every["customers_churned_count"])
    recovered = float(every["recovered_value_usd"])
    standalone = round(float(control["total_control_cost_usd"].sum()), 2)
    headline = pd.DataFrame([
        {"line": "programme_control_cost_deduplicated", "metric": "Staff and customer cost of running "
                "the whole rule set across the test window, one case per account", "value": spend,
         "unit": "USD", "basis": "derived", "source_or_assumption": "cost_model.yaml actions x alerts"},
        {"line": "programme_control_cost_naive_sum", "metric": "The same cost if each rule were billed "
                "standalone, shown because it is what a per-rule table invites a reader to do",
         "value": standalone, "unit": "USD", "basis": "derived",
         "source_or_assumption": "sum of control_cost.csv, overstates by double-billing shared cases"},
        {"line": "review_cases_opened", "metric": "Distinct accounts a review rule sent to the queue",
         "value": int(every["review_cases_opened"]), "unit": "count", "basis": "measured",
         "source_or_assumption": "union of review-rule hits, deduplicated by account"},
        {"line": "wrongly_actioned_cost", "metric": "Cost attributable to false positives alone",
         "value": wronged, "unit": "USD", "basis": "derived",
         "source_or_assumption": "appeal + support + abandoned revenue, charged once per row"},
        {"line": "customers_wrongly_blocked_churned", "metric": "Legitimate customers estimated to "
                "leave after a wrong hard block", "value": churned, "unit": "count",
         "basis": "assumption", "source_or_assumption": "hard_block_churn_rate, priced in people "
                "not dollars"},
        {"line": "recovered_value", "metric": "Fraud value the rule set touched and recovery brought "
                "back", "value": recovered, "unit": "USD", "basis": "derived",
         "source_or_assumption": "deduplicated touched value x recovery rate"},
        {"line": "control_cost_per_dollar_recovered", "metric": "Programme cost per dollar recovered",
         "value": round(spend / recovered, 2) if recovered else float("nan"), "unit": "ratio",
         "basis": "derived", "source_or_assumption": "programme control cost / recovered value"},
        {"line": "rules_without_measured_benefit", "metric": "Registry rules that caught no labelled "
                "fraud in the window", "value": int((control["fraud_units"] == 0).sum()),
         "unit": "count", "basis": "measured", "source_or_assumption": "verdict column"},
        {"line": "fraud_value_vs_control_cost", "metric": "Test-window fraud value divided by what the "
                "controls cost to catch it", "value": round(
                    float(loss.loc[loss["line"] == "fraud_value_test_window", "value"].iloc[0]) / spend,
                    2) if spend else float("nan"),
         "unit": "ratio", "basis": "derived",
         "source_or_assumption": "the funding question, answered with both numbers visible"},
    ])

    preventable = rca["root_cause_by_stage"][
        ["root_cause", "root_cause_name", "stage", "incidents", "accounts",
         "fraud_value_usd", "preventable_value_usd", "fix", "owner"]].copy()
    preventable["enforcement_target_minutes"] = \
        cm["assumptions"]["enforcement_latency"]["target_minutes_to_action"]
    preventable["basis"] = ("fraud_value_usd is measured and exclusive to the cause; "
                            "preventable_value_usd assumes closing the gap stops the listed payouts")

    return {"assumption_ledger": assumption_ledger(cm), "control_cost": control,
            "programme_cost": programme, "threshold_economics": thresholds,
            "enforcement_latency": latency, "loss_model": loss,
            "exposure_summary": pd.concat([loss, headline], ignore_index=True),
            "preventable_loss_by_control": preventable}
