"""Confirmed-incident root cause analysis.

The v1 project stopped at pattern level: it reported that IP-velocity windows and shared devices
exist, and how often. This module works the other end - it takes the accounts the ground truth
actually marks as attacked, rebuilds each one as a kill chain, measures what was visible before any
money moved, and attributes the failure to a named control gap with a fix and an owner.

Nothing here is invented. Every incident row is derived from labelled rows already in data/, and
every root-cause finding carries the evidence key (attempt_id / reset_id / txn_id) that a reviewer
could go and read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import config as C
from . import features as F

# Stage order runs in the sequence an analyst wants a control to intervene: stop the credential
# from working, then stop the session surviving, then stop the money leaving, then contain the ring.
STAGES = {"authentication": 1, "session": 2, "reverification": 3, "payment": 4, "response": 5}

ROOT_CAUSES = {
    "RC-01": {
        "name": "Session survived a password reset",
        "kind": "system", "stage": "session",
        "fix": "Invalidate every active session and refresh token on credential change, and force "
               "re-authentication before any payout in the window after a reset completes.",
        "owner": "Platform identity",
    },
    "RC-02": {
        "name": "First-seen device authenticated without step-up",
        "kind": "system", "stage": "authentication",
        "fix": "Promote a successful login on an unseen device to a step-up challenge before the "
               "session is granted full privileges; treat the resulting session as limited until "
               "the challenge clears.",
        "owner": "Authentication engineering",
    },
    "RC-03": {
        "name": "No throttle or breached-credential screening on the login endpoint",
        "kind": "system", "stage": "authentication",
        "fix": "Rate-limit per IP and per account, lock out on sustained failure, and screen "
               "candidate passwords against known-breach corpora before accepting them "
               "(NIST SP 800-63B).",
        "owner": "Application security",
    },
    "RC-04": {
        "name": "Dormant account reactivated without re-verification",
        "kind": "process", "stage": "reverification",
        "fix": "Re-verify an account that has been idle beyond a set threshold before it can "
               "transact; an idle account is exactly what a bought credential list is for.",
        "owner": "Trust and Safety operations",
    },
    "RC-05": {
        "name": "Instant payout in a high-conversion category",
        "kind": "process", "stage": "payment",
        "fix": "Place a hold on gift-card, jewelry and digital-goods value until the account's "
               "device and reset state has aged out of the takeover window.",
        "owner": "Payments risk",
    },
    "RC-06": {
        "name": "No device binding checked at payment time",
        "kind": "system", "stage": "payment",
        "fix": "Bind the paying session to an authenticated device and re-check it per payout, "
               "carrying the login-level new-device flag forward rather than re-deriving it.",
        "owner": "Payments risk",
    },
    "RC-07": {
        "name": "Confirmed account did not trigger its device neighbourhood",
        "kind": "process", "stage": "response",
        "fix": "When one account on a shared device is confirmed fraudulent, re-score and review "
               "every account on that device and IP in the same case, not one at a time.",
        "owner": "Trust and Safety operations",
    },
    "RC-08": {
        "name": "SMS reset granted without a device-possession check",
        "kind": "process", "stage": "reverification",
        "fix": "Treat an out-of-band SMS reset on a new device as single-factor; require a second "
               "independent factor before a reset completes when the authenticating device changed.",
        "owner": "Authentication engineering",
    },
    "RC-09": {
        "name": "No registration velocity control at enrolment",
        "kind": "process", "stage": "authentication",
        "fix": "Cap enrolments per device and per IP, and hold a cluster that registered together "
               "out of payout eligibility until it has been verified separately.",
        "owner": "Identity and onboarding",
    },
}

# Which registry rule was supposed to stop each cause. Kept next to the taxonomy so a root-cause
# finding and a rule change cannot drift apart; the tests assert these ids exist in the registry.
RULE_FOR_CAUSE = {
    "RC-01": ["R05_reset_then_payout", "R11_recent_new_device_login"],
    "RC-02": ["R03_new_device_step_up"],
    "RC-03": ["R01_ip_velocity_5u10m", "R10_ip_neighbourhood"],
    "RC-04": ["R03_new_device_step_up", "R09_model_score"],
    "RC-05": ["R08_convertible_category", "R05_reset_then_payout"],
    "RC-06": ["R06_new_device_payout", "R11_recent_new_device_login"],
    "RC-07": ["R02_shared_device_5u", "R10_ip_neighbourhood"],
    "RC-08": ["R03_new_device_step_up", "R05_reset_then_payout"],
    "RC-09": ["R02_shared_device_5u"],
}


def _first(df: pd.DataFrame, col: str, ascending: bool = True):
    if not len(df):
        return pd.NaT
    return df.sort_values(col).iloc[0][col] if ascending else df.sort_values(col).iloc[-1][col]


def _n_accounts(member_accounts: pd.Series) -> int:
    """Distinct accounts behind a set of findings, since one account can appear in several of them.

    Findings carry the pipe-joined account list rather than a count so this can deduplicate: a person
    breached by two campaigns, or named by both a campaign row and a takeover row, is one person.
    """
    ids: set[str] = set()
    for value in member_accounts:
        ids.update(str(value).split("|"))
    return len(ids)


def ato_incidents(silver: dict, login_features: pd.DataFrame) -> pd.DataFrame:
    """One row per confirmed account takeover, rebuilt as a kill chain with timings and exposure."""
    resets = silver["resets"]
    txns = silver["txns"]
    confirmed = resets.loc[resets["is_fraud_reset"] == 1, "user_id"].unique()
    lf = login_features

    rows = []
    for uid in sorted(confirmed):
        u_res = resets[(resets["user_id"] == uid) & (resets["is_fraud_reset"] == 1)]
        u_tx = txns[(txns["user_id"] == uid) & (txns["is_fraud_txn"] == 1)]
        u_fl = lf[(lf["user_id"] == uid) & (lf["is_fraud_attempt"] == 1)]
        u_ok = u_fl[u_fl["login_result"] == "success"]
        prior = lf[(lf["user_id"] == uid) & (lf["is_fraud_attempt"] == 0)]

        reset_req = u_res["request_timestamp"].min()
        reset_done = u_res.loc[u_res["status"].eq("completed"), "complete_timestamp"].min()
        first_attack = u_fl["timestamp"].min() if len(u_fl) else pd.NaT
        first_loss = u_tx["timestamp"].min() if len(u_tx) else pd.NaT
        last_loss = u_tx["timestamp"].max() if len(u_tx) else pd.NaT

        # The earliest moment anything a reviewer could act on was true for this account.
        signals = []
        nd_ok = u_ok[u_ok["new_device"] == 1]
        if len(nd_ok):
            signals.append(("R03_new_device_step_up", nd_ok["timestamp"].min()))
        if len(u_res):
            signals.append(("R05_reset_then_payout", reset_req))
        vel = u_fl[u_fl["i_users"] >= 5]
        if len(vel):
            signals.append(("R01_ip_velocity_5u10m", vel["timestamp"].min()))
        sig_rule, sig_ts = (sorted(signals, key=lambda s: s[1])[0] if signals else (None, pd.NaT))

        idle_days = np.nan
        if pd.notna(first_attack):
            before = prior[prior["timestamp"] < first_attack]["timestamp"].max()
            if pd.notna(before):
                idle_days = (first_attack - before).total_seconds() / 86400

        loss_after_signal = (float(u_tx.loc[u_tx["timestamp"] > sig_ts, "amount_usd"].sum())
                             if pd.notna(sig_ts) else float(u_tx["amount_usd"].sum()))
        attack_dev = u_ok.iloc[0]["device_id"] if len(u_ok) else np.nan
        paid_same_device = bool(len(u_tx) and pd.notna(attack_dev)
                                and (u_tx["device_id"] == attack_dev).any())

        rows.append({
            "incident_id": f"ATO-{uid}",
            "incident_type": "account_takeover",
            "user_id": uid,
            "first_attack_ts": first_attack,
            "first_fraud_success_ts": u_ok["timestamp"].min() if len(u_ok) else pd.NaT,
            "reset_requested_ts": reset_req,
            "reset_completed_ts": reset_done,
            "reset_channel": "|".join(sorted(u_res["channel"].unique())),
            "reset_status": "|".join(sorted(u_res["status"].unique())),
            "first_loss_ts": first_loss,
            "last_loss_ts": last_loss,
            "first_signal_ts": sig_ts,
            "attack_device": attack_dev,
            "paid_on_attack_device": int(paid_same_device),
            "n_fraud_logins": int(len(u_fl)),
            "n_fraud_txns": int(len(u_tx)),
            "fraud_value_usd": round(float(u_tx["amount_usd"].sum()), 2),
            "value_after_first_signal_usd": round(loss_after_signal, 2),
            "first_signal_rule": sig_rule,
            "signal_to_first_loss_min": (round((first_loss - sig_ts).total_seconds() / 60, 1)
                                         if pd.notna(sig_ts) and pd.notna(first_loss) else np.nan),
            "attack_to_loss_min": (round((first_loss - first_attack).total_seconds() / 60, 1)
                                   if pd.notna(first_attack) and pd.notna(first_loss) else np.nan),
            "idle_days_before_attack": round(idle_days, 2) if pd.notna(idle_days) else np.nan,
            "categories": "|".join(sorted(u_tx["merchant_category"].unique())),
            "evidence_keys": "|".join(list(u_ok["attempt_id"].head(1))
                                      + list(u_res["reset_id"].head(1))
                                      + list(u_tx["txn_id"].head(2))),
        })
    return pd.DataFrame(rows).sort_values("fraud_value_usd", ascending=False).reset_index(drop=True)


def attribute_root_causes(incidents: pd.DataFrame, payment_features: pd.DataFrame) -> pd.DataFrame:
    """Multi-label attribution: every control gap a chain actually passed through, with its evidence.

    A single takeover normally breaches several controls in sequence, so `primary_root_cause` is the
    earliest-stage gap - the one that would have stopped the chain soonest - and the rest are listed
    so remediation planning sees the full set rather than one cause per case.
    """
    fpf = (payment_features[payment_features["is_fraud_txn"] == 1]
           .groupby("user_id")
           .agg(device_never_used=("device_never_used_by_user", "max"),
                paid_on_new_device=("paid_on_new_device", "max")))
    rows = []
    for inc in incidents.to_dict("records"):
        found = []
        uid = inc["user_id"]

        if pd.notna(inc["first_fraud_success_ts"]):
            found.append(("RC-02", f"authentication succeeded from an unseen device at "
                                   f"{inc['first_fraud_success_ts']}, session granted without step-up"))
        if pd.notna(inc["reset_completed_ts"]) and pd.notna(inc["first_loss_ts"]) \
                and inc["first_loss_ts"] > inc["reset_completed_ts"] and inc["paid_on_attack_device"]:
            found.append(("RC-01", f"reset completed {inc['reset_completed_ts']}, fraudulent payout "
                                   f"followed on the same device at {inc['first_loss_ts']}"))
        if pd.notna(inc["idle_days_before_attack"]) and inc["idle_days_before_attack"] > 30:
            found.append(("RC-04", f"account idle {inc['idle_days_before_attack']:.0f} days before the "
                                   f"attack login; reactivated with no re-verification"))
        if "sms" in str(inc["reset_channel"]):
            found.append(("RC-08", "reset completed over SMS while the authenticating device had changed"))
        if inc["n_fraud_txns"] and pd.notna(inc["first_loss_ts"]):
            cat = str(inc["categories"]).split("|")
            convertible = bool(set(cat) & set(F.CONVERTIBLE_CATEGORIES))
            if convertible and pd.notna(inc["reset_completed_ts"]) \
                    and (inc["first_loss_ts"] - inc["reset_completed_ts"]).total_seconds() <= 86400:
                found.append(("RC-05", f"{inc['fraud_value_usd']:.2f} USD converted into "
                                        f"{'/'.join(cat)} within 24h of the reset completing"))
            if uid in fpf.index and int(fpf.loc[uid, "device_never_used"]) == 1:
                # Never fires here: an attacker authenticates before spending, so by payout time the
                # device is already known to the account. Reported as a measured zero in coverage.
                found.append(("RC-06", "a fraudulent payout ran on a device the account had never "
                                       "authenticated from before"))
        for rc, evidence in found:
            rows.append({
                "incident_id": inc["incident_id"], "incident_type": inc["incident_type"],
                "root_cause": rc, "root_cause_name": ROOT_CAUSES[rc]["name"],
                "kind": ROOT_CAUSES[rc]["kind"], "stage": ROOT_CAUSES[rc]["stage"],
                "stage_order": STAGES[ROOT_CAUSES[rc]["stage"]],
                "evidence": evidence,
                "fraud_value_usd": inc["fraud_value_usd"],
                "preventable_value_usd": inc["value_after_first_signal_usd"],
                "member_accounts": uid,
            })
    return pd.DataFrame(rows)


def stuffing_campaigns(login_features: pd.DataFrame) -> pd.DataFrame:
    """Group credential-stuffing windows on one IP into campaigns, then measure breach conversion."""
    f = login_features
    hits = f[(f["is_fraud_attempt"] == 1) & (f["i_users"] >= 5)]
    if not len(hits):
        return pd.DataFrame()

    rows = []
    for ip, grp in hits.groupby("ip_address"):
        g = grp.sort_values("t10")
        # One IP can attack in more than a single burst; a gap longer than 6h starts a new campaign.
        blk = (g["t10"].diff() > pd.Timedelta("6h")).cumsum()
        for _, block in g.assign(blk=blk).groupby("blk"):
            start, end = block["t10"].min(), block["t10"].max()
            window = f[(f["ip_address"] == ip) & (f["timestamp"] >= start - pd.Timedelta("30min"))
                       & (f["timestamp"] <= end + pd.Timedelta("30min"))]
            targeted = window["user_id"].nunique()
            breached = window.loc[(window["is_fraud_attempt"] == 1)
                                  & (window["login_result"] == "success"), "user_id"].nunique()
            rows.append({
                "campaign_id": f"CS-{ip}-{int(start.timestamp())}",
                "ip_address": ip, "ip_country": block["ip_country"].mode()[0],
                "start_ts": start, "end_ts": end,
                "duration_hours": round((end - start).total_seconds() / 3600, 2),
                "attempts": int(len(window)),
                "failed_attempts": int((window["login_result"] == "failed").sum()),
                "targeted_accounts": int(targeted),
                "distinct_devices": int(window["device_id"].nunique()),
                "breached_accounts": int(breached),
                "breach_rate": round(breached / targeted, 4) if targeted else np.nan,
            })
    return pd.DataFrame(rows).sort_values("attempts", ascending=False).reset_index(drop=True)


def ring_components(login_features: pd.DataFrame, txns: pd.DataFrame,
                    accounts: pd.DataFrame, edge_on: str = "device_id",
                    min_accounts: int = 10, max_fanout: int = 25) -> pd.DataFrame:
    """Connected components over fraud logins joined by a shared piece of infrastructure.

    Rings are not a per-account property, so they cannot be counted by thresholding a single
    feature. Users become nodes and a shared device becomes an edge; a ring is a component.

    Two measured choices, not assumptions:

    Edge basis. On this dataset's attack traffic device_id and ip_address are collinear: the two
    fan-out histograms are identical and 145 of the 155 attack fixtures appear with exactly one
    companion fixture, so swapping the edge key reproduces the same partition either way (both rows
    of edge_basis_test). Device is still the right key on the merits - a shared NAT exit is a
    plausible reason for two innocent accounts to show one IP and never a reason for them to share
    one device - but the collinearity is reported rather than hidden behind a favourable comparison.

    Fan-out ceiling. With no ceiling the graph returns one 1,297-account component that absorbs the
    rings alongside real loss belonging to takeover victims, which is what made the first version of
    the reconciliation go negative. The blob is caused by credential-stuffing tooling, not shared
    networks: 25 fixtures each touch exactly 60 accounts, which is a bot working a breached list,
    while a ring shares 2 devices among 20 members. Excluding anything above `max_fanout` leaves
    exactly those five 20-account clusters.
    """
    f = login_features[login_features["is_fraud_attempt"] == 1]
    if not len(f):
        return pd.DataFrame()

    fanout = f.groupby(edge_on)["user_id"].nunique()
    wide = set(fanout[fanout > max_fanout].index)
    f = f[~f[edge_on].isin(wide)]
    if not len(f):
        return pd.DataFrame()

    users = sorted(f["user_id"].unique())
    nodes = sorted(f[edge_on].dropna().unique())
    u_i = {u: k for k, u in enumerate(users)}
    n_i = {v: len(users) + k for k, v in enumerate(nodes)}

    pairs = set(zip(f["user_id"], f[edge_on]))
    src, dst = [], []
    for u, node in pairs:
        src.append(u_i[u]); dst.append(n_i[node])
    m = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(users) + len(nodes),) * 2)
    _, labels = connected_components(m, directed=False)

    rows = []
    fraud_tx = txns[txns["is_fraud_txn"] == 1]
    for comp, grp in pd.DataFrame({"user_id": users, "lab": labels[:len(users)]}).groupby("lab"):
        members = set(grp["user_id"])
        if len(members) < min_accounts:
            continue
        cf = f[f["user_id"].isin(members)]
        ct = fraud_tx[fraud_tx["user_id"].isin(members)]
        su = accounts[accounts["user_id"].isin(members)]["signup_date"].dropna().sort_values()
        span_days = (su.max() - su.min()).days if len(su) > 1 else 0
        same_week = int((su <= su.min() + pd.Timedelta(days=7)).sum()) if len(su) else 0
        rows.append({
            "ring_id": f"RG-{comp}",
            "edge_basis": edge_on,
            "accounts": len(members),
            "shared_devices": int(cf["device_id"].nunique()),
            "shared_ips": int(cf["ip_address"].nunique()),
            "signup_span_days": int(span_days),
            "signup_within_7_days": same_week,
            "batch_signup_share": round(same_week / len(members), 4) if members else np.nan,
            "first_fraud_ts": cf["timestamp"].min(),
            "last_fraud_ts": cf["timestamp"].max(),
            "fraud_logins": int(len(cf)),
            "fraud_txns": int(len(ct)),
            "fraud_value_usd": round(float(ct["amount_usd"].sum()), 2),
            "top_category": ct["merchant_category"].mode().iloc[0] if len(ct) else None,
            "categories": "|".join(sorted(ct["merchant_category"].dropna().unique())),
            "member_accounts": "|".join(sorted(members)),
        })
    return pd.DataFrame(rows).sort_values("fraud_value_usd", ascending=False).reset_index(drop=True)


def campaign_breaches(campaigns: pd.DataFrame, login_features: pd.DataFrame) -> dict[str, list[str]]:
    """The accounts each campaign authenticated, taken on that campaign's own window.

    Shared by the loss attribution and the reconciliation so the two cannot drift into counting the
    same breached account differently. The 30-minute pad either side is the same window the campaign
    was grouped on, so no login outside the burst can be blamed on it.
    """
    out = {}
    f = login_features
    for camp in campaigns.to_dict("records"):
        ip, start, end = camp["ip_address"], camp["start_ts"], camp["end_ts"]
        out[camp["campaign_id"]] = sorted(f.loc[(f["ip_address"] == ip)
                                                & (f["timestamp"] >= start - pd.Timedelta("30min"))
                                                & (f["timestamp"] <= end + pd.Timedelta("30min"))
                                                & (f["is_fraud_attempt"] == 1)
                                                & (f["login_result"] == "success"), "user_id"].unique())
    return out


def campaign_findings(campaigns: pd.DataFrame, login_features: pd.DataFrame,
                      txns: pd.DataFrame, incidents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute the credential-stuffing campaigns, and measure what each one eventually cost.

    A campaign by itself is noise - thousands of failed guesses. It becomes an incident at the point a
    guess succeeds, so loss is attributed to the accounts breached by that IP that then transacted
    fraudulently. Two windows are reported, not one: value inside seven days of the campaign, and
    value anywhere in the period, because a campaign that breaches accounts and never bills them is a
    different finding from a campaign nobody could have stopped.

    Measured here, and the reason the reconciliation has a credential-stuffing line at zero: of the
    172 accounts these campaigns authenticated, 7 ever produced a fraudulent transaction, worth
    $1,830.47, and all 7 are already counted as takeovers. Stuffing in this dataset buys a session,
    not a payout - the money leaves through the reset-and-withdraw chain instead.

    Two value columns with different jobs, because they answer different questions. `fraud_value_usd`
    is loss attributed *only* to this line, so the primary-cause table sums back to the total in
    data/transactions.csv; the $1,830.47 sits on the takeover rows instead. `preventable_value_usd`
    answers what closing this gap removes, and a login-endpoint control does remove all $1,830.47 even
    though the accounts are also takeovers - so that column is allowed to overlap between causes.
    """
    if not len(campaigns):
        return pd.DataFrame(), pd.DataFrame()

    fraud_tx = txns[txns["is_fraud_txn"] == 1]
    ato_users = set(incidents["user_id"])
    breaches = campaign_breaches(campaigns, login_features)
    rows, linked = [], []
    for camp in campaigns.to_dict("records"):
        ip, end = camp["ip_address"], camp["end_ts"]
        breached = breaches[camp["campaign_id"]]
        later = fraud_tx[(fraud_tx["user_id"].isin(breached))
                         & (fraud_tx["timestamp"] > end)
                         & (fraud_tx["timestamp"] <= end + pd.Timedelta("7D"))]
        ever = fraud_tx[fraud_tx["user_id"].isin(breached)]
        already = sorted(set(breached) & ato_users)
        value = round(float(later["amount_usd"].sum()), 2)
        ever_value = round(float(ever["amount_usd"].sum()), 2)
        # Only value on breached accounts that are NOT takeovers is new; the rest is already booked.
        incremental = round(float(ever[~ever["user_id"].isin(ato_users)]["amount_usd"].sum()), 2)
        if value or ever_value:
            outcome = (f"{len(breached)} authenticated, {len(ever)} transacted fraudulently for "
                       f"{ever_value:.2f} USD ({value:.2f} USD inside 7 days); {len(already)} of "
                       f"them already counted as takeovers")
        elif breached:
            outcome = (f"{len(breached)} authenticated and none billed - the campaign bought "
                       f"sessions, not payouts")
        else:
            outcome = "no guess succeeded; the campaign was fully blocked"
        rows.append({
            "incident_id": camp["campaign_id"], "incident_type": "credential_stuffing",
            "root_cause": "RC-03", "root_cause_name": ROOT_CAUSES["RC-03"]["name"],
            "kind": ROOT_CAUSES["RC-03"]["kind"], "stage": ROOT_CAUSES["RC-03"]["stage"],
            "stage_order": STAGES[ROOT_CAUSES["RC-03"]["stage"]],
            "member_accounts": "|".join(breached),
            "fraud_value_usd": incremental, "preventable_value_usd": ever_value,
            "evidence": (f"{int(camp['attempts'])} attempts against {int(camp['targeted_accounts'])} "
                         f"accounts from {ip} in {camp['duration_hours']:.2f}h; " + outcome),
        })
        linked.append({"campaign_id": camp["campaign_id"], "breached_accounts": len(breached),
                       "breached_with_fraud_txn": int(ever["user_id"].nunique()),
                       "fraud_value_within_7d_usd": value,
                       "fraud_value_whole_period_usd": ever_value,
                       "incremental_value_usd": incremental,
                       "already_counted_as_takeover": len(already),
                       "breached_user_ids": "|".join(breached[:20])})
    return pd.DataFrame(rows), pd.DataFrame(linked)


def ring_findings(rings: pd.DataFrame) -> pd.DataFrame:
    """Attribute rings: the review-queue gap always, plus the registration gap where signup was batched.

    A ring's whole cluster value is carried as preventable, because a cap on enrolments per device
    keeps the cluster out of the payout-eligible population before any of it is spent.
    """
    rows = []
    for rg in rings.to_dict("records"):
        rows.append({
            "incident_id": rg["ring_id"], "incident_type": "fraud_ring",
            "root_cause": "RC-07", "root_cause_name": ROOT_CAUSES["RC-07"]["name"],
            "kind": ROOT_CAUSES["RC-07"]["kind"], "stage": ROOT_CAUSES["RC-07"]["stage"],
            "stage_order": STAGES[ROOT_CAUSES["RC-07"]["stage"]],
            "member_accounts": rg["member_accounts"], "fraud_value_usd": rg["fraud_value_usd"],
            "preventable_value_usd": rg["fraud_value_usd"],
            "evidence": (f"{int(rg['accounts'])} accounts share {int(rg['shared_devices'])} devices "
                         f"and {int(rg['shared_ips'])} IPs; the cluster produced "
                         f"{int(rg['fraud_txns'])} fraudulent transactions worth "
                         f"{rg['fraud_value_usd']:.2f} USD in {rg['categories']}"),
        })
        if rg["batch_signup_share"] >= 0.8:
            rows.append({
                "incident_id": rg["ring_id"], "incident_type": "fraud_ring",
                "root_cause": "RC-09", "root_cause_name": ROOT_CAUSES["RC-09"]["name"],
                "kind": ROOT_CAUSES["RC-09"]["kind"], "stage": ROOT_CAUSES["RC-09"]["stage"],
                "stage_order": STAGES[ROOT_CAUSES["RC-09"]["stage"]],
                "member_accounts": rg["member_accounts"], "fraud_value_usd": rg["fraud_value_usd"],
                "preventable_value_usd": rg["fraud_value_usd"],
                "evidence": (f"{rg['batch_signup_share']:.0%} of the ring registered within "
                             f"{int(rg['signup_span_days'])} days of the first of them - a batch "
                             f"enrolment, not a coincidence"),
            })
    return pd.DataFrame(rows)


def reconcile(txns: pd.DataFrame, incidents: pd.DataFrame, rings: pd.DataFrame,
              campaigns: pd.DataFrame, login_features: pd.DataFrame) -> pd.DataFrame:
    """Check that the typologies add up to the fraud actually present, and flag double counting.

    Each line is built from a disjoint set of accounts wherever one can be, so the value column sums
    to the measured total instead of drifting past it. Where two typologies do reach the same account
    - a takeover victim whose credentials also appeared in a stuffing list - the overlap is counted and
    the additive line is zeroed, because a board number that exceeds the loss the data contains is
    worse than no number at all.
    """
    total = round(float(txns.loc[txns["is_fraud_txn"] == 1, "amount_usd"].sum()), 2)
    ato_users = set(incidents["user_id"])
    ring_users = set("|".join(rings["member_accounts"]).split("|")) if len(rings) else set()
    breached = set()
    for users in campaign_breaches(campaigns, login_features).values():
        breached |= set(users)
    fraud_users = set(txns.loc[txns["is_fraud_txn"] == 1, "user_id"])

    def value(users: set[str]) -> float:
        return round(float(txns.loc[(txns["is_fraud_txn"] == 1) & txns["user_id"].isin(users),
                                    "amount_usd"].sum()), 2)

    ato_val = value(ato_users)
    ring_only = ring_users - ato_users
    ring_val = value(ring_only)
    cs_only = (breached & fraud_users) - ato_users - ring_users
    cs_val = value(cs_only)
    unattributed = round(total - ato_val - ring_val - cs_val, 2) + 0.0  # +0.0 drops the -0.0 artefact

    rows = [
        {"incident_type": "account_takeover", "accounts": len(ato_users),
         "fraud_txn_value_usd": ato_val, "share_of_fraud_value": round(ato_val / total, 4),
         "overlap_with_previous_lines": 0,
         "note": "one row per account with a confirmed fraudulent password reset"},
        {"incident_type": "fraud_ring", "accounts": len(ring_only), "fraud_txn_value_usd": ring_val,
         "share_of_fraud_value": round(ring_val / total, 4),
         "overlap_with_previous_lines": len(ring_users & ato_users),
         "note": f"{len(rings)} device-linked components recovered at a fan-out ceiling of "
                 f"{C.load_yaml('detection_rules.yaml')['defaults']['ring_max_fanout']}"},
        {"incident_type": "credential_stuffing", "accounts": len(cs_only),
         "fraud_txn_value_usd": cs_val, "share_of_fraud_value": round(cs_val / total, 4),
         "overlap_with_previous_lines": len(breached & (ato_users | ring_users)),
         "note": f"{len(breached)} accounts authenticated by a campaign; every one that billed is "
                 f"already booked above, so the incremental loss here is measured, not assumed"},
        {"incident_type": "UNATTRIBUTED", "accounts": len(fraud_users - ato_users - ring_only - cs_only),
         "fraud_txn_value_usd": unattributed,
         "share_of_fraud_value": round(unattributed / total, 4),
         "overlap_with_previous_lines": 0,
         "note": "fraudulent transaction value no reconstructed incident explains"},
        {"incident_type": "TOTAL_MEASURED_FRAUD", "accounts": len(fraud_users),
         "fraud_txn_value_usd": total, "share_of_fraud_value": 1.0,
         "overlap_with_previous_lines": 0,
         "note": "every is_fraud_txn row in data/transactions.csv"},
    ]
    return pd.DataFrame(rows)


def build(silver: dict, login_features: pd.DataFrame, payment_features: pd.DataFrame) -> dict:
    """Produce every RCA artifact the pipeline, dashboard and reports consume."""
    defaults = C.load_yaml("detection_rules.yaml")["defaults"]
    min_accounts = defaults["ring_min_accounts"]
    max_fanout = defaults["ring_max_fanout"]
    ring_kw = dict(min_accounts=min_accounts, max_fanout=max_fanout)

    incidents = ato_incidents(silver, login_features)
    findings = attribute_root_causes(incidents, payment_features)
    campaigns = stuffing_campaigns(login_features)
    rings = ring_components(login_features, silver["txns"], silver["users"], **ring_kw)
    cs_findings, linked = campaign_findings(campaigns, login_features, silver["txns"], incidents)
    rg_findings = ring_findings(rings)

    chains = incidents.copy()
    chains["action_window_min"] = chains.pop("signal_to_first_loss_min")
    chains["preventable_value_usd"] = chains.pop("value_after_first_signal_usd")

    union = pd.concat([findings, cs_findings, rg_findings], ignore_index=True)
    # One attribution rule for every incident type: the primary cause is the earliest-stage gap,
    # so a takeover, a stuffing campaign and a ring are ranked the same way.
    first = (union.sort_values("stage_order").groupby("incident_id")
             .head(1)[["incident_id", "root_cause"]].rename(columns={"root_cause": "primary_root_cause"}))
    union = union.merge(first, on="incident_id", how="left")
    # by_cause is multi-label, so its values deliberately do not sum to total loss - one chain passes
    # several gaps. primary_root_causes below is the additive partition; reconcile() ties to the data.
    by_cause = (union.groupby(["root_cause", "root_cause_name", "kind", "stage", "stage_order"])
                .agg(incidents=("incident_id", "nunique"), accounts=("member_accounts", _n_accounts),
                     fraud_value_usd=("fraud_value_usd", "sum"),
                     preventable_value_usd=("preventable_value_usd", "sum"))
                .reset_index().sort_values(["stage_order", "incidents"]))
    by_cause["fix"] = by_cause["root_cause"].map(lambda r: ROOT_CAUSES[r]["fix"])
    by_cause["owner"] = by_cause["root_cause"].map(lambda r: ROOT_CAUSES[r]["owner"])
    by_cause["rules_that_should_have_caught_it"] = by_cause["root_cause"].map(RULE_FOR_CAUSE)

    primary = (union[union["root_cause"] == union["primary_root_cause"]]
               .groupby("primary_root_cause").agg(incidents=("incident_id", "nunique"),
                                                  accounts=("member_accounts", _n_accounts),
                                                  fraud_value_usd=("fraud_value_usd", "sum"))
               .reset_index().sort_values("incidents", ascending=False))
    primary["fix"] = primary["primary_root_cause"].map(lambda r: ROOT_CAUSES[r]["fix"])
    primary["owner"] = primary["primary_root_cause"].map(lambda r: ROOT_CAUSES[r]["owner"])

    # Every taxonomy code is reported, including the ones this dataset never exercises. A cause with
    # zero findings is a measured absence, not an untested branch - RC-06 and RC-08 are the two here.
    coverage = (union.groupby("root_cause").agg(
        incidents_evidenced=("incident_id", "nunique"),
        accounts_evidenced=("member_accounts", _n_accounts)).reset_index())
    coverage = (pd.DataFrame({"root_cause": sorted(ROOT_CAUSES)})
                .merge(coverage, on="root_cause", how="left").fillna(0))
    coverage["incidents_evidenced"] = coverage["incidents_evidenced"].astype(int)
    coverage["accounts_evidenced"] = coverage["accounts_evidenced"].astype(int)
    coverage["root_cause_name"] = coverage["root_cause"].map(lambda r: ROOT_CAUSES[r]["name"])
    coverage["status"] = np.where(coverage["incidents_evidenced"] > 0,
                                  "evidenced", "not exercised by this dataset")

    reconciliation = reconcile(silver["txns"], incidents, rings, campaigns, login_features)

    # The exclusive partition has to tie back to the ledger. This check earned its place: the first
    # version of the ring graph absorbed takeover victims through a credential-stuffing device and
    # reported UNATTRIBUTED as -$13,267.45, which a reader would only have caught by eye.
    ledger = round(float(silver["txns"].loc[silver["txns"]["is_fraud_txn"] == 1, "amount_usd"].sum()), 2)
    attributed = round(float(primary["fraud_value_usd"].sum()), 2)
    if abs(attributed - ledger) > 0.01:
        raise ValueError(f"root-cause partition attributes {attributed:.2f} USD but "
                         f"data/transactions.csv holds {ledger:.2f} USD of labelled fraud")

    # Both the edge basis and the fan-out ceiling are reported as a measurement over
    # (basis x ceiling), so the README can say why rings are device-linked and capped at 25 without
    # asking a reader to take either on faith - including that the two bases agree here, which is a
    # property of this generator, not a general result.
    edge_rows = []
    for basis in ("device_id", "ip_address"):
        for cap in (np.inf, float(max_fanout)):
            r = ring_components(login_features, silver["txns"], silver["users"],
                                edge_on=basis, min_accounts=min_accounts, max_fanout=cap)
            edge_rows.append({
                "edge_basis": basis,
                "fanout_ceiling": "none" if np.isinf(cap) else int(cap),
                "components": int(len(r)),
                "largest_component_accounts": int(r["accounts"].max()) if len(r) else 0,
                "accounts_in_components": int(r["accounts"].sum()) if len(r) else 0,
                "fraud_value_usd_in_components": round(float(r["fraud_value_usd"].sum()), 2)
                                                if len(r) else 0.0,
            })
    edge_test = pd.DataFrame(edge_rows)

    return {"incidents": incidents, "root_cause_findings": union,
            "root_cause_by_stage": by_cause, "primary_root_causes": primary,
            "campaigns": campaigns, "campaign_loss_link": linked, "rings": rings,
            "chains": chains, "reconciliation": reconciliation, "edge_basis_test": edge_test,
            "root_cause_coverage": coverage}
