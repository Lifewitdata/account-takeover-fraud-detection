"""Fast unit tests: no pipeline run, no model fit. These pin the arithmetic the reports rest on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ato import compliance as CP
from ato import config as C
from ato import features as F
from ato import monitoring as M
from ato import rules as R

DAY_NS = F.DAY_NS


def test_trailing_window_is_a_window_not_a_running_total():
    # Five events, one per day: a 1-day lookback must see at most the previous event.
    ts = np.arange(5, dtype="int64") * DAY_NS
    n, s, _ = F._trailing_stats(ts, np.ones(5), 1.0)
    assert n.tolist() == [0, 1, 1, 1, 1]
    n90, _, _ = F._trailing_stats(ts, np.ones(5), 90.0)
    assert n90.tolist() == [0, 1, 2, 3, 4]


def test_payment_window_respects_microsecond_timestamps():
    # pandas 3 parses CSV timestamps at microsecond resolution; the 24h count must still be daily.
    ts = pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-09"]).astype("datetime64[us]")
    txns = pd.DataFrame({"txn_id": ["t1", "t2", "t3"], "user_id": "U1", "timestamp": ts,
                         "amount_usd": [10.0, 20.0, 30.0], "merchant_category": "grocery",
                         "device_id": "D1", "ip_address": "1.1.1.1", "is_fraud_txn": 0})
    logins = pd.DataFrame({"user_id": ["U1"], "timestamp": ts[:1], "device_id": ["D1"],
                           "ip_address": ["1.1.1.1"], "login_result": ["success"],
                           "new_device": [0]})
    resets = pd.DataFrame({"reset_id": ["r1"], "user_id": ["U9"], "request_timestamp": ts[:1],
                           "complete_timestamp": ts[:1], "status": ["completed"],
                           "device_id": ["D9"], "ip_address": ["9.9.9.9"]})
    out = F.add_payment_features(txns, logins, resets, history_days=90, min_history=5)
    assert out["txn_count_user_24h"].tolist() == [1, 1, 1]
    assert out["history_n"].tolist() == [0, 1, 2]


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert M.psi(a, a.copy()) == pytest.approx(0.0, abs=1e-9)
    assert M.psi(a, a + 2.0) > 0.25


def test_psi_handles_binary_features():
    base = np.array([0] * 900 + [1] * 100, dtype=float)
    same = np.array([0] * 450 + [1] * 50, dtype=float)
    assert M.psi(base, same) == pytest.approx(0.0, abs=1e-9)
    assert M.psi(base, np.array([0] * 500 + [1] * 500, dtype=float)) > 0.25


def test_rule_registry_matches_implementations():
    reg = R.load_rules()
    assert {r["id"] for r in reg["rules"]} == set(R.RULE_SURFACES)


def test_every_disposition_is_priced():
    cm = C.load_yaml("cost_model.yaml")
    used = {r["disposition"] for r in R.load_rules()["rules"]}
    assert used <= set(cm["actions"])


def test_compliance_matrix_parses_and_uses_known_rules():
    m = pd.read_csv(C.CONFIG / "compliance_matrix.csv", encoding="utf-8")
    assert len(m) == 24 and m["obligation_id"].is_unique
    known = {rid.split("_")[0] for rid in R.RULE_SURFACES}
    for ids in m["rule_ids"]:
        assert set(ids.split("|")) <= known, ids
    assert set(m["status"]) <= {"evidence_generated", "design_mapped", "out_of_scope"}


def test_obligation_check_fails_when_evidence_is_missing():
    status = CP.obligation_status({})
    claimed = status["matrix_status"] == "evidence_generated"
    assert status.loc[claimed, "check"].str.startswith("FAIL").all()
    assert not status.loc[~claimed, "check"].str.startswith("FAIL").any()


def test_rule_decay_flags_a_precision_collapse():
    mon = C.load_yaml("monitoring.yaml")
    weekly = pd.DataFrame([
        {"surface": "login", "rule_id": "RX", "window": "baseline", "week_start": "2025-01-06",
         "events": 100, "fraud_events": 10, "alerts": 10, "fraud_hits": 9, "legit_hits": 1,
         "below_min_events": False},
        {"surface": "login", "rule_id": "RX", "window": "current", "week_start": "2025-10-06",
         "events": 100, "fraud_events": 10, "alerts": 20, "fraud_hits": 9, "legit_hits": 11,
         "below_min_events": False},
    ])
    out = M.rule_decay(weekly, mon).iloc[0]
    assert out["status"] == "escalate"
    assert out["worst_metric"] == "precision"
