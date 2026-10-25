"""Invariants over the files `python -m ato.pipeline` writes. CI regenerates outputs/ first.

These check the claims the README makes, not just that files exist: the loss reconciliation ties to
the ledger, the compliance matrix never claims evidence that was not generated, and every exposure
summary line says whether it is measured, derived or assumed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ato import config as C

pytestmark = pytest.mark.skipif(not (C.OUTPUTS / "run_manifest.csv").exists(),
                                reason="run `python -m ato.pipeline` first")


def _read(sub: str, name: str) -> pd.DataFrame:
    return pd.read_csv(C.OUTPUTS / sub / name, encoding="utf-8")


def test_manifest_artifacts_exist_and_match_row_counts():
    manifest = pd.read_csv(C.OUTPUTS / "run_manifest.csv")
    for _, row in manifest.iterrows():
        path = C.ROOT / row["artifact"]
        assert path.exists(), row["artifact"]
        assert len(pd.read_csv(path)) == row["rows"], row["artifact"]


def test_loss_reconciliation_ties_to_the_ledger():
    rec = _read("incidents", "loss_reconciliation.csv").set_index("incident_type")
    total = rec.loc["TOTAL_MEASURED_FRAUD", "fraud_txn_value_usd"]
    parts = rec.drop(index="TOTAL_MEASURED_FRAUD")["fraud_txn_value_usd"].sum()
    assert parts == pytest.approx(total, abs=0.01)
    assert rec.loc["UNATTRIBUTED", "fraud_txn_value_usd"] == pytest.approx(0.0, abs=0.01)


def test_primary_root_causes_partition_the_loss():
    primary = _read("incidents", "primary_root_causes.csv")
    rec = _read("incidents", "loss_reconciliation.csv").set_index("incident_type")
    assert primary["fraud_value_usd"].sum() == pytest.approx(
        rec.loc["TOTAL_MEASURED_FRAUD", "fraud_txn_value_usd"], abs=0.01)


def test_every_root_cause_is_reported_including_unexercised_ones():
    cov = _read("incidents", "root_cause_coverage.csv")
    assert len(cov) == 9
    assert set(cov["status"]) <= {"evidenced", "not exercised by this dataset"}


def test_rings_meet_the_minimum_size():
    rings = _read("incidents", "fraud_rings.csv")
    cap = C.load_yaml("detection_rules.yaml")["defaults"]["ring_min_accounts"]
    assert (rings["accounts"] >= cap).all()


def test_exposure_lines_declare_their_basis():
    s = _read("exposure", "exposure_summary.csv")
    assert set(s["basis"]) <= {"measured", "derived", "assumption"}
    assert s["source_or_assumption"].notna().all()


def test_programme_cost_is_not_more_than_the_naive_sum():
    s = _read("exposure", "exposure_summary.csv").set_index("line")["value"]
    assert s["programme_control_cost_deduplicated"] <= s["programme_control_cost_naive_sum"]


def test_control_verdicts_are_from_the_documented_set():
    cc = _read("exposure", "control_cost.csv")
    assert set(cc["verdict"]) <= {"no measured benefit", "costs more than it saves", "keep",
                                  "signal only, no action taken", "consumes too much queue capacity"}
    assert len(cc) == len(C.load_yaml("detection_rules.yaml")["rules"])


def test_compliance_never_overclaims():
    ob = _read("compliance", "obligation_status.csv")
    assert not ob["check"].str.startswith("FAIL").any()
    claimed = ob[ob["matrix_status"] == "evidence_generated"]
    assert claimed["evidence_present"].all()


def test_every_configured_kpi_is_reported_once():
    kpis = _read("monitoring", "kpi_latest.csv")
    configured = [k["id"] for k in C.load_yaml("monitoring.yaml")["kpis"]]
    assert sorted(kpis["kpi_id"]) == sorted(configured)


def test_alerts_have_an_owner_and_due_date():
    alerts = _read("monitoring", "alerts.csv")
    assert alerts["owner"].notna().all()
    assert (pd.to_datetime(alerts["due_by"]) >= pd.to_datetime(alerts["raised_on"])).all()


def test_drift_reported_for_both_populations():
    d = _read("monitoring", "drift_features.csv")
    assert set(d["population"]) == {"all_rows", "legitimate_only"}
    assert set(d["surface"]) == {"login", "payment"}
