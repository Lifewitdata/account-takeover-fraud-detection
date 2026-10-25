"""Rule engine.

Predicates live in config/detection_rules.yaml as human-readable text, and are implemented here as
explicit callables. They are deliberately NOT evaluated with `df.eval`/`eval`: executing strings
loaded from a config file buys nothing here and turns a config edit into code execution. Instead
`load_rules()` asserts the registry and this module cover exactly the same rule ids, so a rule added
to YAML without an implementation fails loudly at import time rather than silently never firing.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

CATEGORIES = ("gift_cards", "jewelry", "digital_goods")


def _r01(f: pd.DataFrame) -> pd.Series:
    return f["i_users"] >= 5


def _r02(f: pd.DataFrame) -> pd.Series:
    return f["device_n_users"] >= 5


def _r03(f: pd.DataFrame) -> pd.Series:
    return f["new_device"] == 1


def _r04(f: pd.DataFrame) -> pd.Series:
    return f["impossible_travel"] == 1


def _r05(f: pd.DataFrame) -> pd.Series:
    return (f["hours_since_reset"] <= 24) & (f["amount_usd"] >= 100)


def _r06(f: pd.DataFrame) -> pd.Series:
    return (f["paid_on_new_device"] == 1) & (f["amount_usd"] >= 100)


def _r07(f: pd.DataFrame) -> pd.Series:
    return f["amount_zscore"] >= 6


def _r08(f: pd.DataFrame) -> pd.Series:
    return f["merchant_category"].isin(CATEGORIES)


def _r11(f: pd.DataFrame) -> pd.Series:
    return (f["hours_since_new_device_login"] <= 24) & (f["amount_usd"] >= 100)


def _r09(f: pd.DataFrame) -> pd.Series:
    return f["risk_score"] >= C.load_yaml("monitoring.yaml")["thresholds"]["default"]


def _r10(f: pd.DataFrame) -> pd.Series:
    return f["ip_n_users"] >= 10


LOGIN_RULES = {
    "R01_ip_velocity_5u10m": _r01,
    "R02_shared_device_5u": _r02,
    "R03_new_device_step_up": _r03,
    "R04_impossible_travel": _r04,
    "R09_model_score": _r09,
    "R10_ip_neighbourhood": _r10,
}

PAYMENT_RULES = {
    "R05_reset_then_payout": _r05,
    "R06_new_device_payout": _r06,
    "R11_recent_new_device_login": _r11,
    "R07_amount_zscore_spike": _r07,
    "R08_convertible_category": _r08,
}

RULE_SURFACES = {**{k: "login" for k in LOGIN_RULES}, **{k: "payment" for k in PAYMENT_RULES}}


def load_rules() -> dict:
    """Return the registry from YAML, cross-checked against the implementations in this module."""
    reg = C.load_yaml("detection_rules.yaml")
    by_id = {r["id"]: r for r in reg["rules"]}
    missing_impl = sorted(set(by_id) - set(RULE_SURFACES))
    missing_doc = sorted(set(RULE_SURFACES) - set(by_id))
    if missing_impl or missing_doc:
        raise ValueError(
            "detection_rules.yaml and src/ato/rules.py disagree: "
            f"no implementation for {missing_impl}; not in registry {missing_doc}")
    for rid, spec in by_id.items():
        if spec["surface"] != RULE_SURFACES[rid]:
            raise ValueError(f"{rid}: registry says surface {spec['surface']}, "
                             f"implementation is a {RULE_SURFACES[rid]} rule")
    return reg


def apply_rules(f: pd.DataFrame, surface: str) -> pd.DataFrame:
    """Add `flag_<rule_id>` boolean columns for every rule on that surface."""
    out = f.copy()
    registry = LOGIN_RULES if surface == "login" else PAYMENT_RULES
    for rid, fn in registry.items():
        out[f"flag_{rid}"] = fn(out).fillna(False).astype(bool)
    return out


def rule_ids(surface: str | None = None) -> list[str]:
    if surface is None:
        return sorted(RULE_SURFACES)
    return sorted(k for k, v in RULE_SURFACES.items() if v == surface)
