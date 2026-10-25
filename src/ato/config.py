"""Paths, configuration loading and shared constants.

Every module resolves paths from this file rather than from the current working directory,
so `python -m ato.pipeline` works from anywhere in the checkout and the tests do not depend
on where pytest was launched.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

RANDOM_STATE = 42


def _resolve_root() -> Path:
    env = os.environ.get("ATO_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "login_attempts.csv").exists():
            return parent
    # src/ato/config.py -> src/ato -> src -> project root
    return here.parents[2]


ROOT = _resolve_root()
DATA = ROOT / "data"
CONFIG = ROOT / "config"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
SQL = ROOT / "sql"

MODEL_DIR = OUTPUTS / "models"
FEATURE_DIR = OUTPUTS / "features"
SCORED_DIR = OUTPUTS / "scored"
INCIDENT_DIR = OUTPUTS / "incidents"
EXPOSURE_DIR = OUTPUTS / "exposure"
MONITOR_DIR = OUTPUTS / "monitoring"
COMPLIANCE_DIR = OUTPUTS / "compliance"
SQL_DIR = OUTPUTS / "sql"

ALL_DIRS = [OUTPUTS, MODEL_DIR, FEATURE_DIR, SCORED_DIR, INCIDENT_DIR,
            EXPOSURE_DIR, MONITOR_DIR, COMPLIANCE_DIR, SQL_DIR]

TRAIN_END = pd.Timestamp("2025-10-01")
LABEL_COLS = ["is_fraud_attempt", "is_fraud_reset", "is_fraud_txn"]


def load_yaml(name: str) -> dict:
    """Read a YAML file from config/ as UTF-8 (Windows default codepage would mangle it)."""
    path = CONFIG / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, directory: Path, filename: str) -> Path:
    """Write a UTF-8 CSV with LF endings so regenerated and committed bytes agree everywhere."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return path


def write_text(text: str, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
