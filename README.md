# Account Takeover, Payment Fraud & Fraud-Ring Detection

**Trust & Safety analytics: from login and payment telemetry to scored queues, root causes,
dollar exposure, drift monitoring and regulatory evidence.**

One command (`python -m ato.pipeline`) rebuilds every number in this README from `data/`:
42 artifacts, a run manifest, and a non-zero exit if any compliance evidence it claims is missing.
A weekly GitHub Actions job runs the same pipeline and posts the KPI and alert tables.

## TL;DR

| Question | Answer (test window Oct–Dec 2025 unless stated) |
|---|---|
| How much fraud? | **$93,476.68** labelled fraud value across 220 accounts in 2025, fully attributed: account takeover $41,008.68 (120 accounts), fraud rings $52,468.00 (100 accounts), $0 unattributed |
| What does it cost? | **$81,967.51** total fraud cost after 25% recovery, disputes and remediation, i.e. **$683.06 per confirmed takeover** |
| What do the controls cost? | **$9,457.25** deduplicated (the naive per-rule sum, $11,540.45, double-bills shared cases); **$2,559.92** of that is false-positive cost |
| Why did it happen? | Every takeover passed through an unseen device that got no step-up (RC-02); 102 of 120 kept a live session after the password reset (RC-01); rings registered together with no enrolment velocity cap (RC-09) |
| What is drifting? | 2 rules breach the escalation threshold: **R08** precision 0.194 → 0.023, **R03** 0.162 → 0.091. Max feature PSI is 0.192 (moderate) |
| Is the queue staffable? | **5.25** deduplicated review cases per day against a capacity of 30 |
| Compliance | 24 obligations (8 India, 4 EU, 2 UK, 5 US/PCI, 5 global standards). 17 have generated evidence, all verified present; 6 are mapped only, and 1 is listed as out of scope |

**Caveat up front:** the data is synthetic and the fraud is deliberately clean-cut, so the model
scores (AUC ≈ 1.0) are illustrative. The methodology is what transfers to production: past-only
features, time-split evaluation, deduplicated costing, and drift against an out-of-sample baseline.

## Data

All synthetic, seed `42`. The CSVs are committed; the generator script is not part of this repo.
No real users. Ground-truth columns (`is_fraud_*`) are used **only** for evaluation, never as features.

| File | Rows | Fraud |
|---|---|---|
| `data/users.csv` | 5,100 | — |
| `data/devices.csv` | 7,150 | — |
| `data/login_attempts.csv` | 162,778 | 11,885 attempts (7.30%) |
| `data/password_resets.csv` | 483 | 120 resets |
| `data/transactions.csv` | 61,560 | 1,238 txns (2.01%) |

Injected scenarios: credential stuffing (25 campaigns), account takeover (120 users: new-device login,
then password reset, then high-value payouts), fraud rings (5 × 20 users sharing devices and IPs).
Schema: [`data/data_dictionary.md`](data/data_dictionary.md).

## Architecture

```
data/*.csv ─► src/ato/data.py          data-quality audit → outputs/dq_findings.csv
            ─► features.py             past-only login + payment features
            ─► scoring.py / rules.py   time-split models + 11-rule registry (config/detection_rules.yaml)
            ─► incidents.py            confirmed-incident reconstruction + root-cause analysis
            ─► exposure.py             loss model, control cost, threshold economics (config/cost_model.yaml)
            ─► compliance.py           evidence for each obligation in config/compliance_matrix.csv
            ─► monitoring.py           weekly rule KPIs, PSI/KS drift, rule decay, alerts (config/monitoring.yaml)
            ─► pipeline.py             writes outputs/**, run_manifest.csv, fails on missing evidence
dashboard/app.py                       reads outputs/ only
```

All thresholds, costs and obligations live in `config/`. Every cost input is listed in
`outputs/exposure/assumption_ledger.csv` and labelled as either an assumption or config.

## Gap-by-gap

### 1. Payment fraud is now a first-class surface

Payments get their own past-only features, five registry rules (R05–R08, R11), a time-split model, a
threshold table and an alert queue (`outputs/scored/payment_alert_queue.csv`).

The features are:
- amount z-score against the account's own 90-day history
- hours since reset and hours since new-device login
- 24-hour transaction velocity
- first-time category
- device and IP continuity

| Rule | Hits | Precision | Share of payment fraud caught |
|---|---|---|---|
| R11 payout within 24h of a new-device login | 46 | 0.85 | 69.6% |
| R05 payout after a password reset | 38 | 1.00 | 67.9% |
| R08 convertible category (gift cards, jewelry) | 1,554 | 0.02 | 64.3% |
| R07 amount z-score spike | 126 | 0.04 | 8.9% |
| R06 new-device payout | 1 | 0.00 | 0% |

The payment model reaches precision@50 = 1.00 and precision@100 = 0.56, which is exactly the 56
fraud rows in the window. **AUC is 1.0, and that partly reflects how the data was generated:**
- the generator gives every legitimate transaction a fresh IP, which makes `ip_never_used_by_user`
  a near-perfect separator
- the pipeline refits without it (`feature_set = artefact_free`) and the AUC is *still* 1.0, because
  the device-sharing and new-device-timing signals are also cleaner than real data would be

Treat the payment model as a demonstration of the evaluation harness, not of achievable lift.

### 2. Root-cause analysis of confirmed incidents

`incidents.py` rebuilds each confirmed takeover as a kill chain: first signal, unseen-device login,
reset, session reuse, and payouts. It attributes every fraud dollar and ties out to the ledger
(`outputs/incidents/loss_reconciliation.csv`):
- account takeover $41,008.68
- rings $52,468.00
- stuffing $0 incremental: 172 accounts were authenticated by a campaign, and the 7 that went on to
  transact are already booked above
- total $93,476.68

Root causes RC-01..RC-09 are scored two ways. **Primary** is the earliest-stage gap; it is additive
and sums to the ledger. **All causes** overlap and answer "how much would this fix have prevented?"

| Root cause | Incidents | Preventable value | Fix | Owner |
|---|---|---|---|---|
| RC-09 no registration velocity control | 5 rings / 100 accts | $52,468.00 | Cap enrolments per device/IP; hold co-registered clusters out of payout | Identity & onboarding |
| RC-02 unseen device authenticated without step-up | 120 | $40,727.43 | Step-up on first-seen device; limited session until cleared | Authentication eng. |
| RC-05 instant payout in high-conversion category | 101 | $36,064.93 | Hold gift-card/jewelry value until device and reset state age out | Payments risk |
| RC-01 session survived a password reset | 102 | $33,800.49 | Invalidate all sessions and refresh tokens on credential change | Platform identity |
| RC-04 dormant account reactivated without re-verification | 10 | $3,475.01 | Re-verify after idle threshold before transacting | Trust & Safety |
| RC-03 no throttle or breached-credential screening | 25 campaigns | $1,830.47 | Rate-limit, lock out, screen against breach corpora (NIST 800-63B) | AppSec |

Full table with every cause, including RC-07 (a confirmed account that never triggered a re-score of
its device neighbourhood): `outputs/exposure/preventable_loss_by_control.csv`. Per-incident chains:
`outputs/incidents/incident_chains.csv`.

**The action window is wide.** The median gap between the first qualifying signal and the first
payout is **554.5 minutes**, and 98% of chains leave at least the 30-minute action SLA. Acting within
30 minutes would still have stopped $40,558.90 (98.9% of takeover value), and the loss grows by
roughly $380 for every extra 15 minutes of delay past two hours
(`outputs/exposure/enforcement_latency.csv`). The loss is therefore an enforcement gap, not a
detection gap.

### 3. Risk exposure in business terms

`config/cost_model.yaml` prices each **disposition**, not each rule:
- `auto_block`, `hold` and `step_up` prevent a payout, so their benefit is the payout prevented
- `review` recovers value after the fact, so its benefit is recovered value
- `monitor` costs nothing and earns nothing

False positives are charged once per row (appeal, support, abandoned session revenue), and review
cases are deduplicated per account.

| Line | Value | Basis |
|---|---|---|
| Fraud value, whole year | $93,476.68 | measured |
| Net loss after 25% recovery | $70,107.51 | derived |
| Dispute + remediation cost | $11,860.00 | derived |
| **Total fraud cost** | **$81,967.51** | derived |
| Programme control cost, deduplicated (Oct–Dec) | $9,457.25 | derived |
| … of which wrongly-actioned customers | $2,559.92 | derived |
| Control cost per dollar recovered | 1.95 | derived |
| Fraud value stopped before payout (K06) | $9,072.25 of $10,232.18 | measured |

The per-rule verdicts are in `outputs/exposure/control_cost.csv`:
- **keep:** R03, R05 and R11
  - R05 and R11 hold payouts at $106–$133 of cost for $8.6k–$9.1k prevented
- **costs more than it saves:** R01, R02, R09 and R07
  - login rules mostly catch stuffing traffic that never reaches a payout
- **signal only:** R04, R08 and R10
- **no measured benefit:** R06

`threshold_economics.csv` prices every candidate threshold against the review capacity.

### 4. Scheduled monitoring instead of "re-tune monthly"

`config/monitoring.yaml` defines eight KPIs, each with a window, an out-of-sample baseline
(Jan–Sep), a drift test and an action threshold. `monitoring.py` evaluates them.

| KPI | Latest | Status |
|---|---|---|
| K01 fraud attempts detected / week | 346 | ok |
| K02 lowest rule precision | 0.023 (R08) | **escalate** |
| K03 review cases / day | 5.25 vs capacity 30 | ok |
| K04 max feature PSI | 0.192 (`payment.is_first_category`) | moderate |
| K05 false-positive cost / week | $182.85 | trend |
| K06 fraud value stopped pre-payout | $9,072.25 | ok |
| K07 median action window | 554.5 min | ok |
| K08 obligations with evidence | 100% | ok |

Design choices that keep the alerts honest:
- **Two drift populations.** PSI and KS run on legitimate-only rows (pure population drift) and on
  all rows (which also moves when the fraud mix moves).
- **Warm-up exclusion.** The first 90 days of payment history are excluded from the baseline,
  because personal-history features are undefined there and would fake a drift signal.
- **Out-of-sample baseline only.** The model-score rule R09 is trained on the baseline period, so
  it has no out-of-sample baseline. It is masked rather than compared.
- **Alerts carry an owner and a deadline.** `outputs/monitoring/alerts.csv` adds an owner and a
  business-day due date per the SLA, plus an empty `decision` column for the reviewer to close out.

`.github/workflows/monitoring.yml` runs every Monday at 06:17 UTC. It writes the KPI and alert
tables to the job summary and keeps the full `outputs/` as a 90-day artifact.
`python -m ato.pipeline --fail-on-alert` exits 2 when an alert is open, for schedulers that
page on exit codes.

### 5. Regulatory and compliance layer (India-first)

`config/compliance_matrix.csv` maps 24 obligations to a control, the registry rules that operate it,
and the artifact that evidences it.

| Jurisdiction | Instruments |
|---|---|
| India (8) | PMLA / FIU-IND STR, RBI KYC Master Direction, RBI fraud reporting in customer-interfaced products, RBI payment aggregator guidelines, RBI digital lending, DPDP Act 2023 s.8(6), IT Act s.43A, SEBI CSCRF |
| EU (4) | PSD2 Art. 97 + RTS on SCA, PSD2 Art. 95–96, GDPR Art. 33–34, DORA Art. 19 |
| UK (2) | PSR APP-scam reimbursement, FCA operational resilience |
| US / card (5) | BSA SAR (31 CFR 1010.320), Reg E, FTC Safeguards Rule, PCI DSS v4.0 Req. 8 & 12.10, FinCEN/CISA/FBI RAT-takeover advisory |
| Standards (5) | NIST SP 800-63B, ISO/IEC 27001:2022, OWASP ASVS, FATF R.15/16, NIST CSF 2.0 |

`compliance.py` generates the evidence files:
- STR/SAR candidates with filing due dates
- re-KYC queue
- fraud escalation log
- monthly payment-fraud report by category
- 72-hour breach assessment
- SCA decision log
- credential-stuffing throttle report
- control evidence index

`outputs/compliance/obligation_status.csv` checks that each claimed artifact exists and is non-empty.
The pipeline **fails** if the matrix claims evidence that was not produced. Six obligations are marked
`design_mapped` rather than claimed: they apply only to some entity types (DORA, FTC Safeguards,
digital lending) or are audited outside an analytics repo (ISO 27001, FCA impact tolerances, FATF).
SEBI CSCRF is listed as out of scope.

This is a mapping for analytics design, not legal advice. Instrument references were current as of
2025 and should be confirmed with counsel before use.

## Model results

Time split: train Jan–Sep 2025 (122,788 logins), test Oct–Dec 2025 (39,990). All features are
past-only.

| Surface | Model | ROC-AUC | Avg precision |
|---|---|---|---|
| Login | Logistic regression | 0.9996 | 0.9976 |
| Login | Random forest | 0.9998 | 0.9979 |
| Payment | Random forest | 1.000 | 1.000 (see artefact caveat above) |

- **Login precision@k:** 1.00 through k = 2,000, then 0.44 at k = 5,000.
- **Login thresholds:** at 0.5 the model raises 2,195 alerts with 100% precision and 98.7% recall,
  which collapses to 353 accounts.
- **Top login features:** distinct users per IP (0.28) and per device (0.25), then IP 10-minute
  velocity.
- **Impossible travel is a weak signal:** 0.7% recall, 7.5% precision. Takeovers follow long idle
  gaps, so speed checks rarely fire.

## SQL

`sql/fraud_features.sql` is the BigQuery version of the login features. It was **ported and
dialect-verified, executed on a DuckDB mirror, never run on a live warehouse**:
- **Dialect-verified:** sqlglot parses it as BigQuery (`outputs/sql/bigquery_dialect_check.txt`).
- **Executed on a DuckDB mirror:** `sql/duckdb/login_features.sql` runs against the committed CSVs.
- **Parity with pandas:** all 15 features agree row-for-row on 162,778 rows
  (`outputs/sql/sql_parity.csv`; max float difference 3e-12).

See [`sql/README.md`](sql/README.md).

## Correctness note

pandas 3 parses timestamps as `datetime64[us]`, and an earlier version cast them straight to int64
and compared them against nanosecond windows. The result was that a "1-day" payment lookback was
really 1,000 days. Timestamps are now cast to ns first, and
`tests/test_units.py::test_payment_window_respects_microsecond_timestamps` fails on the old code.

## How to run

```bash
pip install -r requirements.txt -r requirements-dev.txt
make pipeline        # or: PYTHONPATH=src python -m ato.pipeline   (~3 min, writes outputs/)
make test            # unit + output-invariant tests
make sql             # DuckDB execution + pandas parity (33 tests in total)
make dashboard       # streamlit run dashboard/app.py
```

CI (`.github/workflows/ci.yml`) runs the unit tests, the pipeline, the output tests and SQL parity
on pushes to `main` and on pull requests. The R replication of the statistical checks is at `r/stats_validation.R`.

The notebook `notebooks/account_takeover_fraud_detection.ipynb` is the original exploratory analysis.
The figures in `reports/figures/` come from it. The pipeline, not the notebook, is now the source of
truth for `outputs/`.

## Project structure

```
account-takeover-fraud-detection/
├── config/                 detection_rules.yaml, cost_model.yaml, monitoring.yaml, compliance_matrix.csv
├── data/                   synthetic CSVs + data_dictionary.md
├── src/ato/                data, features, rules, scoring, incidents, exposure, compliance,
│                           monitoring, sqlrun, pipeline
├── sql/                    fraud_features.sql (BigQuery), duckdb/login_features.sql, README.md
├── tests/                  test_units.py, test_outputs.py, test_sql_parity.py
├── outputs/                scored/, incidents/, exposure/, compliance/, monitoring/, sql/,
│                           dq_findings.csv, run_manifest.csv
├── dashboard/app.py        Streamlit: monitoring, RCA, exposure, payments, compliance, review queue
├── .github/workflows/      ci.yml, monitoring.yml (weekly)
├── notebooks/              original exploratory notebook
├── r/                      stats_validation.R
├── reports/                stakeholder_summary.md, figures/
└── Makefile, pyproject.toml, requirements.txt, requirements-dev.txt
```

## Caveats

- **Synthetic data with clean-cut fraud.** Model metrics are illustrative. The cost model's
  dollar inputs are planning assumptions, listed in `assumption_ledger.csv`.
- **Labels.** They are complete here. In production they are partial (confirmed cases plus review
  outcomes), which changes calibration and makes the rule-decay KPIs lag.
- **Structural sharing counts.** `device_n_users` and `ip_n_users` are whole-period counts by design,
  not strictly past-only.
- **SQL.** It has never run on BigQuery itself, only on a DuckDB mirror (see above).
