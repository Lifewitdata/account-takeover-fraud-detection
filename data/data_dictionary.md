# Data dictionary — Account takeover & fraud-ring detection (synthetic)

All data is synthetic, generated with seed **42**. No real users.
Period: **2025-01-01 → 2025-12-31**.

## Files

| File | Rows | Grain |
|---|---|---|
| `users.csv` | 5,100 | one row per user |
| `devices.csv` | 7,150 | one row per device |
| `login_attempts.csv` | 162,778 | one row per login attempt |
| `password_resets.csv` | 483 | one row per reset request |
| `transactions.csv` | 61,560 | one row per transaction |

## Columns

**users.csv** — `user_id`, `signup_date`, `country`, `home_lat`, `home_lon`
(home location), `email_domain`, `is_verified` (0/1)

**devices.csv** — `device_id`, `device_type` (mobile/desktop/tablet),
`os`, `first_seen_date`

**login_attempts.csv** — `attempt_id`, `user_id`, `timestamp`, `ip_address`,
`device_id`, `ip_country`, `ip_city`, `ip_lat`, `ip_lon`,
`login_result` (success/failed), `failure_reason` (wrong_password/account_locked),
`is_fraud_attempt` (0/1)

**password_resets.csv** — `reset_id`, `user_id`, `request_timestamp`,
`complete_timestamp` (blank if not completed), `ip_address`, `device_id`,
`channel` (email/sms), `status` (completed/expired/failed), `is_fraud_reset` (0/1)

**transactions.csv** — `txn_id`, `user_id`, `timestamp`, `amount_usd`,
`merchant_category`, `device_id`, `ip_address`, `is_fraud_txn` (0/1)

## Injected fraud scenarios (ground truth)

1. **Credential stuffing** — 25 campaigns; bursts of failed logins from one
   IP/device across ~60 users each, ~12% of targeted accounts then breached.
2. **Account takeover** — 120 users; successful login from a distant country on
   a new device, followed by a password reset and 1–3 high-value transactions.
3. **Fraud rings** — 5 rings × 20 users; shared devices/IPs, signups within one
   week, bursts of logins and similar-amount `digital_goods` transactions.

## Label caveat

`is_fraud_attempt`, `is_fraud_reset`, `is_fraud_txn` are **synthetic ground-truth
labels for evaluation only** — never use them as model/scoring features.
In production these labels would be partial (confirmed cases + human review).
