# Stakeholder summary — Account takeover & fraud-ring detection

**For:** Trust & Safety leadership · **Date:** October 2026 · **Status:** analysis complete, synthetic data

## Situation

We analyzed 162,778 login attempts from 5,100 users across 2025 (7.3% fraudulent) to find
which signals actually catch account takeover, credential stuffing, and fraud rings —
and what it would cost to act on them.

## What the data shows

- **Fraud is bursty.** 48 ten-minute windows show 20+ login attempts from a single IP
  across 5+ users each — the classic credential-stuffing shape. 35 devices are shared
  by 5+ users — the fraud-ring shape.
- **New devices are the strongest simple signal.** Logins from a first-seen device are
  fraudulent 14.4% of the time vs 6.3% on known devices (z = 31.0).
- **Impossible-travel checks barely fire.** They catch 0.5% of fraud — attackers strike
  after long idle gaps, so speed-based rules rarely trigger. Don't build the strategy
  around them.
- **Simple rules already catch most of it:** IP-velocity windows catch 77.8% of fraud
  attempts; shared-device flags catch 99.0%.
- **A risk model ranks the rest cleanly.** Tested on Oct–Dec 2025 data the model had
  never seen: precision is 100% down the ranked list to k = 2,000. At a 0.5 threshold it
  raises **2,195 alerts at 100% precision and 98.7% recall — about 24 human reviews per day.**

## Recommended actions

1. **Auto-block high-velocity IP windows** (5+ users / 10 min). ~78% of fraud, near-zero
   review cost. No human in the loop.
2. **Stand up a scored review queue** for the remainder, highest risk first. Budget
   **~24 reviews/day**; the queue stays high-precision to about 2,000 cases.
3. **Re-score device neighborhoods.** When one account on a shared device is confirmed
   bad, review every account on that device — rings share infrastructure.

## Risks & caveats

- Analysis is on **synthetic data with clean-cut fraud patterns** — real attacker behavior
  overlaps more with legitimate behavior, so expect lower precision. Thresholds are re-checked
  by a weekly monitoring job (rule precision decay, feature drift, reviewer load) rather than
  on a fixed re-tune calendar — see `outputs/monitoring/alerts.csv`.
- Fraud labels here are complete; production labels are partial, which affects calibration.
- False positives cost reviewer hours and user friction — monitor appeal/reversal rates
  after any auto-block goes live.

## Since v1: exposure, root cause and compliance

- **$93,476.68** of fraud in 2025, fully attributed (takeover $41.0k, rings $52.5k); total
  fraud cost **$81,967.51**, or **$683 per confirmed takeover**.
- Every takeover passed an unseen device with no step-up; 102 of 120 kept a live session after
  the password reset. Acting within 30 minutes of the first signal would have stopped 98.9% of
  takeover value — the loss is an enforcement gap, not a detection gap.
- Controls cost **$9,457** for the quarter (deduplicated), of which $2,560 is false-positive cost.
  Two rules are flagged for escalation: R08 (precision 0.19 → 0.02) and R03 (0.16 → 0.09).
- 24 obligations mapped India-first (RBI, PMLA/FIU-IND, DPDP, SEBI) across EU, UK, US and global
  standards; 17 have generated evidence, verified on every run.

## Bottom line

Block the obvious stuff automatically, queue the rest by risk score for ~24 reviews a day,
and treat shared devices as the ring signal. Full analysis, code, and dashboard:
`account-takeover-fraud-detection/`.
