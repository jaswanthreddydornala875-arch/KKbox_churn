# EDA & Feature Engineering Summary — KKBox Churn

## Dataset
25,000 sampled users, 25 base columns from `master_merged.csv`. Churn rate: 9.0% / 91.0%.

## Data quality notes
- 2,915 users have no matching `members_v3` record (missing city/bd/registered_via/registration_init_time) — a join gap, not random missingness. Flagged as `has_member_record`.
- 958 users have no transaction history (`has_transactions=False`) — treated as inactive, not imputed.
- 5,638 users have no log/listening history (`has_logs=False`).
- Gender missing for 60% of users (expected — optional field in raw KKBox data).
- `bd_clean` (age) missing for ~60%, consistent with the source notebook's note that raw age data is unreliable.

## Key churn drivers (validated)
- `any_cancel`: 62.0% churn if cancelled vs. 7.2% if not — strongest single signal.
- `has_transactions`: 80.4% churn if no transaction history vs. 6.2% if present.
- `is_auto_renew`: 45.1% churn if off vs. 3.8% if on.
- `has_logs`: weak signal (9.3% vs. 8.9%) — presence of listening activity alone doesn't separate churners well.
- Among active users, churners show lower total engagement (`total_secs_sum`, `active_days`, `num_100_sum`) but similar per-session intensity (`total_secs_mean`, `num_unq_mean`).
- `days_since_last_transaction`: sharp cliff — almost no non-churned users have gaps beyond ~33 days; nearly everyone past that point has churned. Strong candidate feature.
- Age: churners skew slightly younger (peak ~22 vs ~27).

## Engineered features added
- `has_member_record`, `has_transactions`, `has_logs`, `is_age_missing` (boolean flags)
- `days_since_last_transaction`, `days_since_last_log` (recency, relative to 2017-03-31 cutoff)
- Transaction/log numeric nulls imputed to 0 for inactive users

## Output
`master_features.csv` (25000, 31) — ready for modeling. Categorical encoding (`city`, `gender`, `registered_via`, `last_payment_method`) still needed before modeling, per the source notebook's note.

## Recommendation for modeling
Prioritize `any_cancel`, `has_transactions`, `is_auto_renew`, and `days_since_last_transaction` as top features — they show the largest churn-rate separation by far.