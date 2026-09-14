# KKBox Churn Prediction — Data Foundation & Preprocessing

This notebook (`01_data_foundation_preprocessing.ipynb`) builds the base dataset used for downstream churn-prediction modeling on the KKBox dataset. It loads the raw KKBox CSVs, draws a stratified sample of users, joins their transaction and listening-activity history, engineers a first pass of features, and writes a single merged table to disk.

## Input data

Expected under `DATA_DIR` (`/Users/jaswanth/KKbox/dataset`):

| File | Rows | Description |
|---|---|---|
| `train_v2.csv` | 970,960 | User IDs (`msno`) with churn label (`is_churn`) |
| `members_v3.csv` | 6,769,473 | User demographics: city, birthdate (`bd`), gender, registration info |
| `transactions_v2.csv` | large, read in chunks | Subscription/payment transaction history |
| `user_logs_v2.csv` | large, read in chunks | Daily listening activity logs |

## Pipeline steps

1. **Load & inspect** `train_v2` and `members_v3`; check shapes, dtypes, and null counts.
2. **Check label balance** — churn is a minority class (~9% positive), noted as important for metric choice later (accuracy will be misleading).
3. **Stratified sampling** — draw a fixed-size sample (`SAMPLE_SIZE = 25000`, `RANDOM_STATE = 42`) from `train_v2` via `groupby("is_churn").sample(...)` to preserve the churn ratio while keeping the dataset small enough to work with locally.
4. **Filter member IDs** to the sampled users and pull their rows from `members_v3`.
5. **Chunked filtering of large files** — `transactions_v2.csv` and `user_logs_v2.csv` are read in `CHUNK_SIZE = 500,000`-row chunks and filtered down to only the sampled `msno` set, to avoid loading the full files into memory.
6. **Clean `bd` (age)** — raw values include invalid entries (negative, 0, >1000). Values outside `[10, 90]` are set to null in a new `bd_clean` column. (Note: ~55% of sampled users end up null here — the field is unreliable/sparsely valid in this dataset.)
7. **Parse dates** — `registration_init_time`, `transaction_date`, `membership_expire_date`, and log `date` are converted from `YYYYMMDD` integers to proper datetimes.
8. **Aggregate transactions per user** (`transactions_agg`): count of transactions, most recent transaction/expiry dates, last plan price/amount paid/payment method/plan days, last auto-renew flag, and whether any transaction was a cancellation.
9. **Aggregate activity logs per user** (`logs_agg`): number of active days, first/last log date, total and mean listening seconds, mean unique tracks, and sums of full (`num_100`) and short (`num_25`) plays.
10. **Merge into a master table** — left-joins `train_sample` → `members_sample` → `transactions_agg` → `logs_agg` on `msno`, preserving all 25,000 sampled users and the original churn balance.
11. **Save output** to `OUTPUT_DIR` (`/Users/jaswanth/KKbox/dataset/processed/master_merged.csv`), shape `(25000, 25)`.

## Output

- **`processed/master_merged.csv`** — one row per sampled user with demographics, latest transaction summary, and activity summary, ready for feature engineering / EDA in a follow-up notebook.

## Null-handling (resolved)

Users with no matching rows in `transactions_v2.csv` or `user_logs_v2.csv` are treated as a signal (inactive/lapsed users), not a data-quality issue:

- `has_transactions` / `has_logs` boolean flags mark which users had matching rows.
- Transaction/log count, sum, and mean columns are imputed to `0` for users with no activity.
- Date columns (`last_transaction_date`, `first_log_date`, etc.) are left as `NaT` — filter on the `has_*` flags rather than nulls when using them.
- `bd_clean` keeps its own `is_age_missing` flag instead of being imputed.

## Leakage check

Added a check comparing the transaction/log activity window against the March 2017 churn-decision cutoff, and a flag for the fraction of transactions with an expiry date past that cutoff. Review the printed output before trusting date-derived features in modeling — if a non-trivial share of transactions expire after the label window, exclude them from aggregation.

## Data dictionary

The notebook now includes a data dictionary cell (column → source → meaning) right before the final save, so anyone doing EDA doesn't have to reverse-engineer column meaning from code.

## Requirements

- Python 3 (developed on 3.14)
- `pandas`, `numpy`

## Notes on reproducing

- File paths (`DATA_DIR`, `OUTPUT_DIR`) are hardcoded to a local path (`/Users/jaswanth/...`) — **each team member needs to update these two lines to their own local dataset path** before running.
- `SAMPLE_SIZE` and `CHUNK_SIZE` are adjustable constants near the top of the relevant cells — increase `SAMPLE_SIZE` for a larger working set if memory allows.
- All cell outputs have been cleared, since the added cells shift execution order — run the notebook top to bottom before handing it off.

## Still open for the modeling stage (not needed for EDA)

- Categorical codes (`city`, `gender`, `registered_via`, `last_payment_method`) are left unencoded — fine for EDA, but will need encoding before modeling.
- No train/validation split yet — this notebook produces one full sampled+merged file; add a stratified split as a separate step once the team moves from EDA to modeling.
