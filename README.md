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

## Open decision (flagged in-notebook)

Users with no matching rows in `transactions_v2.csv` or `user_logs_v2.csv` show up as nulls in the corresponding aggregated columns after the merge. This may be a meaningful signal (e.g. inactive users) rather than missing data. **Not yet resolved:** whether to impute, flag, or drop these before feature engineering — decide with the team and record the decision here.

## Requirements

- Python 3 (developed on 3.14)
- `pandas`, `numpy`

## Notes on reproducing

- File paths (`DATA_DIR`, `OUTPUT_DIR`) are hardcoded to a local path and will need to be updated to run elsewhere.
- `SAMPLE_SIZE` and `CHUNK_SIZE` are adjustable constants near the top of the relevant cells — increase `SAMPLE_SIZE` for a larger working set if memory allows.
