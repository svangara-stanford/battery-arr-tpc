# Open Battery Agents Attia Surrogate Dataset

label_source: `author_model_prediction`
first_n_cycles: `100`
batch9_included: `False`
n_cells: `179`
n_cycle_rows: `17900`
charge_capacity_status: `available`
skipped_raw_cells: `7`
skipped_label_rows: `2`

## Important Caveat
The `cycle_life` labels for OED/CLO rows are author-model early predictions, not true final measured lifetimes.
Batch 9 is excluded from this training/search dataset and remains locked for final validation.

## Batches
- `2018-08-28_oed_0`: valid labels `46`, included `43`, exclusions `3`
- `2018-09-02_oed_1`: valid labels `45`, included `43`, exclusions `3`
- `2018-09-06_oed_2`: valid labels `47`, included `46`, exclusions `2`
- `2018-09-10_oed_3`: valid labels `48`, included `47`, exclusions `1`

## Exclusion Counts
- `label_filter`: `2`
- `raw_match`: `5`
- `raw_parse`: `2`
