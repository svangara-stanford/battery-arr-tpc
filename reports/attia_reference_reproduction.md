# Attia Reference Reproduction

status: `exact_author_model_replay`
validation_status: `skipped_batch9_zip_present`
validation_status_label: `validation_skipped_batch9_zip_present`

## Model Files
- oed_model.mat found: `True`
- oed_model_batch1.mat found: `True`
- all required model variables found: `True`

## Batch Mapping
- `2018-08-28_oed_0` -> `oed1`
- `2018-09-02_oed_1` -> `oed2`
- `2018-09-06_oed_2` -> `oed3`
- `2018-09-10_oed_3` -> `oed4`

## Batch Results
- `2018-08-28_oed_0`: model `/Users/sreyavangara/Documents/Research/Code/battery-arr/literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model_batch1.mat`, cutoff `98`, cells `46`, predictions `46`, unavailable `0`
- `2018-09-02_oed_1`: model `/Users/sreyavangara/Documents/Research/Code/battery-arr/literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `100`, cells `46`, predictions `46`, unavailable `0`
- `2018-09-06_oed_2`: model `/Users/sreyavangara/Documents/Research/Code/battery-arr/literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `100`, cells `48`, predictions `48`, unavailable `0`
- `2018-09-10_oed_3`: model `/Users/sreyavangara/Documents/Research/Code/battery-arr/literature_models_and_data/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `95`, cells `48`, predictions `48`, unavailable `0`

## Top BayesGap Recommended Protocols
- C1=8.0, C2=5.6, C3=5.6, C4=2.847
- C1=8.0, C2=6.0, C3=5.6, C4=2.754
- C1=8.0, C2=3.6, C3=5.6, C4=3.969
- C1=7.0, C2=3.6, C3=5.6, C4=4.271
- C1=8.0, C2=5.2, C3=5.6, C4=2.963
- C1=6.0, C2=3.6, C3=5.6, C4=4.755
- C1=8.0, C2=4.0, C3=5.6, C4=3.574
- C1=7.0, C2=7.0, C3=3.6, C4=3.706
- C1=8.0, C2=4.8, C3=5.6, C4=3.111
- C1=7.0, C2=6.0, C3=5.6, C4=2.897

## Caveats
- Batch 9 validation was skipped by default; no final validation ranking was written.
