# Attia Reference Reproduction

status: `exact_author_model_replay`
validation_status: `included_validation_batch`
validation_status_label: `validation_included_batch9`
author_model_target_transform_requested: `auto`

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
- `2018-08-28_oed_0`: model `/tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/BMS-autoanalysis/oed_model_batch1.mat`, cutoff `98`, raw files `48`, parsed `46`, exact features `46`, finite predictions `46`, anomalous `0`, BayesGap rows `46`, unavailable `0`
  excluded load-time cells: `2`
  - load exclusion cell `2018-08-28_oed_0_CH17_structure`, channel `17`: /tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/data/2018-08-28_oed_0/2018-08-28_oed_0_CH17_structure.json does not contain summary and cycles_interpolated arrays
  - load exclusion cell `2018-08-28_oed_0_CH27_structure`, channel `27`: /tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/data/2018-08-28_oed_0/2018-08-28_oed_0_CH27_structure.json does not contain summary.discharge_capacity
- `2018-09-02_oed_1`: model `/tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `100`, raw files `46`, parsed `46`, exact features `46`, finite predictions `46`, anomalous `1`, BayesGap rows `45`, unavailable `0`
  BayesGap exclusion reasons: `{'prediction_le_0': 1, 'anomalous_ci_width_gt_2000': 1}`
  - BayesGap exclusion cell `2018-09-02_oed_1_CH11`, channel `11`: prediction_le_0, anomalous_ci_width_gt_2000
- `2018-09-06_oed_2`: model `/tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `100`, raw files `48`, parsed `48`, exact features `48`, finite predictions `48`, anomalous `1`, BayesGap rows `47`, unavailable `0`
  BayesGap exclusion reasons: `{'prediction_le_0': 1, 'anomalous_ci_width_gt_2000': 1}`
  - BayesGap exclusion cell `2018-09-06_oed_2_CH11`, channel `11`: prediction_le_0, anomalous_ci_width_gt_2000
- `2018-09-10_oed_3`: model `/tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/BMS-autoanalysis/oed_model.mat`, cutoff `95`, raw files `48`, parsed `48`, exact features `48`, finite predictions `48`, anomalous `0`, BayesGap rows `48`, unavailable `0`

## Prediction Scale Diagnostics
- `2018-08-28_oed_0`: transform `log10_cycle_life`, raw median `2.849477839828951`, 10^raw median `707.1163565915431`, 1/(10^raw) median `0.0014142793607076255`, Prediction median `707.1163565915431`, CI width median `395.4685951505382`, CI width >2000 `0`, Prediction <=0 `0`
- `2018-09-02_oed_1`: transform `log10_cycle_life`, raw median `2.9471507564512174`, 10^raw median `885.4284738497561`, 1/(10^raw) median `0.0011294108903951595`, Prediction median `882.2903981534288`, CI width median `827.2989169766549`, CI width >2000 `1`, Prediction <=0 `1`
- `2018-09-06_oed_2`: transform `log10_cycle_life`, raw median `2.842427807867623`, 10^raw median `695.7170168256685`, 1/(10^raw) median `0.0014373979142625255`, Prediction median `698.9941622629677`, CI width median `542.5741995160975`, CI width >2000 `1`, Prediction <=0 `1`
- `2018-09-10_oed_3`: transform `log10_cycle_life`, raw median `2.9834452243742278`, 10^raw median `962.6084878229224`, 1/(10^raw) median `0.0010388652945693237`, Prediction median `962.6084878229224`, CI width median `911.8261422780247`, CI width >2000 `0`, Prediction <=0 `0`

## BayesGap Round Indexing
| round_idx | consumed_prediction_file | input_rows | output_file |
| --- | --- | ---: | --- |
| 0 | `None` | 0 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/round_0_next_batch.csv` |
| 1 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/early_predictions/2018-08-28_oed_0.csv` | 46 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/round_1_next_batch.csv` |
| 2 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/early_predictions/2018-09-02_oed_1.csv` | 45 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/round_2_next_batch.csv` |
| 3 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/early_predictions/2018-09-06_oed_2.csv` | 47 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/round_3_next_batch.csv` |
| 4 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/early_predictions/2018-09-10_oed_3.csv` | 48 | `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/round_4_next_batch.csv` |

## Next-Batch Acquisition Recommendations
- round 0: C1=4.8, C2=5.6, C3=5.2, C4=3.935
- round 0: C1=6.0, C2=4.4, C3=4.8, C4=4.328
- round 0: C1=5.6, C2=7.0, C3=4.8, C4=3.294
- round 0: C1=7.0, C2=5.2, C3=4.8, C4=3.45
- round 0: C1=5.6, C2=5.2, C3=4.4, C4=4.252
- round 0: C1=8.0, C2=7.0, C3=4.8, C4=2.8
- round 0: C1=5.6, C2=6.0, C3=5.2, C4=3.381
- round 0: C1=3.6, C2=6.0, C3=5.6, C4=4.755
- round 0: C1=6.0, C2=4.4, C3=5.2, C4=4.047
- round 0: C1=7.0, C2=5.2, C3=4.4, C4=3.691
- round 1: C1=4.4, C2=7.0, C3=4.0, C4=4.69
- round 1: C1=4.0, C2=7.0, C3=4.4, C4=4.69
- round 1: C1=4.4, C2=7.0, C3=4.4, C4=4.239
- round 1: C1=8.0, C2=7.0, C3=5.6, C4=2.585
- round 1: C1=4.8, C2=7.0, C3=4.0, C4=4.308
- round 1: C1=8.0, C2=7.0, C3=5.2, C4=2.68
- round 1: C1=4.8, C2=7.0, C3=4.4, C4=3.924
- round 1: C1=8.0, C2=7.0, C3=4.8, C4=2.8
- round 1: C1=4.4, C2=7.0, C3=4.8, C4=3.924
- round 1: C1=7.0, C2=7.0, C3=5.6, C4=2.71

## Final Posterior Mean Ranking
ranking_file: `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/final_posterior_ranking.csv`
- rank 1: C1=4.4, C2=5.6, C3=5.6, C4=4.017, mean=931.8171500000001, uncertainty=12.842649999999992
- rank 2: C1=4.0, C2=5.6, C3=5.6, C4=4.421, mean=928.6817, uncertainty=14.537699999999973
- rank 3: C1=4.4, C2=5.6, C3=5.2, C4=4.252, mean=925.98485, uncertainty=12.101849999999956
- rank 4: C1=4.0, C2=5.6, C3=5.2, C4=4.707, mean=920.53925, uncertainty=16.037649999999985
- rank 5: C1=4.8, C2=7.0, C3=4.4, C4=3.924, mean=916.3525, uncertainty=18.493300000000033
- rank 6: C1=4.8, C2=5.2, C3=5.2, C4=4.16, mean=912.83875, uncertainty=12.361350000000016
- rank 7: C1=5.6, C2=5.2, C3=4.0, C4=4.707, mean=910.2393999999999, uncertainty=30.04910000000001
- rank 8: C1=4.8, C2=5.2, C3=5.6, C4=3.935, mean=908.85265, uncertainty=13.582449999999994
- rank 9: C1=4.4, C2=5.2, C3=5.2, C4=4.516, mean=908.1030000000001, uncertainty=15.223599999999976
- rank 10: C1=4.0, C2=6.0, C3=5.6, C4=4.2, mean=907.6309, uncertainty=14.588200000000029

## Paper Top Protocol Sanity Check
check_file: `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/bayesgap/final_paper_top_protocol_check.csv`
final top three exactly match paper: `False`
- `4.8C-5.2C-5.2C-4.160C`: exists `True`, rank `6`, mean `912.83875`, uncertainty `12.361350000000016`
- `5.2C-5.2C-4.8C-4.160C`: exists `True`, rank `25`, mean `890.1992`, uncertainty `13.398400000000038`
- `4.4C-5.6C-5.2C-4.252C`: exists `True`, rank `3`, mean `925.98485`, uncertainty `12.101849999999956`
- Audit note: mismatch may reflect implementation differences, exclusion handling, raw-data parsing, or skipped validation; this report does not claim exact paper Figure 3 reproduction.

## Batch 9 Validation
validation_status: `included_validation_batch`
batch_path: `/tmp/tpc-vbwne/oba_attia_12506/battery-fast-charging/data/2019-01-24_batch9`
raw files discovered: `46`
cells parsed: `46`
excluded cells: `0`
protocols parsed: `10`
protocol-level rows used: `9`
validation_protocol_ranking: `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/validation_protocol_ranking.csv`
validation_metrics: `/tmp/tpc-vbwne/oba_attia_12506/runs/attia_reference_reproduction_with_validation/validation_metrics.json`
posterior vs observed: n `9`, Pearson `0.7809235211556285`, Kendall `0.5555555555555556`, RMSE `86.42652872646057`, MAE `73.94970925925928`
early prediction vs observed: n `10`, Pearson `0.9008195405175659`, Kendall `0.28888888888888886`, RMSE `61.76959433613852`, MAE `42.763330796116605`
validation protocol ranking preview:
- observed rank `1`: C1=nan, C2=nan, C3=nan, C4=nan, observed mean `915.0`, posterior mean `nan`, early mean `846.4553199560502`
- observed rank `2`: C1=4.4, C2=5.6, C3=5.2, C4=4.252, observed mean `909.3333333333334`, posterior mean `925.98485`, early mean `759.3948525988244`
- observed rank `3`: C1=6.0, C2=5.6, C3=4.4, C4=3.834, observed mean `901.2`, posterior mean `788.6295`, early mean `848.0686281489368`
- observed rank `4`: C1=7.0, C2=4.8, C3=4.8, C4=3.652, observed mean `896.8`, posterior mean `739.4359999999999`, early mean `862.8683208440114`
- observed rank `5`: C1=5.2, C2=5.2, C3=4.8, C4=4.16, observed mean `871.0`, posterior mean `890.1992`, early mean `871.6894560675471`
- observed rank `6`: C1=4.8, C2=5.2, C3=5.2, C4=4.16, observed mean `856.3333333333334`, posterior mean `912.83875`, early mean `871.2837034116427`
- observed rank `7`: C1=3.6, C2=6.0, C3=5.6, C4=4.755, observed mean `779.0`, posterior mean `900.63705`, early mean `858.2559806311758`
- observed rank `8`: C1=8.0, C2=4.4, C3=4.4, C4=3.94, observed mean `723.0`, posterior mean `672.79515`, early mean `744.7850947464524`
- observed rank `9`: C1=8.0, C2=6.0, C3=4.8, C4=3.0, observed mean `604.25`, posterior mean `670.4974`, early mean `603.2647425548093`
- observed rank `10`: C1=8.0, C2=7.0, C3=5.2, C4=2.68, observed mean `508.8`, posterior mean `573.9674500000001`, early mean `504.3790627930199`

## Caveats
- Final top three posterior-mean protocols do not exactly match the paper-reported CLO top protocols; this may reflect implementation differences, exclusion handling, raw-data parsing, or skipped validation.
