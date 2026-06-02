# Open Battery Agents Role Workflow

run_id: `feature_program_smoke`
split_mode: `random`
offline: `True`
iterations: `1`
candidates_per_iteration: `8`

## Role Sequence

DatasetProfiler, FeatureScientist, ModelArchitect, CodeGenerator, CodeReviewer, Evaluator, ScientistCritic

## Split

- train cells: 134
- validation cells: 45
- validation fraction: 0.25
- split seed: 0

Batch 9 was not used during surrogate search.

## Feature Programs

- include feature programs: `True`
- feature program mode: `table`
- recipe hint: `None`
- feature programs used: 3
- feature-program columns: 752
- feature-family counts: `{'true_curve_shape': 280, 'capacity_summary': 255, 'energy_summary': 244, 'resistance_summary': 122, 'thermal_summary': 122, 'capacity_summary_delta': 48, 'energy_summary_delta': 48, 'true_curve_difference': 42, 'resistance_summary_delta': 24, 'thermal_summary_delta': 24}`
- true raw curve features used: `True`
- proxy features used: `False`
- protocol features used: `False`

## Tool Calls

- profile_dataset success=True duration_ms=23.536708002211526
- build_battery_features success=True duration_ms=645.1402920065448
- review_candidate success=True duration_ms=911.768916004803
- evaluate_candidate success=True duration_ms=1082.9668340011267
- review_candidate success=True duration_ms=583.6997500009602
- evaluate_candidate success=True duration_ms=1016.0850410029525
- review_candidate success=True duration_ms=2011.2870830052998
- evaluate_candidate success=False duration_ms=30049.807624993264
- review_candidate success=True duration_ms=610.972334005055
- evaluate_candidate success=True duration_ms=1407.2044160129735
- review_candidate success=True duration_ms=768.9281670027412
- evaluate_candidate success=True duration_ms=3650.0060000107624
- review_candidate success=True duration_ms=547.7525839960435
- evaluate_candidate success=True duration_ms=2456.7570830113254
- review_candidate success=True duration_ms=544.4365419971291
- evaluate_candidate success=True duration_ms=21669.19187500025
- review_candidate success=True duration_ms=1719.3270419957116
- evaluate_candidate success=False duration_ms=30064.449457990122
- review_candidate success=True duration_ms=678.7238329998218
- evaluate_candidate success=True duration_ms=1540.914083001553
- review_candidate success=True duration_ms=782.8555840096669
- evaluate_candidate success=False duration_ms=30052.36920900643
- review_candidate success=True duration_ms=666.364375007106
- evaluate_candidate success=True duration_ms=1485.5595830013044
- profile_dataset success=True duration_ms=43.62599999876693
- build_battery_features success=True duration_ms=1131.6320420010015
- review_candidate success=True duration_ms=1533.8964580005268
- evaluate_candidate success=True duration_ms=1852.6416669919854
- review_candidate success=True duration_ms=969.8652920051245
- evaluate_candidate success=True duration_ms=1612.9013750032755
- review_candidate success=True duration_ms=3583.236250007758
- evaluate_candidate success=False duration_ms=30142.122917008237
- review_candidate success=True duration_ms=1133.0660829989938
- evaluate_candidate success=True duration_ms=2034.3098750017816
- review_candidate success=True duration_ms=1191.948792009498
- evaluate_candidate success=True duration_ms=6000.13079198834
- review_candidate success=True duration_ms=957.0250420074444
- evaluate_candidate success=True duration_ms=4365.362291995552
- review_candidate success=True duration_ms=890.4354999976931
- evaluate_candidate success=False duration_ms=30088.703624991467
- review_candidate success=True duration_ms=948.8721669913502
- evaluate_candidate success=True duration_ms=1833.0911669909256
- review_candidate success=True duration_ms=2317.925541996374
- evaluate_candidate success=False duration_ms=30066.742166003678
- review_candidate success=True duration_ms=912.3819999949774
- evaluate_candidate success=True duration_ms=1756.8150000006426
- review_candidate success=True duration_ms=1040.966666987515
- evaluate_candidate success=False duration_ms=30085.487541000475
- review_candidate success=True duration_ms=1414.3803750048392
- evaluate_candidate success=True duration_ms=1851.6129579948029

## Candidate

- best candidate id: `role_graph_iter_000_variant_03`
- best candidate path: `runs/open_battery_agents/feature_program_smoke/candidates/role_graph_iter_000_variant_03.py`
- best iteration: `0`
- final iteration candidate path: `runs/open_battery_agents/feature_program_smoke/candidates/role_graph_iter_000_repair_1.py`
- compiled candidate: `True`
- model family: `RandomForestRegressor`
- target transform: `raw`
- feature families: `capacity_cycles_2_10_100, max_minus_cycle2, cycleN_minus_cycle10, early_capacity_slope, late_capacity_slope, log_difference_proxies, protocol_features_disabled`
- include protocol features: `False`
- preprocessing: `drop_all_nan_columns, SimpleImputer(strategy='median'), StandardScaler`
- best review status: `pass`
- review verdict: `pass`

## Per-Iteration Candidates

| iteration | candidate_id | model_family | feature_set | target_transform | include_protocol_features | success | RMSE | MAE | R2 | y_pred_mean | y_true_mean | prediction_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | role_graph_iter_000_variant_00 | Ridge | scalar_only | raw | False | True | 200.9801236466243 | 161.82075869102877 | -0.2767143357740385 | 805.7117430990398 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_variant_00.csv` |
| 0 | role_graph_iter_000_variant_01 | Ridge | curve_only | log10 | False | True | 264.9374722614884 | 205.78924287787154 | -1.2185759136381873 | 836.7524401309641 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_variant_01.csv` |
| 0 | role_graph_iter_000_repair_1 | Ridge | all_available | raw | False | True | 277.2381114152017 | 226.89318514076007 | -1.4293684118288112 | 796.974612807237 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_repair_1.csv` |
| 0 | role_graph_iter_000_variant_03 | RandomForestRegressor | broad_physics | raw | False | True | 151.44427395330862 | 124.86844598578304 | 0.2750747420603723 | 806.9416530483612 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_variant_03.csv` |
| 0 | role_graph_iter_000_variant_04 | GradientBoostingRegressor | all_available | log10 | False | True | 200.38103170994742 | 152.85205453375892 | -0.269114287876423 | 755.2192838837847 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_variant_04.csv` |
| 0 | role_graph_iter_000_repair_1 | Ridge | all_available | raw | False | True | 277.2381114152017 | 226.89318514076007 | -1.4293684118288112 | 796.974612807237 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_repair_1.csv` |
| 0 | role_graph_iter_000_repair_1 | Ridge | all_available | raw | False | True | 277.2381114152017 | 226.89318514076007 | -1.4293684118288112 | 796.974612807237 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_repair_1.csv` |
| 0 | role_graph_iter_000_repair_1 | Ridge | all_available | raw | False | True | 277.2381114152017 | 226.89318514076007 | -1.4293684118288112 | 796.974612807237 | 859.9512439413174 | `runs/open_battery_agents/feature_program_smoke/artifacts/iteration_000/predictions_role_graph_iter_000_repair_1.csv` |

### Best By Feature Set

| feature_set | candidate_id | model_family | target_transform | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| all_available | role_graph_iter_000_variant_04 | GradientBoostingRegressor | log10 | 200.38103170994742 | 152.85205453375892 | -0.269114287876423 |
| broad_physics | role_graph_iter_000_variant_03 | RandomForestRegressor | raw | 151.44427395330862 | 124.86844598578304 | 0.2750747420603723 |
| curve_only | role_graph_iter_000_variant_01 | Ridge | log10 | 264.9374722614884 | 205.78924287787154 | -1.2185759136381873 |
| scalar_only | role_graph_iter_000_variant_00 | Ridge | raw | 200.9801236466243 | 161.82075869102877 | -0.2767143357740385 |

### Best By Target Transform

| target_transform | candidate_id | feature_set | model_family | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| log10 | role_graph_iter_000_variant_04 | all_available | GradientBoostingRegressor | 200.38103170994742 | 152.85205453375892 | -0.269114287876423 |
| raw | role_graph_iter_000_variant_03 | broad_physics | RandomForestRegressor | 151.44427395330862 | 124.86844598578304 | 0.2750747420603723 |

## Validation Metrics

- rmse: 151.44427395330862
- mae: 124.86844598578304
- r2: 0.2750747420603723
- spearman: 0.6523056653491436
- kendall: 0.4969696969696969

## Prediction Diagnostics

- y_true_mean: 859.9512439413174
- y_pred_mean: 806.9416530483612
- y_true_min/max: 558.6822900380835 / 1234.855720545823
- y_pred_min/max: 641.0032144203494 / 974.8278727136685
- residual_mean: -53.00959089295623
- residual_std: 141.86384806075932
- n_predictions: 45
- n_negative_predictions: 0
- n_nonfinite_predictions: 0

## Locked Validation Batch

- status: `not_run`
- role-agent surrogate validation RMSE: `151.44427395330862`
- role-agent locked Batch 9 RMSE: `None`
- Batch 9 weak baseline RMSE: `None`
- author/literature Batch 9 RMSE: `None`
- Battery-PGR on Batch 9: `None`
- predictions: `None`
- metrics: `None`

Batch 9 was used only after search for locked final validation.


## Critique

ScientistCritic summarized the candidate evaluation.

## Artifacts

- artifact index: `runs/open_battery_agents/feature_program_smoke/artifacts/index.json`
