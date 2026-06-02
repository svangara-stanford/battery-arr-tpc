# Battery Feature Discovery Substrate

Open Battery Agents now supports an opt-in feature-discovery substrate that lets agents propose scientific feature programs without writing raw Pandas/Numpy feature-plumbing code.

The substrate is layered:

1. Raw Attia JSON payloads are canonicalized into long-form `cycles_df` and `summary_df` tables by `battery_aar.features.raw_cycles`.
2. Trusted feature operators in `battery_aar.features.operators` compute numeric features with provenance metadata.
3. `FeatureProgram` artifacts declare which operators to run and with what parameters.
4. `battery_aar.features.feature_programs` compiles programs into `feature_table.csv`, `feature_metadata.csv`, exclusions, and dataset cards.
5. `build_all_battery_features` can merge compiled feature-program tables when explicitly enabled.
6. The trusted candidate compiler can select scalar-only, curve-only, scalar-plus-curve, broad-physics, protocol-only, or all-available feature sets using feature metadata.

Default workflows are unchanged. Feature-program tables are only used when `--include-feature-programs` and a table/auto mode are provided.

## Recipes

Deterministic recipes live in `battery_aar.features.program_library`:

- `minimal_debug`
- `scalar_baseline`
- `curve_delta`
- `attia_severson_like`
- `broad_physics`

The Attia/Severson-like recipe is one recipe in the generic algebra. It uses generic cycle-scalar, window-statistic, and cross-cycle discharge-curve-delta operators; it does not use author model coefficients or saved model internals.

## Outputs

`scripts/build_battery_feature_program.py` writes:

- `feature_table.csv`
- `feature_metadata.csv`
- `feature_program.json`
- `feature_program_result.json`
- `exclusions.csv`
- `dataset_card.json`
- `dataset_card.md`

Each feature metadata row records the feature family, source, operator, cycle indices, step type, x/y signals, aggregation, transform, and whether it is a true raw-curve feature or a proxy feature.

## Batch 9

Batch 9 remains locked. Feature-program tables for Batch 9 may be built without labels and used only after surrogate search for final locked validation. The role workflow offsets Batch 9 row IDs before merging search and Batch 9 feature-program tables so candidate row alignment remains explicit.

## Tools

The existing native/FastAPI tool layer exposes:

- `/features/build` for feature table construction and feature-program table merging.
- `/features/programs` for listing trusted recipes and registered operators.

All tool calls are logged to `artifacts/tool_calls.jsonl`.
