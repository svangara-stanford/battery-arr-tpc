#!/usr/bin/env python3
from pathlib import Path

p = Path("src/battery_aar/workflows/role_prompts.py")
s = p.read_text()
orig = s
s = s.replace(
'''FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, broad-physics, and Attia/Severson-like feature sets.

Strong plans should consider author-inspired but coefficient-free feature families:
- discharge capacity at cycles 2, 10, and 100 when available
- maximum early capacity minus cycle-2 capacity
- cycle-N minus cycle-10 capacity differences
- early and late capacity slopes
- log-transformed difference-statistic proxies
- protocol-current features only when allowed

The literature early predictor modeled log10(cycle life). Feature plans can support either raw cycle-life
models or log10 target modeling; the ModelArchitect chooses the target transform.
''',
'''FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, and broad-physics feature sets.

Strong plans should reason from general battery degradation principles and choose among:
- early capacity level and retention
- capacity-fade slopes, curvature, variance, and early-window statistics
- resistance, thermal, energy, and efficiency proxies when available
- charge/discharge curve-shape statistics and cross-cycle curve deltas
- conservative protocol-current features only when explicitly allowed
- raw or transformed target modeling when statistically and physically justified

Do not assume a particular paper feature set, coefficient vector, target transform, or model family. The goal is to discover transferable early-life predictors from the available early-cycle data and validation feedback.
'''
)
s = s.replace(
'''Use robust preprocessing. Prefer Ridge, ElasticNet, ElasticNetCV, RandomForest, or GradientBoosting over neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
The literature early predictor modeled log10(cycle life). For positive lifetime targets, log10 target modeling
may improve stability under batch shifts. Choose target_transform as either "raw" or "log10" and justify the choice.
''',
'''Use robust preprocessing. Prefer small-data regressors such as Ridge, ElasticNetCV, LassoCV, RandomForest, or GradientBoosting before neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
Choose target_transform as either "raw" or "log10" and justify it from the target distribution, error structure, and transfer-stability considerations.
Do not assume a paper-specific target transform or model family.
'''
)
if s != orig:
    p.write_text(s)
    print(f"Patched generic Track A prompts in {p}")
else:
    print("No changes made; prompts may already be generic or source wording changed.")
