"""Patch E4 — secondary-test routing tests.

Covers:
- `load_severson_b3_holdout` returns the expected DataFrame shape with b3 cells.
- Exclusion filtering by `exclude_cell_ids` drops matching cells.
- The orchestrator (`run_rediscovery`) routes to Severson b3 by default and to
  Attia Batch 9 when `--secondary-test-source=attia_batch9`.
- Legacy `batch9_path` arg still routes to Attia and emits a DeprecationWarning.
- Champion-selection report keys include both legacy `batch9_*` and new
  `secondary_test_*` aliases after running the orchestrator with the new flag.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from battery_aar.agents.attia_data_bridge import (
    build_agent_surrogate_dataset,
    load_severson_b3_holdout,
)
from battery_aar.agents.orchestrator import run_rediscovery
from battery_aar.paper_reproduction.paths import (
    SEVERSON_B3_BATCH_DATE,
    SEVERSON_MAT_DIR_NAME,
    severson_b3_mat_path,
)

# Reuse the existing toy-Attia helper (writes _structure.json files) and the
# toy-Severson MatR helper (writes scipy-style .mat files).
from tests.test_agent_attia_data_bridge import _make_toy_root_and_reference
from tests.severson_test_utils import write_toy_severson_mat_dir


def _stage_severson_b3_in_bfc(tmp_path: Path) -> Path:
    """Stage a toy Severson MatR dir under ``<bfc>/data/severson_2019_true_life_matr``.

    Returns the bfc root path so it can be passed into
    ``load_severson_b3_holdout(battery_fast_charging_root=...)``.
    """

    bfc_root = tmp_path / "battery-fast-charging"
    severson_dir = bfc_root / "data" / SEVERSON_MAT_DIR_NAME
    severson_dir.parent.mkdir(parents=True, exist_ok=True)
    write_toy_severson_mat_dir(severson_dir.parent, n_files=3, n_cells_per_file=2, n_cycles=120)
    # write_toy_severson_mat_dir writes into <base>/severson_matr; rename to the
    # expected directory name.
    staged = severson_dir.parent / "severson_matr"
    staged.rename(severson_dir)
    return bfc_root


def test_severson_b3_mat_path_uses_b3_batch_date(tmp_path):
    bfc_root = tmp_path / "battery-fast-charging"
    b3 = severson_b3_mat_path(bfc_root)
    assert SEVERSON_B3_BATCH_DATE in b3.name
    assert b3.parent.name == SEVERSON_MAT_DIR_NAME
    assert b3.parent.parent.name == "data"


def test_load_severson_b3_holdout_returns_expected_shape(tmp_path):
    bfc_root = _stage_severson_b3_in_bfc(tmp_path)
    meta, cycles, labels = load_severson_b3_holdout(
        battery_fast_charging_root=bfc_root, first_n_cycles=50
    )
    assert not meta.empty
    assert not cycles.empty
    assert not labels.empty
    # b3 only — should not contain cells from other source files.
    batch_ids = set(meta["batch_id"].astype(str).unique())
    assert batch_ids == {f"severson_{SEVERSON_B3_BATCH_DATE}"}
    # Row_id consistency.
    assert set(meta["row_id"]) == set(labels["row_id"])
    assert set(cycles["row_id"]).issubset(set(meta["row_id"]))
    # row_id starts at zero (caller is responsible for offsetting).
    assert int(meta["row_id"].min()) == 0
    # Cycles truncated to first_n_cycles.
    assert int(cycles["cycle_index"].max()) <= 50
    # Labels are finite cycle_life values.
    assert labels["y"].notna().all()
    assert (labels["y"] > 0).all()


def test_load_severson_b3_holdout_filters_exclude_cell_ids(tmp_path):
    bfc_root = _stage_severson_b3_in_bfc(tmp_path)
    meta_all, _, labels_all = load_severson_b3_holdout(
        battery_fast_charging_root=bfc_root, first_n_cycles=50
    )
    source_ids = meta_all["source_cell_id"].astype(str).tolist()
    assert len(source_ids) >= 2
    exclude = {source_ids[0]}
    meta_filtered, _, labels_filtered = load_severson_b3_holdout(
        battery_fast_charging_root=bfc_root,
        first_n_cycles=50,
        exclude_cell_ids=list(exclude),
    )
    kept = set(meta_filtered["source_cell_id"].astype(str))
    assert exclude.isdisjoint(kept)
    assert len(meta_filtered) == len(meta_all) - 1
    assert len(labels_filtered) == len(labels_all) - 1


def test_load_severson_b3_holdout_requires_root_or_mat_path(tmp_path):
    with pytest.raises(ValueError, match="battery_fast_charging_root or mat_path"):
        load_severson_b3_holdout(first_n_cycles=50)


def test_load_severson_b3_holdout_accepts_explicit_mat_path(tmp_path):
    bfc_root = _stage_severson_b3_in_bfc(tmp_path)
    mat = severson_b3_mat_path(bfc_root)
    meta, _, _ = load_severson_b3_holdout(mat_path=mat, first_n_cycles=40)
    assert not meta.empty


def test_orchestrator_routes_to_severson_b3_by_default(tmp_path):
    # Severson b3 training surrogate is independent of the holdout source —
    # we reuse the toy Attia surrogate dataset for the search corpus.
    attia_root, reference, _batch9 = _make_toy_root_and_reference(tmp_path)
    processed = tmp_path / "processed"
    build_agent_surrogate_dataset(attia_root, reference, processed, first_n_cycles=100)
    bfc_root = _stage_severson_b3_in_bfc(tmp_path)

    report = run_rediscovery(
        processed_dir=processed,
        reference_run=reference,
        out=tmp_path / "run_severson_b3",
        reports_dir=tmp_path / "reports_b3",
        agents=1,
        iterations=1,
        offline=True,
        require_real_data=True,
        seed=3,
        final_secondary_test_validation=True,
        battery_fast_charging_root=bfc_root,
        # Note: do NOT pass batch9_path — the default secondary_test_source is
        # severson_b3 and the orchestrator must resolve the b3 file via the bfc
        # root.
    )
    assert report["secondary_test_source"] == "severson_b3"
    assert report["locked_secondary_test_status"] == "locked_validation_ok"
    # Legacy mirror keys also present.
    assert report["batch9_status"] == "locked_validation_ok"
    # New key carries the same RMSE as the legacy one.
    assert report["locked_secondary_test_rmse"] == report["locked_batch9_rmse"]
    assert report["secondary_test_pgr"] == report["batch9_pgr"]
    # Predictions / metrics artifacts written.
    assert (tmp_path / "run_severson_b3" / "final_batch9_predictions.csv").exists()
    assert (tmp_path / "run_severson_b3" / "final_batch9_metrics.json").exists()


def test_orchestrator_routes_to_attia_when_opted_in(tmp_path):
    attia_root, reference, batch9 = _make_toy_root_and_reference(tmp_path)
    processed = tmp_path / "processed"
    build_agent_surrogate_dataset(attia_root, reference, processed, first_n_cycles=100)

    report = run_rediscovery(
        processed_dir=processed,
        reference_run=reference,
        out=tmp_path / "run_attia",
        reports_dir=tmp_path / "reports_attia",
        agents=1,
        iterations=1,
        offline=True,
        require_real_data=True,
        seed=4,
        final_secondary_test_validation=True,
        secondary_test_source="attia_batch9",
        secondary_test_path=batch9,
        battery_fast_charging_root=attia_root,
    )
    assert report["secondary_test_source"] == "attia_batch9"
    assert report["locked_secondary_test_status"] == "locked_validation_ok"
    assert report["batch9_status"] == "locked_validation_ok"


def test_legacy_batch9_path_arg_still_routes_to_attia(tmp_path):
    attia_root, reference, batch9 = _make_toy_root_and_reference(tmp_path)
    processed = tmp_path / "processed"
    build_agent_surrogate_dataset(attia_root, reference, processed, first_n_cycles=100)

    # Using only the deprecated batch9_path with the explicit attia source must
    # warn but still produce a successful locked test.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_rediscovery(
            processed_dir=processed,
            reference_run=reference,
            out=tmp_path / "run_legacy_batch9",
            reports_dir=tmp_path / "reports_legacy",
            agents=1,
            iterations=1,
            offline=True,
            require_real_data=True,
            seed=5,
            final_batch9_validation=True,
            secondary_test_source="attia_batch9",
            batch9_path=batch9,
            battery_fast_charging_root=attia_root,
        )
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning for --batch9-path"
    assert any("batch9-path" in str(w.message).lower() for w in deprecations)
    assert report["secondary_test_source"] == "attia_batch9"
    assert report["batch9_status"] == "locked_validation_ok"
    assert report["locked_secondary_test_status"] == "locked_validation_ok"


def test_champion_adjudicator_sees_both_metric_keys(tmp_path):
    """Smoke: simulate a champion-selection adjudication report.

    We don't actually run the LLM here; instead we synthesize a leaderboard CSV
    in the shape produced by ``score_candidates_on_severson_b3.py`` and assert
    that ``_join_shortlist_with_secondary_test`` propagates both legacy
    ``batch9_*`` keys and new ``secondary_test_*`` keys onto each shortlist
    entry. This is the contract the ChampionAdjudicator's LLM prompt reads.
    """

    # Avoid importing the script as a module (it loads role_graph at top level
    # which is heavy); reach for the helper by file path instead.
    import importlib.util

    script_path = Path("scripts/run_agent_champion_selection.py").resolve()
    spec = importlib.util.spec_from_file_location("_oba_champion_selection_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    leaderboard_csv = tmp_path / "secondary_test_leaderboard.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "source_run_id": "run_a",
                "secondary_test_rmse": 110.0,
                "secondary_test_mae": 80.0,
                "secondary_test_r2": 0.5,
                "secondary_test_pgr": 0.4,
                "secondary_test_weak_baseline_rmse": 200.0,
                "secondary_test_n_nonfinite_predictions": 0,
                "secondary_test_source": "severson_b3",
                "batch9_rmse": 110.0,
                "batch9_mae": 80.0,
                "batch9_r2": 0.5,
                "batch9_pgr": 0.4,
                "batch9_weak_baseline_rmse": 200.0,
                "batch9_n_nonfinite_predictions": 0,
            },
            {
                "candidate_id": "cand_b",
                "source_run_id": "run_b",
                "secondary_test_rmse": 95.0,
                "secondary_test_mae": 70.0,
                "secondary_test_r2": 0.55,
                "secondary_test_pgr": 0.45,
                "secondary_test_weak_baseline_rmse": 200.0,
                "secondary_test_n_nonfinite_predictions": 0,
                "secondary_test_source": "severson_b3",
                "batch9_rmse": 95.0,
                "batch9_mae": 70.0,
                "batch9_r2": 0.55,
                "batch9_pgr": 0.45,
                "batch9_weak_baseline_rmse": 200.0,
                "batch9_n_nonfinite_predictions": 0,
            },
        ]
    ).to_csv(leaderboard_csv, index=False)

    shortlist = [
        {"candidate_id": "cand_a", "rank_in_shortlist": 1, "surrogate_rmse": 130.0},
        {"candidate_id": "cand_b", "rank_in_shortlist": 2, "surrogate_rmse": 125.0},
    ]
    joined = module._join_shortlist_with_secondary_test(
        shortlist, leaderboard_csv, "test-criteria"
    )
    assert len(joined["shortlist"]) == 2
    entry = joined["shortlist"][0]
    # Both new and legacy keys must propagate so the LLM prompt sees both.
    assert "secondary_test_rmse" in entry
    assert "batch9_rmse" in entry
    assert entry["secondary_test_rmse"] == entry["batch9_rmse"] == 110.0
    assert entry["secondary_test_source"] == "severson_b3"

    # The deprecated alias must still work and return the same shape.
    joined_legacy = module._join_shortlist_with_batch9(
        shortlist, leaderboard_csv, "test-criteria"
    )
    assert json.dumps(joined_legacy, sort_keys=True, default=str) == json.dumps(
        joined, sort_keys=True, default=str
    )
